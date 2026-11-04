"""Local-only, read-only Holter waveform viewer; no clinical interpretation."""

import argparse
import csv
import functools
import json
import math
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from Members.sang.model.config import load_basic_config, resolve_data_path  # noqa: E402
from Members.sang.model.DL.dataloader.segmented_ecg import read_patient_split  # noqa: E402

RATE = 200
SEGMENT_SAMPLES = RATE * 30
VIEW_SECONDS = 30
BASE = Path(__file__).parent


class HolterStore:
    def __init__(self, basic_path, segment_results=None):
        basic = load_basic_config(basic_path)
        raw = basic["data"]["raw_ecg"]
        if int(raw["sample_rate_hz"]) != RATE or int(raw["segment_seconds"]) != 30:
            raise ValueError("Viewer requires configured 200 Hz and 30-second quality windows")
        self.wave_dir = resolve_data_path(raw["wave_directory"], basic_path)
        self.quality_path = resolve_data_path(raw["quality_index_path"], basic_path)
        self.segment_results = segment_results
        lead_file = resolve_data_path(raw["lead_metadata_path"], basic_path)
        split_file = resolve_data_path(raw["split_path"], basic_path)
        self.records = {
            pid: group
            for pid, group, _ in read_patient_split(split_file)
            if re.fullmatch(r"P\d{4}", pid) and (self.wave_dir / f"{pid}.npy").is_file()
        }
        self.leads = {}
        with lead_file.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                name, has_z = row["file"].strip(), row["has_z"].strip().lower()
                if name in self.leads or has_z not in {"true", "false"}:
                    raise ValueError("Invalid or duplicate lead metadata")
                self.leads[name] = ("X", "Y", "Z") if has_z == "true" else ("X", "Y")

    def check_record(self, pid):
        if pid not in self.records or not re.fullmatch(r"P\d{4}", pid):
            raise ValueError("Unknown record")
        if f"{pid}.npy" not in self.leads:
            raise ValueError("Lead metadata unavailable")

    def wave(self, pid):
        self.check_record(pid)
        wave = np.load(self.wave_dir / f"{pid}.npy", mmap_mode="r", allow_pickle=False)
        if (
            wave.ndim != 2
            or wave.shape[1] != 3
            or wave.dtype.kind not in "fiu"
            or len(wave) < RATE * VIEW_SECONDS
        ):
            raise ValueError("Unsupported waveform")
        return wave

    @functools.lru_cache(maxsize=16)  # noqa: B019
    def quality(self, pid):
        wave = self.wave(pid)
        count = len(wave) // SEGMENT_SAMPLES
        flags = [None] * count
        with self.quality_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not {"pid", "file", "window_idx", "start_idx", "exclude"}.issubset(
                reader.fieldnames or []
            ):
                raise ValueError("Quality CSV schema mismatch")
            for row in reader:
                if row["file"].strip() != f"{pid}.npy":
                    continue
                index, start = int(row["window_idx"]), int(row["start_idx"])
                flag = row["exclude"].strip().lower()
                if (
                    row["pid"].strip() != pid
                    or not 0 <= index < count
                    or start != index * SEGMENT_SAMPLES
                    or flag not in {"true", "false"}
                    or flags[index] is not None
                ):
                    raise ValueError("Quality record mismatch")
                flags[index] = flag == "true"
        return tuple(flags)

    def record(self, pid):
        wave = self.wave(pid)
        return {
            "id": pid,
            "split": self.records[pid],
            "sample_rate": RATE,
            "samples": len(wave),
            "duration_seconds": len(wave) / RATE,
            "leads": self.leads[f"{pid}.npy"],
            "quality_30s_excluded": self.quality(pid),
            "start_time": None,
        }

    def window(self, pid, start, lead):
        wave = self.wave(pid)
        if lead not in self.leads[f"{pid}.npy"]:
            raise ValueError("Lead unavailable")
        if not math.isfinite(start) or start < 0:
            raise ValueError("Invalid start")
        offset = int(round(start * RATE))
        end = offset + VIEW_SECONDS * RATE
        if end > len(wave):
            raise ValueError("30-second window exceeds recording")
        values = np.asarray(wave[offset:end, "XYZ".index(lead)], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Waveform contains non-finite samples")
        flags = self.quality(pid)
        begin_i, end_i = offset // SEGMENT_SAMPLES, (end - 1) // SEGMENT_SAMPLES
        overlapping = [flags[i] if i < len(flags) else None for i in range(begin_i, end_i + 1)]
        return {
            "id": pid,
            "lead": lead,
            "start_seconds": offset / RATE,
            "sample_rate": RATE,
            "samples": values.tolist(),
            "quality_excluded": overlapping,
        }

    def segment_scores(self, pid, start, lead):
        self.check_record(pid)
        if self.segment_results is None:
            return {"status": "unconfigured", "results": []}
        selection = json.loads((self.segment_results / "selection.json").read_text())
        same = (
            pid == selection["patient_id"]
            and lead == selection["lead"]
            and math.isfinite(start)
            and abs(start - selection["start_seconds"]) < 0.0025
        )
        if not same:
            return {"status": "different_segment", "results": []}
        scores = self.segment_results / "model_scores.csv"
        metadata = self.segment_results / "metadata.json"
        if not scores.exists() or not metadata.exists():
            return {"status": "running", "results": []}
        with scores.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        model_ids = {"gpt_oss", "qwen32", "qwen_moe", "exaone"}
        if len(rows) > 4 or any(row.get("model") not in model_ids for row in rows):
            raise ValueError("Unexpected result schema")
        results = []
        for row in rows:
            score = float(row["score"]) if row.get("status") == "ok" and row.get("score") else None
            if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("Score outside expected bounds")
            results.append({"model": row["model"], "score": score, "status": row["status"]})
        return {
            "status": json.loads(metadata.read_text()).get("status", "UNKNOWN"),
            "results": results,
        }


def experimental_qt(window):
    """Return candidate fiducials only; never classify QT or SCD risk."""
    if any(flag is not False for flag in window["quality_excluded"]):
        return {
            "status": "unavailable",
            "reason": "구간에 제외 또는 품질 미확인 부분이 있습니다.",
            "beats": [],
        }
    import neurokit2 as nk

    signal = np.asarray(window["samples"], dtype=np.float64)
    try:
        cleaned = nk.ecg_clean(signal, sampling_rate=RATE, method="neurokit")
        _, peaks = nk.ecg_peaks(cleaned, sampling_rate=RATE, method="neurokit")
        r = np.asarray(peaks["ECG_R_Peaks"], dtype=int)
        if len(r) < 3:
            return {"status": "unavailable", "reason": "R파 후보가 부족합니다.", "beats": []}
        _, waves = nk.ecg_delineate(cleaned, r, sampling_rate=RATE, method="dwt", check=False)
        onsets, offsets = waves.get("ECG_R_Onsets", []), waves.get("ECG_T_Offsets", [])
        if len(onsets) != len(r) or len(offsets) != len(r):
            return {
                "status": "unavailable",
                "reason": "파형 경계 배열이 일치하지 않습니다.",
                "beats": [],
            }
        beats = []
        for i, (rp, onset, offset) in enumerate(zip(r, onsets, offsets, strict=False)):
            if not (np.isfinite(onset) and np.isfinite(offset)):
                continue
            onset, offset = int(onset), int(offset)
            next_r = int(r[i + 1]) if i + 1 < len(r) else len(signal)
            if not (RATE // 2 <= onset < rp < offset < min(next_r, len(signal) - RATE // 2)):
                continue
            beats.append(
                {
                    "qrs_start_s": onset / RATE,
                    "t_end_s": offset / RATE,
                    "r_peak_s": int(rp) / RATE,
                    "qt_ms": (offset - onset) * 1000 / RATE,
                }
            )
        return {
            "status": "experimental" if beats else "unavailable",
            "reason": "검증되지 않은 경계 후보이며 QT 연장·위험 판정이 아닙니다."
            if beats
            else "유효한 경계 후보를 찾지 못했습니다.",
            "beats": beats,
            "method": f"NeuroKit2 {nk.__version__}, ecg_clean/neurokit + dwt",
        }
    except (ValueError, IndexError, TypeError, RuntimeError, ZeroDivisionError, FloatingPointError):
        return {
            "status": "unavailable",
            "reason": "파형 경계 후보를 계산할 수 없습니다.",
            "beats": [],
        }


def segment_features(window):
    """Compute limited RR-based inputs for a single selected 30s; no diagnosis."""
    if len(window["samples"]) != RATE * VIEW_SECONDS or any(
        flag is not False for flag in window["quality_excluded"]
    ):
        raise ValueError("30초 구간의 품질이 통과되지 않았습니다.")
    import neurokit2 as nk

    x = np.asarray(window["samples"], dtype=np.float64)
    cleaned = nk.ecg_clean(x, sampling_rate=RATE, method="neurokit")
    _, peak_info = nk.ecg_peaks(cleaned, sampling_rate=RATE, method="neurokit")
    peaks = np.asarray(peak_info["ECG_R_Peaks"], dtype=np.int64)
    rr = np.diff(peaks) * 1000.0 / RATE
    valid = (rr >= 300) & (rr <= 2000)
    if len(rr) < 2 or valid.sum() < 2 or valid.mean() < 0.5:
        raise ValueError("30초 구간의 유효 RR 간격이 부족합니다.")
    good = rr[valid]
    successive = valid[1:] & valid[:-1]
    diff = np.diff(rr)[successive]
    return {
        "duration_seconds": VIEW_SECONDS,
        "sample_rate_hz": RATE,
        "lead": window["lead"],
        "r_peak_count": len(peaks),
        "rr_valid_count": int(valid.sum()),
        "rr_mean_ms": round(float(good.mean()), 2),
        "rr_sd_ms": round(float(good.std(ddof=1)), 2) if len(good) > 1 else None,
        "rr_rmssd_ms": round(float(np.sqrt(np.mean(diff**2))), 2) if len(diff) else None,
        "mean_heart_rate_bpm": round(float(60000 / good.mean()), 2),
        "note": "R peak detection and normal beat class are not validated; no QT or rhythm diagnosis",
    }


def handler_for(store, enable_qt=False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # Do not record patient IDs or waveform requests.

        def send_content(self, status, data, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            host = self.headers.get("Host", "").split(":")[0]
            if host not in {"localhost", "127.0.0.1"}:
                self.send_content(403, b"Forbidden", "text/plain")
                return
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self.send_content(
                        200, (BASE / "index.html").read_bytes(), "text/html; charset=utf-8"
                    )
                    return
                if parsed.path == "/api/records":
                    data = {
                        "records": [
                            {"id": pid, "split": split}
                            for pid, split in sorted(store.records.items())
                        ],
                        "experimental_qt_enabled": enable_qt,
                    }
                elif parsed.path == "/api/record":
                    data = store.record(query["id"][0])
                elif parsed.path == "/api/segment-scores":
                    data = store.segment_scores(
                        query["id"][0], float(query["start"][0]), query["lead"][0]
                    )
                elif parsed.path in {"/api/window", "/api/qt"}:
                    if parsed.path == "/api/qt" and not enable_qt:
                        self.send_content(
                            403, b'{"error":"Experimental QT is disabled"}', "application/json"
                        )
                        return
                    result = store.window(
                        query["id"][0], float(query["start"][0]), query["lead"][0]
                    )
                    data = experimental_qt(result) if parsed.path == "/api/qt" else result
                else:
                    self.send_content(404, b"Not Found", "text/plain")
                    return
                self.send_content(
                    200,
                    json.dumps(data, ensure_ascii=False, allow_nan=False).encode(),
                    "application/json; charset=utf-8",
                )
            except (ValueError, KeyError, IndexError, OSError):
                self.send_content(
                    400, b'{"error":"Invalid request or source data"}', "application/json"
                )

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--experimental-qt",
        action="store_true",
        help="Enable unvalidated QT boundary research only",
    )
    parser.add_argument(
        "--segment-results",
        type=Path,
        help="Local result directory for one preselected 30s segment",
    )
    parser.add_argument("--basic", type=Path, default=REPO / "Members/sang/model/basic.yaml")
    args = parser.parse_args()
    store = HolterStore(
        args.basic.resolve(), args.segment_results.resolve() if args.segment_results else None
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(store, args.experimental_qt))
    print(
        f"Local Holter viewer: http://127.0.0.1:{args.port}/ (records: {len(store.records)})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
