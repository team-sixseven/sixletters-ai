"""Research-only full-recording Holter RR/HRV features; not clinically validated.

A duration/relative-change rule cannot establish normal beat type. HRV and DC
are estimates based on filtered RR intervals, never adjudicated NN intervals.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path

import neurokit2 as nk
import numpy as np
from Members.sang.model.config import resolve_data_path
from Members.sang.model.DL.dataloader.segmented_ecg import read_patient_split

CACHE_VERSION = 1


def select_development(split_path, wave_dir, reference_path, sample_size, seed):
    """Select only original train/validation records; never include test or augmented files."""
    rows = read_patient_split(Path(split_path))
    dev = [
        (pid, group, label)
        for pid, group, label in rows
        if group in {"train", "validation"} and not re.search(r"-\d+$", pid)
    ]
    if any(not re.fullmatch(r"P\d{4}", pid) for pid, _, _ in dev):
        raise ValueError("예상하지 못한 원본 환자 ID 형식")
    if not dev:
        raise ValueError("개발용 원본 기록이 없습니다.")
    if sample_size is not None:
        if type(sample_size) is not int or not 1 <= sample_size <= len(dev):
            raise ValueError("sample_size는 개발 기록 수 이내의 양의 정수 또는 null이어야 합니다.")
        indexes = sorted(
            np.random.default_rng(seed).choice(len(dev), size=sample_size, replace=False).tolist()
        )
        dev = [dev[i] for i in indexes]
    with Path(reference_path).open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    ref = {}
    for row in records:
        pid = row.get("patient_id", "").strip()
        if pid in ref:
            raise ValueError("리듬 참고 정보에 중복 ID가 있습니다.")
        ref[pid] = row
    output = []
    for pid, group, label in dev:
        path = Path(wave_dir) / f"{pid}.npy"
        if not path.is_file():
            raise FileNotFoundError("선택한 원본 파형이 없습니다. 데이터 경로를 확인하세요.")
        if pid not in ref:
            raise ValueError("선택한 기록의 리듬 참고 정보가 없습니다.")
        output.append(
            {
                "patient_id": pid,
                "split": group,
                "label": int(label),
                "wave_path": path,
                "reference": ref[pid],
            }
        )
    return output


def load_lead_metadata(path):
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row.get("file", "").strip()
            value = row.get("has_z", "").strip().lower()
            if not name or name in result or value not in {"true", "false"}:
                raise ValueError("유도 메타데이터가 잘못됐거나 중복됐습니다.")
            result[name] = [True, True, value == "true"]
    return result


def load_quality_index(path, records, sample_rate, segment_seconds):
    """Read only selected original records' 30s flags; require complete aligned coverage."""
    segment_samples = int(sample_rate) * int(segment_seconds)
    targets = {record["patient_id"]: record for record in records}
    quality = {}
    seen = {}
    for pid, record in targets.items():
        wave = np.load(record["wave_path"], mmap_mode="r", allow_pickle=False)
        if wave.ndim != 2 or wave.shape[1] != 3 or wave.dtype.kind not in "fiu":
            raise ValueError("파형은 샘플×XYZ 숫자 배열이어야 합니다.")
        n = wave.shape[0] // segment_samples
        if n == 0:
            raise ValueError("파형이 30초 미만입니다.")
        quality[pid] = np.zeros(n, dtype=bool)
        seen[pid] = np.zeros(n, dtype=bool)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"pid", "file", "window_idx", "start_idx", "exclude"}.issubset(
            reader.fieldnames or []
        ):
            raise ValueError("품질 인덱스 스키마가 다릅니다.")
        for row in reader:
            filename = (row.get("file") or "").strip()
            if not filename.endswith(".npy"):
                continue
            pid = filename[:-4]
            if pid not in targets:
                continue
            if (row.get("pid") or "").strip() != pid:
                raise ValueError("품질 인덱스의 환자 ID가 일치하지 않습니다.")
            i, start = int(row["window_idx"]), int(row["start_idx"])
            excluded = (row.get("exclude") or "").strip().lower()
            if (
                i < 0
                or i >= len(seen[pid])
                or start != i * segment_samples
                or excluded not in {"true", "false"}
                or seen[pid][i]
            ):
                raise ValueError("품질 인덱스의 위치·제외값·중복이 올바르지 않습니다.")
            seen[pid][i] = True
            quality[pid][i] = excluded == "false"
    if any(not flags.all() for flags in seen.values()):
        raise ValueError("선택된 파형에 대한 품질 행이 누락됐습니다.")
    return quality


def detect_peaks_by_lead(
    wave, allowed, sample_rate, segment_seconds, lead_mask, chunk_seconds=300, context_seconds=2
):
    """Detect peaks over all accepted segments using overlapped bounded chunks."""
    sr, window = int(sample_rate), int(sample_rate * segment_seconds)
    chunk = int(chunk_seconds * sr)
    context = int(context_seconds * sr)
    if chunk <= 0 or chunk % window or len(allowed) != wave.shape[0] // window:
        raise ValueError("청크·품질 인덱스 길이가 일치하지 않습니다.")
    if len(lead_mask) != 3 or not any(lead_mask):
        raise ValueError("측정된 유도가 없습니다.")
    n_full = len(allowed) * window
    candidates = []
    for lead, present in enumerate(lead_mask):
        if not present:
            continue
        lead_good = np.asarray(allowed, dtype=bool).copy()
        peak_parts = []
        failed_chunks = 0
        for start in range(0, n_full, chunk):
            end = min(start + chunk, n_full)
            first, last = start // window, end // window
            if not lead_good[first:last].any():
                continue
            a, b = max(0, start - context), min(n_full, end + context)
            signal = np.asarray(wave[a:b, lead], dtype=np.float64)
            if not np.isfinite(signal).all():
                lead_good[first:last] = False
                failed_chunks += 1
                continue
            try:
                found = (
                    np.asarray(
                        nk.ecg_findpeaks(signal, sampling_rate=sr, method="neurokit")[
                            "ECG_R_Peaks"
                        ],
                        dtype=np.int64,
                    )
                    + a
                )
            except (ValueError, IndexError, ZeroDivisionError, FloatingPointError):
                lead_good[first:last] = False
                failed_chunks += 1
                continue
            found = found[(found >= start) & (found < end)]
            found = found[lead_good[found // window]]
            if len(found):
                peak_parts.append(found)
        peaks = np.unique(np.concatenate(peak_parts)) if peak_parts else np.empty(0, dtype=np.int64)
        seg_counts = np.bincount(peaks // window, minlength=len(lead_good))
        # Exploratory detector quality: 30-second peak counts equivalent to 30–200 bpm.
        plausible_segments = lead_good & (seg_counts >= 15) & (seg_counts <= 100)
        segment_score = float(plausible_segments.sum() / max(1, int(lead_good.sum())))
        gaps = np.cumsum(~lead_good)[peaks // window] if len(peaks) else np.empty(0, dtype=int)
        intervals = np.diff(peaks) * (1000.0 / sr)
        same = gaps[1:] == gaps[:-1]
        rr_score = (
            float(np.mean((intervals[same] >= 300) & (intervals[same] <= 2000)))
            if same.any()
            else 0.0
        )
        candidates.append(
            {
                "lead": ["X", "Y", "Z"][lead],
                "peaks": peaks,
                "good": lead_good,
                "block_ids": gaps,
                "failed_chunks": failed_chunks,
                "score": [segment_score, rr_score],
                "plausible_segments": int(plausible_segments.sum()),
            }
        )
    if not candidates or max(len(c["peaks"]) for c in candidates) < 3:
        raise ValueError("유효 R파를 충분히 찾지 못했습니다.")
    # The score is only an automatic heuristic and must be checked against annotations.
    return max(candidates, key=lambda c: (c["score"][0], c["score"][1], len(c["peaks"])))


def filtered_rr(peaks, block_ids, sample_rate, min_rr_ms=300, max_rr_ms=2000, relative_limit=0.2):
    peaks = np.asarray(peaks, dtype=np.int64)
    blocks = np.asarray(block_ids)
    if len(peaks) != len(blocks) or len(peaks) < 3 or np.any(np.diff(peaks) <= 0):
        raise ValueError("R파 위치 또는 구간 정보가 잘못됐습니다.")
    rr = np.diff(peaks).astype(np.float64) * (1000.0 / sample_rate)
    same_block = blocks[1:] == blocks[:-1]
    base = same_block & (rr >= min_rr_ms) & (rr <= max_rr_ms)
    good = base.copy()
    comparable = base[1:] & base[:-1] & (blocks[1:-1] == blocks[:-2])
    relative = np.zeros(len(rr) - 1, dtype=np.float64)
    np.divide(np.abs(rr[1:] - rr[:-1]), rr[:-1], out=relative, where=rr[:-1] > 0)
    good[1:] &= ~comparable | (relative <= relative_limit)
    return rr, good, same_block


def prsa_dc(rr, good, same_block):
    """Four-point PRSA DC estimate, with no bridging rejected or excluded intervals."""
    rr, good, same_block = (
        np.asarray(rr),
        np.asarray(good, dtype=bool),
        np.asarray(same_block, dtype=bool),
    )
    if len(rr) < 4:
        return None, 0
    anchors = np.arange(2, len(rr) - 1)
    keep = (
        good[anchors - 2]
        & good[anchors - 1]
        & good[anchors]
        & good[anchors + 1]
        & same_block[anchors - 2]
        & same_block[anchors - 1]
        & same_block[anchors]
        & same_block[anchors + 1]
        & (rr[anchors] > rr[anchors - 1])
    )
    anchors = anchors[keep]
    if not len(anchors):
        return None, 0
    value = np.mean((rr[anchors] + rr[anchors + 1] - rr[anchors - 1] - rr[anchors - 2]) / 4.0)
    return float(value), len(anchors)


def interval_features(
    peaks,
    block_ids,
    sample_rate,
    rhythm,
    allowed_seconds,
    segment_seconds,
    lead,
    failed_chunks=0,
    min_rr_ms=300,
    max_rr_ms=2000,
    relative_limit=0.2,
    sdann_min_coverage=0.8,
    hourly_min_coverage=0.5,
    lead_plausible_segment_fraction=0.0,
):
    rr, good, same_block = filtered_rr(
        peaks, block_ids, sample_rate, min_rr_ms, max_rr_ms, relative_limit
    )
    eligible_count, accepted_count = int(same_block.sum()), int(good.sum())
    if accepted_count < 2:
        raise ValueError("유효 RR 간격이 부족합니다.")
    end_seconds = np.asarray(peaks[1:], dtype=np.float64) / sample_rate
    coverage_hours = float(rr[good].sum() / 3_600_000)

    def eligible_window_means(window_seconds, coverage):
        groups = (end_seconds // window_seconds).astype(np.int64)
        out = []
        for group in np.unique(groups[good]):
            selection = good & (groups == group)
            values = rr[selection]
            if values.sum() >= window_seconds * 1000 * coverage:
                out.append(float(values.mean()))
        return out

    hour_rr = eligible_window_means(3600, hourly_min_coverage)
    rhythm_value = str(rhythm).strip()
    rhythms = {
        "0": "sinus",
        "1": "permanent_atrial_fibrillation",
        "2": "atrial_flutter",
        "3": "paced",
    }
    rhythm_name = rhythms.get(rhythm_value, "unknown")
    hrv, dc, anchors = None, None, 0
    if rhythm_name == "sinus":
        nn = rr[good]
        successive = good[1:] & good[:-1] & same_block[1:] & same_block[:-1]
        diffs = np.diff(rr)[successive]
        means_5min = eligible_window_means(300, sdann_min_coverage)
        hrv = {
            "SDNN_ms": float(np.std(nn, ddof=1)) if len(nn) >= 2 else None,
            "SDANN_ms": float(np.std(means_5min, ddof=1)) if len(means_5min) >= 2 else None,
            "RMSSD_ms": float(np.sqrt(np.mean(diffs**2))) if len(diffs) else None,
            "pNN50_percent": float(100 * (np.abs(diffs) > 50).sum() / len(nn))
            if len(diffs)
            else None,
        }
        dc, anchors = prsa_dc(rr, good, same_block)
    return {
        "recording": {
            "analyzed_hours": round(allowed_seconds / 3600, 4),
            "accepted_interval_hours": round(coverage_hours, 4),
            "lead_used": lead,
            "sample_rate_hz": int(sample_rate),
            "segment_seconds": int(segment_seconds),
            "excluded_interval_percent": round(
                100 * (1 - accepted_count / max(1, eligible_count)), 3
            ),
            "detector_failed_chunks": int(failed_chunks),
            "holter_rhythm": rhythm_name,
            "method": "NeuroKit2 R peaks; heuristic filtered RR, normal beat type unverified",
        },
        "heart_rate_bpm": {
            "mean_from_filtered_rr": round(float(60000.0 / rr[good].mean()), 3),
            "hourly_min": round(min(60000.0 / x for x in hour_rr), 3) if hour_rr else None,
            "hourly_max": round(max(60000.0 / x for x in hour_rr), 3) if hour_rr else None,
        },
        "hrv_time_domain_estimates": hrv,
        "deceleration_capacity_estimate_ms": dc,
        "quality": {
            "eligible_rr_count": eligible_count,
            "accepted_rr_count": accepted_count,
            "consecutive_pair_count": int(
                (good[1:] & good[:-1] & same_block[1:] & same_block[:-1]).sum()
            ),
            "lead_plausible_segment_fraction": float(lead_plausible_segment_fraction),
            "hourly_windows_used": len(hour_rr),
            "prsa_anchor_count": anchors,
            "hrv_status": "experimental_rr_surrogate"
            if rhythm_name == "sinus"
            else "not_available_non_sinus_or_unknown",
            "reference_beat_annotations_available": False,
            "sdann_min_window_coverage_fraction": sdann_min_coverage,
            "hourly_min_window_coverage_fraction": hourly_min_coverage,
        },
    }


def validate_llm_features(features):
    """Reject extra fields, free text, non-finite values and inappropriate rhythm data."""
    if not isinstance(features, dict) or set(features) != {
        "recording",
        "heart_rate_bpm",
        "hrv_time_domain_estimates",
        "deceleration_capacity_estimate_ms",
        "quality",
    }:
        raise ValueError("LLM 입력 최상위 필드가 다릅니다.")
    recording, heart, quality = (
        features["recording"],
        features["heart_rate_bpm"],
        features["quality"],
    )
    rec_keys = {
        "analyzed_hours",
        "accepted_interval_hours",
        "lead_used",
        "sample_rate_hz",
        "segment_seconds",
        "excluded_interval_percent",
        "detector_failed_chunks",
        "holter_rhythm",
        "method",
    }
    quality_keys = {
        "eligible_rr_count",
        "accepted_rr_count",
        "consecutive_pair_count",
        "lead_plausible_segment_fraction",
        "hourly_windows_used",
        "prsa_anchor_count",
        "hrv_status",
        "reference_beat_annotations_available",
        "sdann_min_window_coverage_fraction",
        "hourly_min_window_coverage_fraction",
    }
    if (
        not isinstance(recording, dict)
        or set(recording) != rec_keys
        or not isinstance(heart, dict)
        or set(heart) != {"mean_from_filtered_rr", "hourly_min", "hourly_max"}
        or not isinstance(quality, dict)
        or set(quality) != quality_keys
    ):
        raise ValueError("LLM 입력 내부 필드가 다릅니다.")
    if recording["lead_used"] not in {"X", "Y", "Z"} or recording["holter_rhythm"] not in {
        "sinus",
        "permanent_atrial_fibrillation",
        "atrial_flutter",
        "paced",
        "unknown",
    }:
        raise ValueError("유도 또는 리듬 값이 다릅니다.")
    if (
        recording["method"]
        != "NeuroKit2 R peaks; heuristic filtered RR, normal beat type unverified"
    ):
        raise ValueError("계산 방법 표시가 다릅니다.")
    if quality["reference_beat_annotations_available"] is not False:
        raise ValueError("확인되지 않은 박동 판독 정보입니다.")
    expected_status = (
        "experimental_rr_surrogate"
        if recording["holter_rhythm"] == "sinus"
        else "not_available_non_sinus_or_unknown"
    )
    if quality["hrv_status"] != expected_status:
        raise ValueError("리듬과 HRV 상태가 일치하지 않습니다.")
    hrv = features["hrv_time_domain_estimates"]
    if hrv is not None and (
        not isinstance(hrv, dict)
        or set(hrv) != {"SDNN_ms", "SDANN_ms", "RMSSD_ms", "pNN50_percent"}
    ):
        raise ValueError("HRV 필드가 다릅니다.")
    if recording["holter_rhythm"] != "sinus" and (
        hrv is not None or features["deceleration_capacity_estimate_ms"] is not None
    ):
        raise ValueError("비동성 리듬의 HRV·감속 능력은 전송하지 않습니다.")

    def check_numbers(mapping, allowed_none=()):
        for name, value in mapping.items():
            if value is None and name in allowed_none:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("LLM 입력에 유한하지 않은 수치가 있습니다.")

    check_numbers(
        {k: v for k, v in recording.items() if k not in {"lead_used", "holter_rhythm", "method"}}
    )
    check_numbers(heart, {"hourly_min", "hourly_max"})
    check_numbers(
        {
            k: v
            for k, v in quality.items()
            if k not in {"hrv_status", "reference_beat_annotations_available"}
        }
    )
    if hrv is not None:
        check_numbers(hrv, set(hrv))
    if features["deceleration_capacity_estimate_ms"] is not None:
        check_numbers({"dc": features["deceleration_capacity_estimate_ms"]})
    return features


def cache_key(record, quality_digest, lead_mask, settings):
    path = Path(record["wave_path"])
    s = path.stat()
    obj = {
        "version": CACHE_VERSION,
        "file": str(path.resolve()),
        "size": s.st_size,
        "mtime_ns": s.st_mtime_ns,
        "ctime_ns": s.st_ctime_ns,
        "inode": s.st_ino,
        "quality_sha256": quality_digest,
        "lead_mask": lead_mask,
        "rhythm": record["reference"]["holter_rhythm"],
        "settings": settings,
        "neurokit2": importlib.metadata.version("neurokit2"),
        "numpy": np.__version__,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def private_cache_dir(repo):
    path = Path(repo) / "Members/sang/results/monorouter_holter/clinical_cache_v1"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise PermissionError("임상 특징 캐시 폴더 권한을 확인하세요.")
    return path


def read_feature_cache(directory, key):
    p = Path(directory) / f"{key}.json"
    if not p.exists():
        return None
    s = p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or stat.S_IMODE(s.st_mode) & 0o077:
        raise PermissionError("특징 캐시 파일 권한이 올바르지 않습니다.")
    try:
        data = json.loads(p.read_text())
        if data.get("cache_version") != CACHE_VERSION or data.get("key") != key:
            return None
        features = data.get("features")
        return validate_llm_features(features)
    except (ValueError, TypeError, AttributeError):
        return None


def write_feature_cache(directory, key, features):
    validate_llm_features(features)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                {"cache_version": CACHE_VERSION, "key": key, "features": features},
                stream,
                allow_nan=False,
                ensure_ascii=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, Path(directory) / f"{key}.json")
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_development(repo, basic, basic_path, reference_path, sample_size, seed, settings):
    """Prepare research features for development records, reusing per-record caches."""
    raw = basic["data"]["raw_ecg"]
    sample_rate, segment_seconds = int(raw["sample_rate_hz"]), int(raw["segment_seconds"])
    if (
        raw.get("source_metadata_confirmed") is not True
        or raw.get("quality_index_format") != "drive_segment_rpeaks"
    ):
        raise ValueError("원본 샘플링 정보와 30초 품질 인덱스 확인이 필요합니다.")
    if sample_rate != 200 or segment_seconds != 30:
        raise ValueError("이 실험은 확인된 200 Hz·30초 품질 인덱스 전용입니다.")
    split_path = resolve_data_path(raw["split_path"], basic_path)
    wave_dir = resolve_data_path(raw["wave_directory"], basic_path)
    quality_path = resolve_data_path(raw["quality_index_path"], basic_path)
    lead_path = resolve_data_path(raw["lead_metadata_path"], basic_path)
    selected = select_development(split_path, wave_dir, reference_path, sample_size, seed)
    lead_meta = load_lead_metadata(lead_path)
    if any(record["wave_path"].name not in lead_meta for record in selected):
        raise ValueError("선택된 파형의 유도 메타데이터가 없습니다.")
    digest = _sha256_file(quality_path)
    initial_stat = quality_path.stat()
    cache_dir = private_cache_dir(repo)
    keys = {
        r["patient_id"]: cache_key(r, digest, lead_meta[r["wave_path"].name], settings)
        for r in selected
    }
    cached = {
        r["patient_id"]: read_feature_cache(cache_dir, keys[r["patient_id"]]) for r in selected
    }
    missing = [r for r in selected if cached[r["patient_id"]] is None]
    quality = (
        load_quality_index(quality_path, missing, sample_rate, segment_seconds) if missing else {}
    )
    prepared = []
    for index, record in enumerate(selected, 1):
        pid = record["patient_id"]
        features = cached[pid]
        error = None
        if features is None:
            try:
                wave = np.load(record["wave_path"], mmap_mode="r", allow_pickle=False)
                chosen = detect_peaks_by_lead(
                    wave,
                    quality[pid],
                    sample_rate,
                    segment_seconds,
                    lead_meta[record["wave_path"].name],
                    settings["chunk_seconds"],
                    settings["context_seconds"],
                )
                features = interval_features(
                    chosen["peaks"],
                    chosen["block_ids"],
                    sample_rate,
                    record["reference"]["holter_rhythm"],
                    int(chosen["good"].sum()) * segment_seconds,
                    segment_seconds,
                    chosen["lead"],
                    failed_chunks=chosen["failed_chunks"],
                    min_rr_ms=settings["min_rr_ms"],
                    max_rr_ms=settings["max_rr_ms"],
                    relative_limit=settings["relative_limit"],
                    sdann_min_coverage=settings["sdann_min_coverage"],
                    hourly_min_coverage=settings["hourly_min_coverage"],
                    lead_plausible_segment_fraction=chosen["score"][0],
                )
                validate_llm_features(features)
                now = quality_path.stat()
                if (now.st_size, now.st_mtime_ns, now.st_ctime_ns, now.st_ino) != (
                    initial_stat.st_size,
                    initial_stat.st_mtime_ns,
                    initial_stat.st_ctime_ns,
                    initial_stat.st_ino,
                ):
                    raise RuntimeError("품질 인덱스가 처리 중 바뀌었습니다. 다시 실행하세요.")
                if (
                    cache_key(record, digest, lead_meta[record["wave_path"].name], settings)
                    != keys[pid]
                ):
                    raise RuntimeError(
                        "파형 또는 계산 코드가 처리 중 바뀌었습니다. 다시 실행하세요."
                    )
                write_feature_cache(cache_dir, keys[pid], features)
            except (ValueError, OSError, FloatingPointError, IndexError, TypeError) as exc:
                error = type(exc).__name__
                features = None
        prepared.append((pid, record["label"], features, error))
        print(
            f"특징 준비 {index}/{len(selected)}: "
            + ("캐시" if cached[pid] is not None else "계산" if features is not None else "실패"),
            flush=True,
        )
    return selected, prepared


def reference_quality_report(selected, prepared):
    """Compare local estimates with source HRV values; never send references to the LLM."""
    refs = {r["patient_id"]: r["reference"] for r in selected}
    keys = {
        "SDNN_ms": "sdnn_reference_ms",
        "SDANN_ms": "sdann_reference_ms",
        "RMSSD_ms": "rmssd_reference_ms",
        "pNN50_percent": "pnn50_reference_percent",
    }
    summary = {}
    for metric, column in keys.items():
        diffs, calculated, reference = [], [], []
        for pid, _, features, _ in prepared:
            if not features or not features["hrv_time_domain_estimates"]:
                continue
            val = features["hrv_time_domain_estimates"].get(metric)
            text = refs[pid].get(column, "").strip().replace(",", ".")
            try:
                ref = float(text)
            except ValueError:
                continue
            if val is None or not math.isfinite(ref) or ref == 999:
                continue
            diffs.append(abs(val - ref))
            calculated.append(val)
            reference.append(ref)
        summary[metric] = {
            "matched": len(diffs),
            "median_absolute_difference": float(np.median(diffs)) if diffs else None,
            "mean_absolute_difference": float(np.mean(diffs)) if diffs else None,
        }
    return summary
