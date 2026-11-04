"""Prepare quality-passing 30-second RR summaries; research only.

The manifest contains patient_id, window_idx. Output is one JSONL per patient;
completed files are atomic and can be resumed after interruption.
"""

import argparse
import concurrent.futures
import csv
import json
import os
import time
from pathlib import Path

import neurokit2 as nk
import numpy as np

RATE = 200
SIZE = 30 * RATE


def patient_task(task):
    pid, indices, lead, wave_dir, output, limit = task
    output = Path(output)
    final = output / f"{pid}.jsonl"
    if final.exists():
        return ("already_complete", len(indices))
    partial = output / f".{pid}.{os.getpid()}.partial"
    wave = np.load(Path(wave_dir) / f"{pid}.npy", mmap_mode="r", allow_pickle=False)
    if wave.ndim != 2 or wave.shape[1] != 3 or lead not in "XYZ":
        raise ValueError("unsupported waveform or lead")
    rows = indices[:limit] if limit else indices
    count = 0
    with partial.open("x", encoding="utf-8") as f:
        for idx in rows:
            record = {"window_idx": idx, "status": "error"}
            try:
                samples = np.asarray(
                    wave[idx * SIZE : (idx + 1) * SIZE, "XYZ".index(lead)], dtype=np.float64
                )
                if len(samples) != SIZE or not np.isfinite(samples).all():
                    raise ValueError("invalid waveform")
                clean = nk.ecg_clean(samples, sampling_rate=RATE, method="neurokit")
                _, peaks = nk.ecg_peaks(clean, sampling_rate=RATE, method="neurokit")
                r = np.asarray(peaks["ECG_R_Peaks"], dtype=np.int64)
                rr = np.diff(r) * 1000.0 / RATE
                valid = (rr >= 300) & (rr <= 2000)
                if len(rr) < 2 or valid.sum() < 2 or valid.mean() < 0.5:
                    raise ValueError("insufficient valid RR")
                good = rr[valid]
                diff = np.diff(rr)[valid[1:] & valid[:-1]]
                record.update(
                    status="ok",
                    features={
                        "duration_seconds": 30,
                        "sample_rate_hz": RATE,
                        "lead": lead,
                        "r_peak_count": len(r),
                        "rr_valid_count": int(valid.sum()),
                        "rr_mean_ms": round(float(good.mean()), 2),
                        "rr_sd_ms": round(float(good.std(ddof=1)), 2),
                        "rr_rmssd_ms": round(float(np.sqrt(np.mean(diff**2))), 2)
                        if len(diff)
                        else None,
                        "mean_heart_rate_bpm": round(float(60000 / good.mean()), 2),
                        "note": "R peak detection and normal beat class are not validated; no QT or rhythm diagnosis",
                    },
                )
            except (ValueError, IndexError, TypeError, RuntimeError, FloatingPointError) as exc:
                record["error_type"] = type(exc).__name__
            f.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
            count += 1
        f.flush()
        os.fsync(f.fileno())
    os.replace(partial, final)
    return ("complete", count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--wave-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patients", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    if a.workers < 1 or a.limit < 0 or a.patients < 0:
        p.error("invalid limits")
    cohort = json.loads(a.input.read_text())["patients"]
    if len(cohort) != 356 or len({r["patient_id"] for r in cohort}) != 356:
        raise ValueError("cohort mismatch")
    indices = {r["patient_id"]: [] for r in cohort}
    with a.manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            pid, idx = row["patient_id"], int(row["window_idx"])
            if pid not in indices or idx < 0 or (indices[pid] and idx <= indices[pid][-1]):
                raise ValueError("manifest mismatch or unordered/duplicate windows")
            indices[pid].append(idx)
    if any(not values for values in indices.values()):
        raise ValueError("missing cohort patient in manifest")
    a.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    selected = cohort[: a.patients] if a.patients else cohort
    tasks = [
        (
            r["patient_id"],
            indices[r["patient_id"]],
            r["holter"]["recording"]["lead_used"],
            str(a.wave_dir),
            str(a.output),
            a.limit,
        )
        for r in selected
    ]
    t = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, result in enumerate(pool.map(patient_task, tasks), 1):
            print(
                "patient",
                i,
                "/",
                len(tasks),
                result[0],
                "windows",
                result[1],
                "elapsed_s",
                round(time.monotonic() - t, 1),
                flush=True,
            )


if __name__ == "__main__":
    main()
