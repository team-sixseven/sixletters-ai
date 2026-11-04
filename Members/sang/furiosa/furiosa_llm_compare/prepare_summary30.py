"""Build patient-level research summaries and review candidates from all 30s windows."""

import argparse
import collections
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source = json.loads(a.input.read_text())
    patients = source["patients"]
    by = {r["patient_id"]: r for r in patients}
    if len(patients) != 356 or len(by) != 356:
        raise ValueError("cohort mismatch")
    indices = {pid: [] for pid in by}
    with a.manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            indices[row["patient_id"]].append(int(row["window_idx"]))
    a.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    summaries, candidates = {}, {}
    failure_counts = collections.Counter()
    for pid, rec in by.items():
        with (a.features / f"{pid}.jsonl").open() as f:
            rows = [json.loads(line) for line in f]
        if [r["window_idx"] for r in rows] != indices[pid]:
            raise ValueError("feature / quality manifest mismatch")
        valid = [r for r in rows if r["status"] == "ok"]
        failure_counts.update(r.get("error_type", "unknown") for r in rows if r["status"] != "ok")
        if not valid:
            raise ValueError("patient has no valid features")
        stats = {}
        for name in ("mean_heart_rate_bpm", "rr_mean_ms", "rr_sd_ms", "rr_rmssd_ms"):
            values = np.array(
                [r["features"][name] for r in valid if r["features"].get(name) is not None],
                dtype=float,
            )
            if not np.isfinite(values).all():
                raise ValueError("non-finite feature")
            stats[name] = (
                {
                    "median": round(float(np.median(values)), 2),
                    "p05": round(float(np.quantile(values, 0.05)), 2),
                    "p95": round(float(np.quantile(values, 0.95)), 2),
                }
                if len(values)
                else None
            )
        summaries[pid] = {
            "recording": rec["holter"]["recording"],
            "quality_accepted_windows": len(rows),
            "features_succeeded_windows": len(valid),
            "features_failed_windows": len(rows) - len(valid),
            "window_seconds": 30,
            "window_statistics": stats,
            "note": "Each statistic summarizes 30-second RR estimates. RR standard deviation per window is not 24-hour SDNN; normal beat classification and R peaks are not validated.",
        }
        # Descriptive extrema only: do not imply diagnosis or patient-level model attribution.
        chosen = []
        seen = set()
        orders = [
            ("lowest_window_mean_heart_rate", "mean_heart_rate_bpm", False),
            ("highest_window_mean_heart_rate", "mean_heart_rate_bpm", True),
            ("highest_window_rr_sd", "rr_sd_ms", True),
        ]
        for reason, field, descending in orders:
            eligible = [r for r in valid if r["features"].get(field) is not None]
            eligible.sort(key=lambda r: r["features"][field], reverse=descending)
            r = next((r for r in eligible if r["window_idx"] not in seen), None)
            if r:
                seen.add(r["window_idx"])
                chosen.append(
                    {
                        "start_seconds": 30 * r["window_idx"],
                        "duration_seconds": 30,
                        "lead": r["features"]["lead"],
                        "reason": reason,
                        "value": r["features"][field],
                        "feature": field,
                        "status": "unvalidated_review_candidate",
                    }
                )
        candidates[pid] = {
            "patient_id": pid,
            "segments": chosen,
            "note": "Descriptive extremes; not SCD risk, diagnosis, or model attribution.",
        }
    result = copy.deepcopy(source)
    result["metadata"]["summary30"] = {
        "method": "median, p05 and p95 across all quality-passing 30-second RR windows",
        "feature_errors": dict(failure_counts),
        "research_only": True,
        "input_sha256": hashlib.sha256(a.input.read_bytes()).hexdigest(),
    }
    for rec in result["patients"]:
        rec["holter"] = summaries[rec["patient_id"]]
        for ex, donor in zip(rec["examples"], rec["example_patient_ids"], strict=False):
            if (
                donor not in by
                or by[donor]["fold"] == rec["fold"]
                or ex["outcome_4y_scd"] != by[donor]["label"]
            ):
                raise ValueError("example mismatch or fold leakage")
            ex["holter"] = summaries[donor]
    (a.output / "input_summary30.json").write_text(
        json.dumps(result, allow_nan=False, separators=(",", ":")), encoding="utf-8"
    )
    (a.output / "review_candidates.json").write_text(
        json.dumps(candidates, allow_nan=False, separators=(",", ":")), encoding="utf-8"
    )
    print(
        "patients",
        len(patients),
        "quality_passed_windows",
        sum(len(v) for v in indices.values()),
        "feature_errors",
        dict(failure_counts),
        "positive",
        sum(r["label"] for r in patients),
        flush=True,
    )


if __name__ == "__main__":
    main()
