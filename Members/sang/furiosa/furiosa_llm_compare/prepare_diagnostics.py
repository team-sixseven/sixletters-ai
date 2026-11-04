"""Build leakage-free case retrieval alternatives on fixed 5-fold data."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def seed_for(pid, seed):
    return int.from_bytes(hashlib.sha256(f"{seed}:{pid}".encode()).digest()[:8], "big")


def holter_vector(record):
    h = record["holter"]
    hr = h["heart_rate_bpm"]
    hrv = h["hrv_time_domain_estimates"] or {}
    rec = h["recording"]
    vals = [
        hr["mean_from_filtered_rr"],
        hr["hourly_min"],
        hr["hourly_max"],
        hrv.get("SDNN_ms"),
        hrv.get("SDANN_ms"),
        hrv.get("RMSSD_ms"),
        hrv.get("pNN50_percent"),
        h["deceleration_capacity_estimate_ms"],
        rec["excluded_interval_percent"],
        rec["analyzed_hours"],
    ]
    return [float(v) if v is not None else float("nan") for v in vals]


def as_example(r):
    return {"clinical": r["clinical"], "holter": r["holter"], "outcome_4y_scd": r["label"]}


def main(args):
    raw = args.input.read_bytes()
    p = json.loads(raw)
    rows = p["patients"]
    cols = p["metadata"]["features"]
    if len(rows) != 356 or len({r["patient_id"] for r in rows}) != 356:
        raise ValueError("cohort mismatch")
    by = {r["patient_id"]: r for r in rows}
    clin = np.array([[r["clinical"][k] for k in cols] for r in rows], float)
    ecg = np.array([holter_vector(r) for r in rows], float)
    strategies = ("random_101", "random_202", "random_303", "hybrid", "nearest_unbalanced")
    for fold in range(1, 6):
        train = np.array([i for i, r in enumerate(rows) if r["fold"] != fold])
        test = [i for i, r in enumerate(rows) if r["fold"] == fold]
        cs = StandardScaler().fit(clin[train])
        cz = cs.transform(clin[train])
        imp = SimpleImputer(strategy="median", add_indicator=True).fit(ecg[train])
        hz0 = imp.transform(ecg[train])
        hs = StandardScaler().fit(hz0)
        hz = hs.transform(hz0)
        for i in test:
            rec = rows[i]
            c = cs.transform(clin[i : i + 1])[0]
            h = hs.transform(imp.transform(ecg[i : i + 1]))[0]
            cdist = np.mean((cz - c) ** 2, axis=1)
            hdist = np.mean((hz - h) ** 2, axis=1)
            hybrid = (cdist + hdist) / 2
            alternatives = {}
            for strategy in strategies:
                if strategy.startswith("random_"):
                    rng = np.random.default_rng(
                        seed_for(rec["patient_id"], int(strategy.split("_")[1]))
                    )
                    chosen = []
                    for label in (0, 1):
                        pool = np.array([int(idx) for idx in train if rows[idx]["label"] == label])
                        chosen.extend(map(int, rng.choice(pool, size=2, replace=False)))
                elif strategy == "hybrid":
                    chosen = []
                    for label in (0, 1):
                        candidates = sorted(
                            (float(hybrid[j]), rows[idx]["patient_id"], int(idx))
                            for j, idx in enumerate(train)
                            if rows[idx]["label"] == label
                        )
                        chosen.extend(idx for _, _, idx in candidates[:2])
                else:
                    chosen = [
                        idx
                        for _, _, idx in sorted(
                            (float(cdist[j]), rows[idx]["patient_id"], int(idx))
                            for j, idx in enumerate(train)
                        )[:4]
                    ]
                if (
                    len(chosen) != 4
                    or len(set(chosen)) != 4
                    or any(rows[idx]["fold"] == fold for idx in chosen)
                ):
                    raise ValueError("case leakage")
                if (
                    strategy != "nearest_unbalanced"
                    and sum(rows[idx]["label"] for idx in chosen) != 2
                ):
                    raise ValueError("balanced case mismatch")
                alternatives[strategy] = {
                    "example_patient_ids": [rows[idx]["patient_id"] for idx in chosen],
                    "examples": [as_example(rows[idx]) for idx in chosen],
                }
            rec["alternatives"] = alternatives
    for r in rows:
        for v in r["alternatives"].values():
            for pid, e in zip(v["example_patient_ids"], v["examples"], strict=False):
                donor = by[pid]
                if donor["fold"] == r["fold"] or e != as_example(donor):
                    raise ValueError("case verification failure")
    out = {
        "metadata": {
            "baseline_cv_sha256": hashlib.sha256(raw).hexdigest(),
            "strategies": strategies,
            "hybrid_holter_fields": [
                "mean_hr",
                "hourly_min_hr",
                "hourly_max_hr",
                "SDNN",
                "SDANN",
                "RMSSD",
                "pNN50",
                "DC",
                "excluded_RR_percent",
                "analyzed_hours",
            ],
            "hybrid": "clinical and Holter standardized feature squared distances averaged equally; missing Holter values imputed on training folds with indicators",
            "seed": "sha256(patient_id, seed)",
            "caveats": p["metadata"]["caveats"],
        },
        "patients": rows,
    }
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(out, f, ensure_ascii=False, allow_nan=False)
    print("대체 사례 준비:", len(rows), "명 ×", len(strategies), "전략")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
