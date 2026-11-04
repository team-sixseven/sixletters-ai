"""Build outer-test / calibration / donor-separated prompts for threshold study."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


def main(args):
    raw = args.input.read_bytes()
    p = json.loads(raw)
    rows = p["patients"]
    cols = p["metadata"]["features"]
    X = np.array([[r["clinical"][k] for k in cols] for r in rows])
    prepared = []
    if len(rows) != 356 or len({r["patient_id"] for r in rows}) != 356:
        raise ValueError("cohort mismatch")
    for outer in range(1, 6):
        calibration = outer % 5 + 1
        pool = [i for i, r in enumerate(rows) if r["fold"] not in (outer, calibration)]
        eval_ids = {r["patient_id"] for r in rows if r["fold"] == outer}
        cal_ids = {r["patient_id"] for r in rows if r["fold"] == calibration}
        if eval_ids & cal_ids:
            raise ValueError("evaluation/calibration overlap")
        scaler = StandardScaler().fit(X[pool])
        transformed = scaler.transform(X[pool])
        for i, r in enumerate(rows):
            if r["fold"] not in (outer, calibration):
                continue
            z = scaler.transform(X[i : i + 1])[0]
            dist = np.sum((transformed - z) ** 2, axis=1)
            chosen = []
            for label in (0, 1):
                candidates = sorted(
                    (float(dist[j]), rows[idx]["patient_id"], idx)
                    for j, idx in enumerate(pool)
                    if rows[idx]["label"] == label
                )
                if len(candidates) < 2:
                    raise ValueError("insufficient training examples")
                chosen.extend(rows[idx] for _, _, idx in candidates[:2])
            q = {
                "patient_id": r["patient_id"],
                "fold": r["fold"],
                "outer_fold": outer,
                "role": "evaluation" if r["fold"] == outer else "calibration",
                "label": r["label"],
                "clinical": r["clinical"],
                "holter": r["holter"],
                "example_patient_ids": [x["patient_id"] for x in chosen],
                "examples": [
                    {"clinical": x["clinical"], "holter": x["holter"], "outcome_4y_scd": x["label"]}
                    for x in chosen
                ],
            }
            if any(x["fold"] in (outer, calibration) for x in chosen):
                raise ValueError("case leakage")
            prepared.append(q)
    meta = {
        "cv_input_sha256": hashlib.sha256(raw).hexdigest(),
        "roles": "outer test fold; next fold for calibration; remaining 3 folds donors and scaler",
        "sensitivity_target": 0.8,
        "caveat": "calibration has 8 positive patients per fold; samples overlap between different outer folds; exploratory research only",
    }
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"metadata": meta, "patients": prepared}, f, ensure_ascii=False, allow_nan=False)
    print(
        "threshold study prompts:",
        len(prepared),
        "outer evaluation:",
        sum(r["role"] == "evaluation" for r in prepared),
        "calibration:",
        sum(r["role"] == "calibration" for r in prepared),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
