"""Create 5-fold patient-separated LLM study inputs from verified server data."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from feature_versions import MODEL_4
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/root/std3/furiosa_holter_project")
from Members.sang.experiments.monorouter_holter.clinical_features import validate_llm_features


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(args):
    clinical = pd.read_csv(args.clinical, dtype={"Patient ID": str})
    clinical.columns = clinical.columns.str.strip()
    split = pd.read_csv(args.split, dtype={"pid": str})
    split["split"] = split["split"].replace({"val": "validation"})
    if (
        clinical.columns.duplicated().any()
        or clinical["Patient ID"].duplicated().any()
        or split.pid.duplicated().any()
    ):
        raise ValueError("duplicate patient/column")
    if not {"pid", "SCD", "split"}.issubset(split.columns) or not set(MODEL_4).issubset(
        clinical.columns
    ):
        raise ValueError("missing fields")
    if not split["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("invalid split")
    clinical = clinical.set_index("Patient ID")
    source = json.loads(args.holter.read_text())
    features = source["records"]
    records = []
    for _, r in split[split["split"].isin(["train", "validation"])].iterrows():
        pid = r["pid"]
        label = r["SCD"]
        group = r["split"]
        if pid not in clinical.index or pid not in features:
            raise ValueError("missing patient")
        row = clinical.loc[pid]
        h = features[pid]
        if (
            label not in (0, 1)
            or row["SCD"] not in (0, 1)
            or int(label) != int(row["SCD"])
            or int(label) != h["holter_label"]
            or group != h["holter_split"]
        ):
            raise ValueError("label/split mismatch")
        v = pd.to_numeric(row[MODEL_4], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(v).all():
            raise ValueError("invalid clinical data")
        validate_llm_features(h["holter"])
        records.append(
            {
                "patient_id": pid,
                "original_split": group,
                "label": int(label),
                "clinical": dict(zip(MODEL_4, map(float, v), strict=False)),
                "holter": h["holter"],
            }
        )
    records.sort(key=lambda x: x["patient_id"])
    if len(records) != 356 or len({r["patient_id"] for r in records}) != 356:
        raise ValueError("development cohort mismatch")
    assert {r["patient_id"] for r in records}.isdisjoint(
        set(split.loc[split["split"] == "test", "pid"])
    )
    y = np.asarray([r["label"] for r in records])
    X = np.asarray([[r["clinical"][k] for k in MODEL_4] for r in records])
    fold_sizes = []
    for fold, (train_idx, test_idx) in enumerate(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y), 1
    ):
        if set(train_idx) & set(test_idx):
            raise ValueError("fold overlap")
        scaler = StandardScaler().fit(X[train_idx])
        Z = scaler.transform(X[train_idx])
        fold_sizes.append(
            {
                "fold": fold,
                "train": len(train_idx),
                "test": len(test_idx),
                "positive": int(y[test_idx].sum()),
            }
        )
        for i in test_idx:
            rec = records[i]
            rec["fold"] = fold
            z = scaler.transform(X[i : i + 1])[0]
            dist = np.sum((Z - z) ** 2, axis=1)
            selected = []
            for label in (0, 1):
                candidates = sorted(
                    (float(dist[j]), records[idx]["patient_id"], idx)
                    for j, idx in enumerate(train_idx)
                    if y[idx] == label
                )
                if len(candidates) < 2:
                    raise ValueError("insufficient examples")
                selected.extend(records[idx] for _, _, idx in candidates[:2])
            rec["example_patient_ids"] = [r["patient_id"] for r in selected]
            rec["examples"] = [
                {"clinical": r["clinical"], "holter": r["holter"], "outcome_4y_scd": r["label"]}
                for r in selected
            ]
    if any("fold" not in r for r in records):
        raise ValueError("missing fold")
    assignment = {r["patient_id"]: r["fold"] for r in records}
    for r in records:
        if len(r["example_patient_ids"]) != 4 or any(
            assignment[pid] == r["fold"] for pid in r["example_patient_ids"]
        ):
            raise ValueError("example fold leakage")
    metadata = {
        "split": "5-fold stratified CV of original train+validation; original test excluded",
        "seed": 42,
        "fold_sizes": fold_sizes,
        "n_patients": len(records),
        "positive": int(y.sum()),
        "features": MODEL_4,
        "clinical_sha256": digest(args.clinical),
        "split_sha256": digest(args.split),
        "holter_sha256": digest(args.holter),
        "holter_provenance": source["metadata"],
        "selection": "2 nearest opposite-fold examples per class; StandardScaler fit on other 4 folds",
        "caveats": [
            "clinical measurement timing unverified",
            "folds are new and were not used in prior DL comparisons",
            "balanced examples do not represent prevalence",
            "previously examined development patients; not external validation",
        ],
    }
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(
            {"metadata": metadata, "patients": records}, f, ensure_ascii=False, allow_nan=False
        )
    print("환자:", len(records), "양성:", int(y.sum()), "fold:", fold_sizes, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clinical", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--holter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
