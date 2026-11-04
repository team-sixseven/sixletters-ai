"""Build Qwen comparison on the server from the explicitly requested clinical sources."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from feature_versions import MODEL_4
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/root/std3/furiosa_holter_project")
from Members.sang.experiments.monorouter_holter.clinical_features import validate_llm_features


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    c = pd.read_csv(args.clinical, dtype={"Patient ID": str})
    c.columns = c.columns.str.strip()
    s = pd.read_csv(args.split, dtype={"pid": str})
    if (
        c.columns.duplicated().any()
        or c["Patient ID"].duplicated().any()
        or s.pid.duplicated().any()
    ):
        raise ValueError("중복 환자 또는 열")
    if not set(MODEL_4).issubset(c.columns) or not {"pid", "SCD", "split"}.issubset(s.columns):
        raise ValueError("입력 열 누락")
    c = c.set_index("Patient ID")
    s["split"] = s["split"].replace({"val": "validation"})
    if not s["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("예상하지 못한 분할")
    source = json.loads(args.holter.read_text())
    holter = source["records"]
    records = []
    excluded = {
        "missing_clinical": 0,
        "missing_holter": 0,
        "label_disagreement": 0,
        "split_disagreement": 0,
        "invalid_clinical": 0,
    }
    for _, r in s[s["split"].isin(["train", "validation"])].iterrows():
        pid = r["pid"]
        split = r["split"]
        label = r["SCD"]
        if pid not in c.index:
            excluded["missing_clinical"] += 1
            continue
        if pid not in holter:
            excluded["missing_holter"] += 1
            continue
        row = c.loc[pid]
        h = holter[pid]
        if (
            label not in (0, 1)
            or row["SCD"] not in (0, 1)
            or int(label) != int(row["SCD"])
            or int(label) != h["holter_label"]
        ):
            excluded["label_disagreement"] += 1
            continue
        if split != h["holter_split"]:
            excluded["split_disagreement"] += 1
            continue
        values = pd.to_numeric(row[MODEL_4], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            excluded["invalid_clinical"] += 1
            continue
        validate_llm_features(h["holter"])
        records.append(
            {
                "patient_id": pid,
                "split": split,
                "label": int(label),
                "clinical": dict(zip(MODEL_4, map(float, values), strict=False)),
                "holter": h["holter"],
            }
        )
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "validation"]
    if len(train) != 285 or len(val) != 71 or any(excluded.values()):
        raise ValueError("개발 환자 매칭·라벨 검증 실패")
    scaler = StandardScaler().fit([[r["clinical"][k] for k in MODEL_4] for r in train])
    matrix = scaler.transform([[r["clinical"][k] for k in MODEL_4] for r in train])
    for rec in val:
        z = scaler.transform([[rec["clinical"][k] for k in MODEL_4]])[0]
        dist = np.sum((matrix - z) ** 2, axis=1)
        selected = []
        for label in (0, 1):
            candidates = sorted(
                (float(dist[i]), r["patient_id"], i)
                for i, r in enumerate(train)
                if r["label"] == label
            )
            if len(candidates) < 2:
                raise ValueError("학습 사례 부족")
            selected.extend(train[i] for _, _, i in candidates[:2])
        rec["example_patient_ids"] = [r["patient_id"] for r in selected]
        rec["examples"] = [
            {"clinical": r["clinical"], "holter": r["holter"], "outcome_4y_scd": r["label"]}
            for r in selected
        ]
    meta = {
        "split": "provided clinical final_split (aligned with Holter)",
        "train_cases": len(train),
        "validation": len(val),
        "exclusions": excluded,
        "features": MODEL_4,
        "clinical_source": str(args.clinical),
        "clinical_sha256": sha(args.clinical),
        "split_source": str(args.split),
        "split_sha256": sha(args.split),
        "holter_features_sha256": sha(args.holter),
        "holter_provenance": source["metadata"],
        "selection": "2 nearest train cases per class; StandardScaler fit on train only",
        "source_feature_version": "experimental_rr_v1",
        "caveats": [
            "clinical measurement timing not independently verified",
            "test was previously observed",
            "nearest examples balanced, not prevalence representative",
        ],
    }
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"metadata": meta, "validation": val}, f, ensure_ascii=False, allow_nan=False)
    print("train", len(train), "validation", len(val), "exclusions", excluded)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clinical", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--holter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
