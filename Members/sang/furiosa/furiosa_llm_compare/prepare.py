"""Build private Holter split-aligned LLM study input from validated local caches."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from Members.sang.experiments.monorouter_holter.clinical_features import (  # noqa: E402
    _sha256_file,
    cache_key,
    load_lead_metadata,
    read_feature_cache,
    select_development,
)
from Members.sang.model.config import load_basic_config  # noqa: E402
from Members.sang.model.ML.feature_versions import MODEL_4  # noqa: E402


def main(output):
    basic_path = REPO / "Members/sang/model/basic.yaml"
    basic = load_basic_config(basic_path)
    raw = basic["data"]["raw_ecg"]
    cfg = yaml.safe_load((REPO / "Members/sang/notebooks/monorouter_holter.yaml").read_text())
    settings = {
        k: cfg["clinical"][k]
        for k in (
            "chunk_seconds",
            "context_seconds",
            "min_rr_ms",
            "max_rr_ms",
            "relative_limit",
            "sdann_min_coverage",
            "hourly_min_coverage",
        )
    }
    selected = select_development(
        raw["split_path"],
        raw["wave_directory"],
        REPO / "Members/sang/datas/Holter/physionet_reference/rhythm_hrv_dev.csv",
        None,
        cfg["experiment"]["sample_seed"],
    )
    lead = load_lead_metadata(raw["lead_metadata_path"])
    digest = _sha256_file(raw["quality_index_path"])
    cache = REPO / "Members/sang/results/monorouter_holter/clinical_cache_v1"
    clinical = pd.read_csv(
        REPO / "Members/sang/model/ML/data/clinical_clean.csv", dtype={"Patient ID": str}
    )
    clinical.columns = clinical.columns.str.strip()
    if clinical.columns.duplicated().any() or clinical["Patient ID"].duplicated().any():
        raise ValueError("임상 열 또는 환자 ID 중복")
    clinical = clinical.set_index("Patient ID")
    if not set(MODEL_4).issubset(clinical.columns):
        raise ValueError("임상 변수 누락")
    records = []
    excluded = {
        "missing_clinical": 0,
        "label_disagreement": 0,
        "missing_feature": 0,
        "invalid_clinical": 0,
    }
    for record in selected:
        pid = record["patient_id"]
        if pid not in clinical.index:
            excluded["missing_clinical"] += 1
            continue
        row = clinical.loc[pid]
        if pd.isna(row["SCD"]) or int(row["SCD"]) != record["label"]:
            excluded["label_disagreement"] += 1
            continue
        values = pd.to_numeric(row[MODEL_4], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            excluded["invalid_clinical"] += 1
            continue
        key = cache_key(record, digest, lead[record["wave_path"].name], settings)
        feat = read_feature_cache(cache, key)
        if feat is None:
            excluded["missing_feature"] += 1
            continue
        records.append(
            {
                "patient_id": pid,
                "split": record["split"],
                "label": record["label"],
                "clinical": dict(zip(MODEL_4, map(float, values), strict=False)),
                "holter": feat,
            }
        )
    train = [x for x in records if x["split"] == "train"]
    val = [x for x in records if x["split"] == "validation"]
    if len(val) != 71 or len(train) < 4 or {x["label"] for x in train} != {0, 1}:
        raise ValueError("예상 개발 환자/라벨 불일치")
    scaler = StandardScaler().fit(np.array([[x["clinical"][k] for k in MODEL_4] for x in train]))
    matrix = scaler.transform([[x["clinical"][k] for k in MODEL_4] for x in train])
    for rec in val:
        z = scaler.transform([[rec["clinical"][k] for k in MODEL_4]])[0]
        dist = np.sum((matrix - z) ** 2, axis=1)
        chosen = []
        for label in (0, 1):
            candidates = sorted(
                (float(dist[i]), x["patient_id"], i)
                for i, x in enumerate(train)
                if x["label"] == label
            )
            if len(candidates) < 2:
                raise ValueError("학습 사례 부족")
            chosen.extend(train[i] for _, _, i in candidates[:2])
        rec["examples"] = [
            {"clinical": x["clinical"], "holter": x["holter"], "outcome_4y_scd": x["label"]}
            for x in chosen
        ]
    payload = {
        "metadata": {
            "split": "Holter validation",
            "train_cases": len(train),
            "validation": len(val),
            "exclusions": excluded,
            "features": MODEL_4,
            "selection": "2 nearest train cases per class; StandardScaler fit on Holter train only",
            "source_feature_version": "experimental_rr_v1",
            "settings": settings,
            "caveats": [
                "clinical measurement timing not independently verified",
                "clinical and Holter historical splits differ",
                "test was previously observed",
                "nearest examples balanced, not prevalence representative",
            ],
        },
        "validation": val,
    }
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, ensure_ascii=False, allow_nan=False)
    print("train", len(train), "validation", len(val), "excluded", excluded, "output", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
