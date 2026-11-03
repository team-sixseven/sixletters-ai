"""Export existing verified RR/HRV features to a private server-side study input."""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

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


def main(output):
    p = REPO / "Members/sang/model/basic.yaml"
    basic = load_basic_config(p)
    raw = basic["data"]["raw_ecg"]
    cfg = yaml.safe_load((REPO / "Members/sang/notebooks/monorouter_holter.yaml").read_text())
    keys = (
        "chunk_seconds",
        "context_seconds",
        "min_rr_ms",
        "max_rr_ms",
        "relative_limit",
        "sdann_min_coverage",
        "hourly_min_coverage",
    )
    settings = {k: cfg["clinical"][k] for k in keys}
    selected = select_development(
        raw["split_path"],
        raw["wave_directory"],
        REPO / "Members/sang/datas/Holter/physionet_reference/rhythm_hrv_dev.csv",
        None,
        cfg["experiment"]["sample_seed"],
    )
    leads = load_lead_metadata(raw["lead_metadata_path"])
    digest = _sha256_file(raw["quality_index_path"])
    cache = REPO / "Members/sang/results/monorouter_holter/clinical_cache_v1"
    records = {}
    for r in selected:
        key = cache_key(r, digest, leads[r["wave_path"].name], settings)
        features = read_feature_cache(cache, key)
        if features is None:
            raise RuntimeError("필요한 기존 전처리 캐시가 없거나 무효입니다.")
        records[r["patient_id"]] = {
            "holter": features,
            "holter_split": r["split"],
            "holter_label": r["label"],
        }
    if len(records) != 356:
        raise ValueError("Holter 개발 환자 수 불일치")
    meta = {
        "feature_version": "experimental_rr_v1",
        "settings": settings,
        "quality_index_sha256": digest,
        "holter_split_sha256": _sha256_file(Path(raw["split_path"])),
        "feature_code_sha256": _sha256_file(
            REPO / "Members/sang/experiments/monorouter_holter/clinical_features.py"
        ),
        "cache_source": "locally verified canonical caches",
    }
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"metadata": meta, "records": records}, f, ensure_ascii=False, allow_nan=False)
    print("검증된 Holter 개발 특징:", len(records))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args().output)
