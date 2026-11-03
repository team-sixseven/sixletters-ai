"""Warm the existing per-record feature cache with independent development partitions."""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))


def prepare_partition(partition):
    from Members.sang.experiments.monorouter_holter import clinical_features as cf
    from Members.sang.model.config import load_basic_config

    original = cf.select_development

    def subset(*args, **kwargs):
        records = original(*args, **kwargs)
        return [r for i, r in enumerate(records) if i % 4 == partition]

    cf.select_development = subset
    basic_path = REPO / "Members/sang/model/basic.yaml"
    basic = load_basic_config(basic_path)
    config = yaml.safe_load((REPO / "Members/sang/notebooks/monorouter_holter.yaml").read_text())
    experiment, clinical = config["experiment"], config["clinical"]
    if experiment["split"] != "development" or experiment["sample_size"] is not None:
        raise ValueError("개발 전체 설정이 필요합니다.")
    settings = {
        k: clinical[k]
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
    _, results = cf.prepare_development(
        REPO,
        basic,
        basic_path,
        REPO / "Members/sang/datas/Holter/physionet_reference/rhythm_hrv_dev.csv",
        None,
        experiment["sample_seed"],
        settings,
    )
    return partition, len(results), sum(x[2] is not None for x in results)


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as pool:
        for partition, count, success in pool.map(prepare_partition, range(4)):
            print(f"병렬 처리 {partition + 1}/4 완료: {success}/{count}", flush=True)
