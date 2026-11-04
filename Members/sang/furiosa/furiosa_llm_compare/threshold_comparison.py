"""Aggregate previously generated predictions; never export individual records."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = ("holter", "clinical_only", "holter_clinical", "holter_clinical_cases")
RUNS = {
    "qwen32": "run_20261102_cv5",
    "gpt_oss": "run_20261103_gpt_oss",
    "qwen_moe": "run_20261103_qwen_moe",
    "exaone": "run_20261103_exaone",
}


def measures(y, scores, threshold):
    positive = scores >= threshold
    tp = int(((y == 1) & positive).sum())
    fp = int(((y == 0) & positive).sum())
    fn = int(((y == 1) & ~positive).sum())
    tn = int(((y == 0) & ~positive).sum())
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp) if tp + fp else None,
        "flagged": tp + fp,
    }


def main(root, output):
    baseline = json.loads((root / "run_20261102_cv5/metadata.json").read_text())["input_sha256"]
    diag = json.loads((root / "run_20261102_diagnostics/metadata.json").read_text())
    assert diag["input_metadata"]["baseline_cv_sha256"] == baseline
    aggregate = json.loads((root / "multimodel_aggregate_20261103.json").read_text())
    expected = None
    result = {
        "method": "fixed existing scores; positive if score >= threshold; threshold selected on same cohort",
        "fixed_thresholds": [0.05, 0.1, 0.2, 0.3, 0.5],
        "models": {},
    }
    for name, run in RUNS.items():
        meta = json.loads((root / run / "metadata.json").read_text())
        assert meta["status"] == "COMPLETE" and meta["input_sha256"] == baseline
        all_predictions = pd.read_csv(root / run / "predictions.csv")
        clinical = (
            pd.read_csv(root / "run_20261102_diagnostics/predictions.csv")
            if name == "qwen32"
            else None
        )
        result["models"][name] = {}
        for condition in CONDITIONS:
            source = (
                clinical if name == "qwen32" and condition == "clinical_only" else all_predictions
            )
            g = source[(source.condition == condition) & (source.status == "ok")].copy()
            assert len(g) == 356 and g.patient_id.nunique() == 356
            assert set(g.y_true.unique()) == {0, 1} and int(g.y_true.sum()) == 40
            assert np.isfinite(g.p_positive).all() and g.p_positive.between(0, 1).all()
            patients = g.set_index("patient_id").y_true.sort_index()
            if expected is None:
                expected = patients
            else:
                assert patients.equals(expected), "patient or label mismatch"
            y = g.y_true.to_numpy(int)
            scores = g.p_positive.to_numpy(float)
            fixed = [measures(y, scores, t) for t in result["fixed_thresholds"]]
            at_half = fixed[-1]
            assert [[at_half["tn"], at_half["fp"]], [at_half["fn"], at_half["tp"]]] == aggregate[
                "conditions"
            ][name][condition]["confusion_0_5"]
            target_results = {}
            for goal in (0.8, 0.9):
                candidates = sorted(set(scores), reverse=True)
                chosen = next(
                    measures(y, scores, t)
                    for t in candidates
                    if measures(y, scores, t)["sensitivity"] >= goal
                )
                target_results[str(goal)] = chosen
            result["models"][name][condition] = {"fixed": fixed, "targets": target_results}
    assert not output.exists(), "refusing to overwrite existing output"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    output.chmod(0o600)
    print("Validated 4 models x 4 conditions x 356 matched patients; output:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
