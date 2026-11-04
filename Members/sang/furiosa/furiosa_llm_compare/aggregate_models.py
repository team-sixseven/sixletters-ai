"""Private aggregate of fixed-cohort model comparisons without patient-level export."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_server import atomic
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def summary(g):
    y = g.y_true.to_numpy(int)
    p = g.p_positive.to_numpy(float)
    tn, fp, fn, tp = confusion_matrix(y, p >= 0.5, labels=[0, 1]).ravel().tolist()
    return {
        "n": len(g),
        "positive": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p))
        if len(np.unique(y)) == 2
        else None,
        "brier": float(brier_score_loss(y, p)),
        "confusion_0_5": [[tn, fp], [fn, tp]],
        "sensitivity_0_5": tp / (tp + fn) if tp + fn else None,
        "specificity_0_5": tn / (tn + fp) if tn + fp else None,
        "mean_seconds": float(g.seconds.mean()),
    }


def compare(a, b, draws=1500):
    j = a[["patient_id", "y_true", "p_positive"]].merge(
        b[["patient_id", "y_true", "p_positive"]],
        on="patient_id",
        suffixes=("_new", "_qwen"),
        validate="one_to_one",
    )
    if len(j) == 0 or not (j.y_true_new == j.y_true_qwen).all():
        raise ValueError("patient/label mismatch")
    y = j.y_true_new.to_numpy(int)
    x = j.p_positive_new.to_numpy(float)
    z = j.p_positive_qwen.to_numpy(float)
    if len(np.unique(y)) < 2:
        return {"paired_n": len(y), "auc_difference": None}
    rng = np.random.default_rng(42)
    values = []
    for _ in range(draws):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            values.append(roc_auc_score(y[i], x[i]) - roc_auc_score(y[i], z[i]))
    return {
        "paired_n": len(y),
        "auc_difference": float(roc_auc_score(y, x) - roc_auc_score(y, z)),
        "fixed_prediction_bootstrap_95": [float(v) for v in np.quantile(values, [0.025, 0.975])],
        "caveat": "fixed outputs; does not include model reruns, fold/retrieval reselection, or multiple comparison uncertainty",
    }


def main(args):
    root = args.root
    groups = {}
    status = {}
    baseline_sha = json.loads((root / "run_20261102_cv5/metadata.json").read_text())["input_sha256"]
    for name, run in [
        ("qwen32", "run_20261102_cv5"),
        ("gpt_oss", "run_20261103_gpt_oss"),
        ("qwen_moe", "run_20261103_qwen_moe"),
        ("exaone", "run_20261103_exaone"),
    ]:
        p = root / run
        if not (p / "metadata.json").exists():
            status[name] = "NOT_STARTED"
            continue
        meta = json.loads((p / "metadata.json").read_text())
        status[name] = meta["status"]
        if meta["input_sha256"] != baseline_sha:
            raise ValueError("input hash mismatch")
        if not (p / "predictions.csv").exists():
            continue
        a = pd.read_csv(p / "predictions.csv")
        a = a[a.status == "ok"]
        for cond, g in a.groupby("condition"):
            if len(g) >= 2 and g.patient_id.nunique() == len(g):
                groups[(name, cond)] = g
    diag_meta = json.loads((root / "run_20261102_diagnostics/metadata.json").read_text())
    if diag_meta["input_metadata"]["baseline_cv_sha256"] != baseline_sha:
        raise ValueError("clinical baseline input hash mismatch")
    p = root / "run_20261102_diagnostics/predictions.csv"
    a = pd.read_csv(p)
    g = a[(a.condition == "clinical_only") & (a.status == "ok")]
    groups[("qwen32", "clinical_only")] = g
    result = {
        "statuses": status,
        "conditions": {},
        "paired_vs_qwen32": {},
        "caveat": "development data already examined; same patient/folds/prompt text; different templates and quantization; research only",
    }
    for (name, cond), g in groups.items():
        result["conditions"].setdefault(name, {})[cond] = summary(g)
        if name != "qwen32" and ("qwen32", cond) in groups:
            result["paired_vs_qwen32"].setdefault(name, {})[cond] = compare(
                g, groups[("qwen32", cond)]
            )
    atomic(args.output, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print("모델 비교 집계 완료", args.output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
