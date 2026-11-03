"""Aggregate private 356-patient comparison outputs without exporting patient rows."""

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


def stats(frame):
    y = frame["y_true"].to_numpy(int)
    p = frame["p_positive"].to_numpy(float)
    tn, fp, fn, tp = confusion_matrix(y, p >= 0.5, labels=[0, 1]).ravel().tolist()
    return {
        "n": len(y),
        "positive": int(y.sum()),
        "mean_score": float(p.mean()),
        "observed_positive_fraction": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_uncalibrated": float(brier_score_loss(y, p)),
        "threshold_0_5_confusion": [[tn, fp], [fn, tp]],
        "sensitivity_at_0_5": tp / (tp + fn) if tp + fn else None,
        "specificity_at_0_5": tn / (tn + fp) if tn + fp else None,
    }


def paired(a, b, draws=3000):
    j = a[["patient_id", "y_true", "p_positive"]].merge(
        b[["patient_id", "y_true", "p_positive"]],
        on="patient_id",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if len(j) != 356 or not (j.y_true_a == j.y_true_b).all():
        raise ValueError("paired population mismatch")
    y = j.y_true_a.to_numpy(int)
    x = j.p_positive_a.to_numpy(float)
    z = j.p_positive_b.to_numpy(float)
    rng = np.random.default_rng(42)
    auc = []
    ap = []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        sample = y[idx]
        if len(np.unique(sample)) < 2:
            continue
        auc.append(roc_auc_score(sample, x[idx]) - roc_auc_score(sample, z[idx]))
        ap.append(average_precision_score(sample, x[idx]) - average_precision_score(sample, z[idx]))
    return {
        "n": len(j),
        "auroc_difference_a_minus_b": float(roc_auc_score(y, x) - roc_auc_score(y, z)),
        "auroc_difference_fixed_prediction_ci95": [
            float(v) for v in np.quantile(auc, [0.025, 0.975])
        ],
        "ap_difference_a_minus_b": float(
            average_precision_score(y, x) - average_precision_score(y, z)
        ),
        "ap_difference_fixed_prediction_ci95": [float(v) for v in np.quantile(ap, [0.025, 0.975])],
        "bootstrap_seed": 42,
        "draws": draws,
        "caveat": "fixed out-of-fold predictions; does not repeat fold assignment, retrieval, or generation",
    }


def main(args):
    root = args.root
    paths = {
        "base": root / "run_20261102_cv5",
        "ablation": root / "run_20261102_diagnostics",
        "smoking": root / "run_20261102_smoking",
        "threshold": root / "run_20261102_threshold",
    }
    for path in paths.values():
        if json.loads((path / "metadata.json").read_text())["status"] != "COMPLETE":
            raise RuntimeError("an experiment is not complete")
    groups = {}
    for key in ("base", "ablation", "smoking"):
        frame = pd.read_csv(paths[key] / "predictions.csv")
        for name, g in frame.groupby("condition"):
            if len(g) != 356 or g.patient_id.nunique() != 356 or not (g.status == "ok").all():
                raise ValueError("incomplete group")
            if name in groups:
                raise ValueError("duplicate condition")
            groups[name] = g
    summary = {"arms": {name: stats(g) for name, g in groups.items()}, "paired_comparisons": {}}
    pairs = [
        ("clinical_only_corrected_prompt", "clinical_only"),
        ("holter_clinical", "clinical_only"),
        ("holter_clinical", "clinical_only_corrected_prompt"),
        ("holter_clinical_cases", "holter_clinical"),
        ("random_101", "holter_clinical_cases"),
        ("random_202", "holter_clinical_cases"),
        ("random_303", "holter_clinical_cases"),
        ("hybrid", "holter_clinical_cases"),
        ("nearest_unbalanced", "holter_clinical_cases"),
        ("clinical_no_smoking", "clinical_only"),
        ("combined_no_smoking", "holter_clinical"),
    ]
    for a, b in pairs:
        summary["paired_comparisons"][a + "_vs_" + b] = paired(groups[a], groups[b])
    original = json.loads((root / "clinical_source_20261102/input_cv_5fold.json").read_text())[
        "patients"
    ]
    feature = pd.DataFrame(
        [
            {
                "patient_id": r["patient_id"],
                "hrv_missing": r["holter"]["hrv_time_domain_estimates"] is None,
                "rhythm": "sinus"
                if r["holter"]["recording"]["holter_rhythm"] == "sinus"
                else "non_sinus_unknown",
                "excluded_rr": r["holter"]["recording"]["excluded_interval_percent"],
                "fold": r["fold"],
            }
            for r in original
        ]
    )
    base = groups["holter_clinical_cases"].merge(
        feature[["patient_id", "hrv_missing", "rhythm", "excluded_rr"]],
        on="patient_id",
        validate="one_to_one",
    )
    summary["quality_groups"] = {}
    for rhythm, g in base.groupby("rhythm"):
        summary["quality_groups"][rhythm] = stats(g)
    for classification, g in base.groupby(
        np.select(
            [
                (base.y_true == 1) & (base.p_positive >= 0.5),
                (base.y_true == 1) & (base.p_positive < 0.5),
                (base.y_true == 0) & (base.p_positive >= 0.5),
            ],
            ["TP", "FN", "FP"],
            default="TN",
        )
    ):
        summary["quality_groups"][classification] = {
            "n": len(g),
            "hrv_missing": int(g.hrv_missing.sum()),
            "rr_exclusion_gt20_exploratory": int((g.excluded_rr > 20).sum()),
            "median_rr_exclusion": float(g.excluded_rr.median()),
        }
    summary["threshold_cases"] = json.loads((paths["threshold"] / "metrics.json").read_text())
    # Reuse no-example predictions: unlike few-shot results, they never depend on any donor fold.
    clin = groups["holter_clinical"].merge(
        feature[["patient_id", "fold"]],
        on="patient_id",
        suffixes=("", "_mapping"),
        validate="one_to_one",
    )
    truth = []
    pred = []
    thresholds = []
    for fold in range(1, 6):
        cal = clin[clin["fold_mapping"] == fold % 5 + 1]
        ev = clin[clin["fold_mapping"] == fold]
        pos = cal.loc[cal.y_true == 1, "p_positive"].to_numpy()
        if len(pos) != 8:
            raise ValueError("calibration count mismatch")
        threshold = next(
            float(v) for v in sorted(set(cal.p_positive), reverse=True) if np.mean(pos >= v) >= 0.8
        )
        thresholds.append(threshold)
        truth.extend(ev.y_true.astype(int))
        pred.extend((ev.p_positive >= threshold).astype(int))
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel().tolist()
    summary["threshold_clinical_holter"] = {
        "thresholds": thresholds,
        "confusion": [[tn, fp], [fn, tp]],
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "note": "calibration fold separate; no cases; no threshold tuning on held-out patient",
    }
    atomic(args.output, json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print("집계 결과 저장", args.output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
