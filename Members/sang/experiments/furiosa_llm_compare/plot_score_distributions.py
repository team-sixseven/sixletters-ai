"""Plot class-normalized score histograms without exporting patient predictions."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threshold_comparison import CONDITIONS, RUNS


def main(root, output):
    if output.exists():
        raise FileExistsError(output)
    baseline = json.loads((root / "run_20261102_cv5/metadata.json").read_text())["input_sha256"]
    diag = json.loads((root / "run_20261102_diagnostics/metadata.json").read_text())
    assert diag["input_metadata"]["baseline_cv_sha256"] == baseline
    prior = json.loads((root / "multimodel_aggregate_20261103.json").read_text())
    bins = np.linspace(0, 1, 21)
    names = {
        "qwen32": "Qwen3-32B",
        "gpt_oss": "GPT-OSS-20B",
        "qwen_moe": "Qwen3-30B-A3B",
        "exaone": "EXAONE-4.0-32B",
    }
    titles = {
        "holter": "Holter",
        "clinical_only": "Clinical",
        "holter_clinical": "Holter + clinical",
        "holter_clinical_cases": "Holter + clinical + cases",
    }
    fig, axes = plt.subplots(4, 4, figsize=(19, 14), sharex=True, sharey=True)
    expected = None
    for row, (name, run) in enumerate(RUNS.items()):
        meta = json.loads((root / run / "metadata.json").read_text())
        assert meta["status"] == "COMPLETE" and meta["input_sha256"] == baseline
        predictions = pd.read_csv(root / run / "predictions.csv")
        clinical = (
            pd.read_csv(root / "run_20261102_diagnostics/predictions.csv")
            if name == "qwen32"
            else None
        )
        for col, condition in enumerate(CONDITIONS):
            source = clinical if name == "qwen32" and condition == "clinical_only" else predictions
            g = source[(source.condition == condition) & (source.status == "ok")]
            assert len(g) == 356 and g.patient_id.nunique() == 356
            assert set(g.y_true.unique()) == {0, 1} and int(g.y_true.sum()) == 40
            assert np.isfinite(g.p_positive).all() and g.p_positive.between(0, 1).all()
            patients = g.set_index("patient_id").y_true.sort_index()
            if expected is None:
                expected = patients
            else:
                assert patients.equals(expected)
            ax = axes[row, col]
            counts = {}
            for label, color, display in [
                (0, "#3975a8", "SCD negative (n=316)"),
                (1, "#dc6659", "SCD positive (n=40)"),
            ]:
                values = g.loc[g.y_true == label, "p_positive"].to_numpy(float)
                count, _ = np.histogram(values, bins=bins)
                assert int(count.sum()) == len(values)
                counts[label] = count
                ax.stairs(
                    count * 100 / len(values),
                    bins * 100,
                    fill=True,
                    alpha=0.34,
                    color=color,
                    linewidth=1.1,
                    label=display,
                )
                ax.stairs(count * 100 / len(values), bins * 100, color=color, linewidth=1.2)
            cm = prior["conditions"][name][condition]["confusion_0_5"]
            assert int(counts[0][10:].sum()) == cm[0][1]
            assert int(counts[1][10:].sum()) == cm[1][1]
            ax.axvline(50, color="#555555", linestyle="--", linewidth=1)
            ax.grid(axis="y", alpha=0.2)
            ax.set_xlim(0, 100)
            if row == 0:
                ax.set_title(titles[condition], fontsize=13, weight="bold")
            if col == 0:
                ax.set_ylabel(names[name] + "\nPercent of class (%)", fontsize=10)
            if row == 3:
                ax.set_xlabel("Model score (%)")
    fig.suptitle(
        "SCD prediction score distributions by model and input", fontsize=18, weight="bold"
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.025),
    )
    fig.text(
        0.5,
        0.005,
        "5-point score bins; dashed line = 50% threshold. Scores are not calibrated risks.",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("Validated 16 histograms: matched cohort, class counts, 50% TP/FP:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
