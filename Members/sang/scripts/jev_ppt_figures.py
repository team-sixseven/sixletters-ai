"""Jev 실행 폴더의 결과로 PPT용 요약 표와 점수 히스토그램 PNG를 만든다.

사용: python jev_ppt_figures.py <실행 폴더>
입력: patient_predictions.csv, metrics.csv   출력: 같은 폴더의 ppt_table.png, ppt_histogram.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

NEGATIVE_COLOR, POSITIVE_COLOR = "#4C78A8", "#E45756"
VERSION_NAMES = {"clinical_holter": "임상 + Holter", "holter_only": "Holter만"}


def make_table(run_dir, metrics, valid):
    n_pos, n_neg = int((valid["y_true"] == 1).sum()), int((valid["y_true"] == 0).sum())
    detected, missed = int(metrics["positive_detected"]), int(metrics["positive_missed"])
    false_pos, true_neg = int(metrics["false_positive"]), int(metrics["true_negative"])

    def spread(label):
        v = valid.loc[valid["y_true"] == label, "p_positive"].astype(float)
        return f"{v.min():.2f} / {v.median():.2f} / {v.max():.2f}"

    rows = [
        ["입력", VERSION_NAMES.get(metrics["input"], metrics["input"])],
        ["평가 대상", f"test {len(valid)}명 (양성 {n_pos} / 음성 {n_neg})"],
        ["AUROC", f"{metrics['auroc']:.3f}"],
        ["AP (평균 정밀도)", f"{metrics['average_precision']:.3f}"],
        [f"양성 탐지 (기준 {metrics['threshold']})", f"{detected} / {detected + missed}"],
        ["거짓 양성", f"{false_pos} / {false_pos + true_neg}"],
        ["양성 점수 (최소/중앙/최대)", spread(1)],
        ["음성 점수 (최소/중앙/최대)", spread(0)],
    ]
    fig, ax = plt.subplots(figsize=(8, 0.55 * (len(rows) + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["항목", f"Jev 결과 ({metrics['model']})"],
        colWidths=[0.5, 0.5],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 1.8)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#BBBBBB")
        if row == 0:
            cell.set_facecolor("#2F3E54")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F2F4F7")
    fig.savefig(run_dir / "ppt_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_histogram(run_dir, metrics, valid):
    threshold = float(metrics["threshold"])
    upper = max(threshold + 0.1, float(valid["p_positive"].max()) + 0.05)
    bins = np.arange(0, upper + 0.05, 0.05)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, color, name in [(0, NEGATIVE_COLOR, "실제 음성"), (1, POSITIVE_COLOR, "실제 양성")]:
        values = valid.loc[valid["y_true"] == label, "p_positive"].astype(float)
        ax.hist(values, bins=bins, alpha=0.65, color=color, edgecolor="white", label=f"{name} (n={len(values)})")
    ax.axvline(threshold, color="black", linestyle="--", label=f"기준 {threshold}")
    ax.set(
        xlabel="Jev 점수 (양성 확률)",
        ylabel="환자 수",
        title=f"Jev 점수 분포 - {VERSION_NAMES.get(metrics['input'], metrics['input'])}",
        xlim=(0, upper),
    )
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "ppt_histogram.png", dpi=200)
    plt.close(fig)


def main(run_dir):
    run_dir = Path(run_dir)
    metrics = pd.read_csv(run_dir / "metrics.csv").iloc[0]
    predictions = pd.read_csv(run_dir / "patient_predictions.csv")
    valid = predictions[predictions["status"] == "ok"]
    if len(valid) != int(metrics["succeeded"]):
        raise ValueError("metrics.csv와 predictions의 성공 건수가 다릅니다.")
    make_table(run_dir, metrics, valid)
    make_histogram(run_dir, metrics, valid)
    print("저장:", run_dir / "ppt_table.png", run_dir / "ppt_histogram.png")


if __name__ == "__main__":
    main(sys.argv[1])
