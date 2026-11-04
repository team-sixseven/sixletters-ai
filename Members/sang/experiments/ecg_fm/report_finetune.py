"""Render fine-tuning metrics and two patient-level decision thresholds."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("Members/sang/results/ecg_fm_finetune"))
p = parser.parse_args().output
d = json.loads((p / "metrics.json").read_text())
s = d["splits"]
h = d["history"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
parts = ["train", "validation", "test"]
x = np.arange(3)
for off, key, label, color in [
    (-0.17, "auroc", "AUROC", "#436cd7"),
    (0.17, "average_precision", "AP", "#9662c5"),
]:
    bars = axes[0].bar(
        x + off, [s[k]["fixed_0_5"][key] for k in parts], 0.34, label=label, color=color
    )
    axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
axes[0].set(
    xticks=x,
    xticklabels=["Train", "Validation", "Test"],
    ylim=(0, 1.1),
    title="Patient-level ranking",
)
axes[0].legend()
axes[1].plot(
    [x["epoch"] for x in h],
    [x["validation"]["average_precision"] for x in h],
    marker="o",
    color="#9662c5",
)
axes[1].axvline(d["best_epoch"], ls="--", color="#de8a36", label="Selected")
axes[1].set(title="Validation AP", xlabel="Epoch", ylabel="AP", ylim=(0, 1))
axes[1].legend()
c = s["test"]["validation_f1_threshold"]["confusion"]
m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
axes[2].imshow(m, cmap="Blues")
axes[2].set(
    title="Test, validation F1 threshold",
    xticks=[0, 1],
    xticklabels=["Negative", "Positive"],
    yticks=[0, 1],
    yticklabels=["Negative", "Positive"],
    xlabel="Predicted",
    ylabel="Actual",
)
for (i, j), v in np.ndenumerate(m):
    axes[2].text(
        j,
        i,
        str(v),
        ha="center",
        va="center",
        color="white" if v > m.max() / 2 else "#172033",
        fontsize=15,
    )
fig.savefig(p / "performance.png", dpi=150)
plt.close(fig)


def fmt(x):
    return "정의되지 않음" if x is None else f"{x:.3f}"


rows = "".join(
    f"<tr><td>{label}</td><td>{val['n']} / {val['positive']}</td><td>{val['auroc']:.3f}</td><td>{val['average_precision']:.3f}</td><td>{fmt(val['precision'])}</td><td>{val['recall']:.3f}</td><td>{val['specificity']:.3f}</td></tr>"
    for label, val in [
        ("검증 · 0.5", s["validation"]["fixed_0_5"]),
        ("검증 · 선택 기준", s["validation"]["validation_f1_threshold"]),
        ("테스트 · 0.5", s["test"]["fixed_0_5"]),
        ("테스트 · 선택 기준", s["test"]["validation_f1_threshold"]),
    ]
)
page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ECG-FM 미세조정 결과</title><style>body{{font-family:system-ui,"Malgun Gothic",sans-serif;background:#f4f7fb;color:#19263b;line-height:1.65;margin:0}}main{{max-width:1100px;margin:auto;padding:40px 24px}}section{{background:#fff;border:1px solid #e1e8f1;border-radius:14px;padding:22px;margin:16px 0;overflow-x:auto}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #e7edf4}}th:first-child,td:first-child{{text-align:left}}.note{{background:#fff8e9;border-left:4px solid #dda341}}.muted{{color:#617087}}</style></head><body><main><h1>ECG-FM 미세조정 결과</h1><p class="muted">로컬 GPU · 마지막 2개 인코더 층 동결 · epoch {d["best_epoch"]} 선택</p><section><h2>결과 시각화</h2><img src="performance.png" alt="환자 단위 지표, 검증 AP 학습 곡선 및 테스트 혼동행렬"></section><section><h2>환자 단위 평가</h2><p>선택 기준은 검증 데이터에서 F1을 최대화한 {d["validation_selected_threshold"]:.6f}입니다.</p><table><thead><tr><th>분할 및 양성 판정 기준</th><th>전체 / 양성</th><th>AUROC</th><th>AP</th><th>Precision</th><th>Recall</th><th>Specificity</th></tr></thead><tbody>{rows}</tbody></table><p>양성 예측이 없으면 precision은 정의되지 않습니다. AUROC와 AP는 판정 기준에 영향받지 않습니다.</p></section><section class="note"><h2>해석 주의</h2><p>환자당 최대 8개 구간을 사용했고 인코더 {d["trainable_layer_indices"]}번 층, 채널 변환층과 분류기만 학습했습니다. 마지막 {d["frozen_last_layer_indices"]}번 층을 포함한 나머지 인코더는 동결했습니다. 학습된 12채널은 표준 12유도를 복원한 것이 아닙니다.</p><p>검증 양성은 8명, 테스트 양성은 10명입니다. 작은 검증 세트에서 선택한 기준은 불안정할 수 있습니다. 테스트 세트가 앞선 실험에도 쓰여 독립적인 최종 검증이 아닙니다. 단일 seed 탐색 결과이며 진단에 사용할 수 없습니다.</p></section></main></body></html>"""
(p / "report.html").write_text(page, encoding="utf-8")
print(p / "report.html")
