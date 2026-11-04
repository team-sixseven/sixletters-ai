"""Create a local visual report from the adapter experiment's metrics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = Path("Members/sang/results/ecg_fm_adapter")
r = json.loads((out / "metrics.json").read_text())
old = json.loads(Path("Members/sang/results/ecg_fm_pilot/metrics.json").read_text())
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
parts = ["train", "validation", "test"]
x = np.arange(3)
width = 0.36
for j, (key, label, color) in enumerate(
    [("auroc", "AUROC", "#426bdd"), ("average_precision", "Average precision", "#9b63cc")]
):
    vals = [r["splits"][part][key] for part in parts]
    bars = axes[0].bar(x + (j - 0.5) * width, vals, width, label=label, color=color)
    axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
axes[0].set(
    xticks=x,
    xticklabels=["Train", "Validation", "Test"],
    ylim=(0, 1.1),
    title="Patient-level metrics",
)
axes[0].legend(fontsize=8)
h = r["history"]
axes[1].plot(
    [a["epoch"] for a in h],
    [a["validation"]["average_precision"] for a in h],
    marker="o",
    color="#9b63cc",
)
axes[1].axvline(r["best_epoch"], color="#d17b2f", ls="--", label="Selected epoch")
axes[1].set(title="Validation AP during training", xlabel="Epoch", ylabel="AP", ylim=(0, 1))
axes[1].legend(fontsize=8)
c = r["splits"]["test"]["confusion"]
matrix = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
axes[2].imshow(matrix, cmap="Blues")
axes[2].set(
    title="Test confusion matrix",
    xticks=[0, 1],
    xticklabels=["Negative", "Positive"],
    yticks=[0, 1],
    yticklabels=["Negative", "Positive"],
    xlabel="Predicted",
    ylabel="Actual",
)
for (i, j), v in np.ndenumerate(matrix):
    axes[2].text(
        j,
        i,
        str(v),
        ha="center",
        va="center",
        color="white" if v > matrix.max() / 2 else "#19263a",
        fontsize=15,
    )
fig.savefig(out / "performance.png", dpi=150)
plt.close(fig)
labels = {"train": "학습", "validation": "검증", "test": "테스트"}


def precision_text(v):
    return "정의되지 않음" if v is None else f"{v:.3f}"


rows = "".join(
    f"<tr><td>{labels[k]}</td><td>{v['n']} / {v['positive']}</td><td>{v['auroc']:.3f}</td><td>{v['average_precision']:.3f}</td><td>{precision_text(v['precision'])}</td><td>{v['sensitivity']:.3f}</td><td>{v['specificity']:.3f}</td></tr>"
    for k, v in r["splits"].items()
)
v = r["splits"]["test"]
o = old["splits"]["test"]
cm = v["confusion"]
page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ECG-FM 채널 변환 실험</title><style>body{{font-family:system-ui,"Malgun Gothic",sans-serif;background:#f4f7fb;color:#18263b;line-height:1.65;margin:0}}main{{max-width:1100px;margin:auto;padding:40px 24px}}h1{{font-size:30px}}section{{background:white;border:1px solid #e1e8f1;border-radius:14px;padding:22px;margin:16px 0;overflow-x:auto}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{border-bottom:1px solid #e8edf3;text-align:right;padding:10px}}th:first-child,td:first-child{{text-align:left}}.note{{background:#fff7e7;border-left:4px solid #e2a33f}}.muted{{color:#637086}}strong{{color:#3857ad}}</style></head><body><main><h1>ECG-FM 3채널 변환 실험</h1><p class="muted">로컬 GPU 학습 · 선택 epoch {r["best_epoch"]} · 2026-10-30</p><section><h2>결과 시각화</h2><img src="performance.png" alt="학습, 검증, 테스트 성능과 검증 AP 학습 곡선, 테스트 혼동행렬"></section><section><h2>환자 단위 평가</h2><table><thead><tr><th>분할</th><th>전체 / 양성</th><th>AUROC</th><th>AP</th><th>Precision</th><th>Sensitivity</th><th>Specificity</th></tr></thead><tbody>{rows}</tbody></table><p>Precision, sensitivity, specificity는 0.5 기준입니다. 양성 예측이 없으면 precision은 정의되지 않습니다. 테스트에서 양성으로 예측한 {cm["tp"] + cm["fp"]}명 중 실제 양성은 {cm["tp"]}명이며, 실제 양성 {v["positive"]}명 중 {cm["tp"]}명을 찾았습니다.</p></section><section><h2>이전 실험과 참고 비교</h2><p>테스트 AUROC: 0 채우기 <strong>{o["auroc"]:.3f}</strong> → 학습된 변환층 <strong>{v["auroc"]:.3f}</strong><br>테스트 AP: 0 채우기 <strong>{o["average_precision"]:.3f}</strong> → 학습된 변환층 <strong>{v["average_precision"]:.3f}</strong></p><p class="muted">환자와 입력 구간은 같지만 분류기와 학습 방식도 다르므로 변화가 채널 변환층만의 효과라고 판단할 수 없습니다. 이 테스트 세트는 이전 실험에서도 사용되어 독립적인 최종 검증 세트가 아닙니다.</p></section><section class="note"><h2>실험 조건과 한계</h2><p>X/Y/Z에서 12개 내부 채널로 가는 학습 가능한 변환층과 SCD 분류기를 학습하고 ECG-FM 본체는 고정했습니다. 학습 가능한 채널은 실제 12유도를 복원하지 않습니다. 샘플별 최대 8개 구간을 사용했고, 모델 선택은 검증 AP로 했습니다.</p><p>테스트 양성은 {v["positive"]}명으로 불확실성이 크며, 단일 seed의 탐색 실험입니다. 이 결과는 임상 진단용이 아닙니다.</p></section></main></body></html>"""
(out / "report.html").write_text(page, encoding="utf-8")
print(out / "report.html")
