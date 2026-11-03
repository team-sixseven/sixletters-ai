"""Report a completed, patient-level train-only Haiku experiment."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.run / "metadata.json").read_text())
    assert metadata["status"] == "COMPLETE" and metadata["split"] == "train"
    assert metadata["models"] == ["~anthropic/claude-haiku-5-5"] and metadata["sample_size"] == 285
    prediction_files = list(args.run.glob("*/predictions.csv"))
    assert len(prediction_files) == 1
    frame = pd.read_csv(prediction_files[0])
    assert len(frame) == 285
    assert frame.patient_id.nunique() == 285 and (frame.status == "ok").all()
    assert frame.y_true.sum() == 32 and frame.p_positive.between(0, 1).all()
    y = frame.y_true.to_numpy(int)
    score = frame.p_positive.to_numpy(float)
    tn, fp, fn, tp = confusion_matrix(y, score >= 0.5, labels=[0, 1]).ravel()
    auc, ap = roc_auc_score(y, score), average_precision_score(y, score)
    old = json.loads((prediction_files[0].parent / "metrics.json").read_text())
    assert old["confusion_matrix"] == [[int(tn), int(fp)], [int(fn), int(tp)]]
    assert abs(old["auroc"] - auc) < 1e-12 and abs(old["average_precision"] - ap) < 1e-12
    bins = np.linspace(0, 1, 21)
    neg, _ = np.histogram(score[y == 0], bins=bins)
    pos, _ = np.histogram(score[y == 1], bins=bins)
    assert int(neg.sum()) == 253 and int(pos.sum()) == 32
    hist = args.run / "score_distribution_bins.csv"
    with hist.open("x", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["score_from_percent", "score_to_percent", "negative_count", "positive_count"])
        for i in range(20):
            w.writerow([i * 5, (i + 1) * 5, neg[i], pos[i]])
    hist.chmod(0o600)
    fig, ax = plt.subplots(figsize=(10, 5.3))
    for counts, n, color, label in [
        (neg, 253, "#3975a8", "SCD negative (n=253)"),
        (pos, 32, "#dc6659", "SCD positive (n=32)"),
    ]:
        ax.stairs(counts * 100 / n, bins * 100, fill=True, alpha=0.35, color=color, label=label)
        ax.stairs(counts * 100 / n, bins * 100, color=color)
    ax.axvline(50, color="#555", linestyle="--", label="50% threshold")
    ax.set(
        xlabel="Model score (%)",
        ylabel="Percent within class (%)",
        xlim=(0, 100),
        title="Claude Haiku 5.5 · Holter train cohort",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    figure = args.run / "score_distribution.png"
    if figure.exists():
        raise FileExistsError(figure)
    fig.savefig(figure, dpi=170)
    plt.close(fig)
    figure.chmod(0o600)
    threshold_rows = []
    for t in [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        a, b, c, d = confusion_matrix(y, score >= t, labels=[0, 1]).ravel()
        threshold_rows.append(
            {
                "threshold": t,
                "tp": int(d),
                "fn": int(c),
                "fp": int(b),
                "tn": int(a),
                "sensitivity": float(d / (d + c)),
                "ppv": float(d / (d + b)) if d + b else None,
            }
        )
    for target in [0.8, 0.9]:
        for t in sorted(set(score), reverse=True):
            a, b, c, d = confusion_matrix(y, score >= t, labels=[0, 1]).ravel()
            if d / (d + c) >= target:
                threshold_rows.append(
                    {
                        "threshold": float(t),
                        "tp": int(d),
                        "fn": int(c),
                        "fp": int(b),
                        "tn": int(a),
                        "sensitivity": float(d / (d + c)),
                        "ppv": float(d / (d + b)) if d + b else None,
                        "target": target,
                    }
                )
                break
    thfile = args.run / "threshold_summary.csv"
    with thfile.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["target", "threshold", "tp", "fn", "fp", "tn", "sensitivity", "ppv"])
        w.writeheader()
        w.writerows(threshold_rows)
    thfile.chmod(0o600)
    report = f"""# Claude Haiku 5.5 Holter train 전체 추론

## 대상 및 방법

- MonoRouter `~anthropic/claude-haiku-5-5`, train 원본 285명, SCD 양성 32명·음성 253명. validation과 test는 추론하지 않았다.
- 기존 MonoRouter Holter 노트북의 RR/HRV 요약과 시스템 프롬프트를 사용했다. 정상박동 주석 없는 RR/HRV는 검증되지 않은 추정치다. 환자 ID와 정답은 API로 전송하지 않았다.
- 모델을 환자 데이터로 학습시키지 않았다. train 대상 후향적 결과이며 독립 검증 성능이 아니다. 모델 위험도 출력은 보정된 실제 확률이 아니다.
- 모델 요청 성공 285/285, temperature는 기존 노트북의 API 기본값, max_completion_tokens 1024, reasoning_effort low. 두 키 HTTP 402 전환을 지원한다. 순수 JSON과 JSON 코드 블록을 동일 스키마로 검증했다.

## 성능

| 지표 | 결과 |
|---|---:|
| AUROC | {auc:.4f} |
| AP | {ap:.4f} |
| 민감도 (50% 기준) | {tp}/32 = {tp / 32:.1%} |
| 특이도 (50% 기준) | {tn}/253 = {tn / 253:.1%} |
| 위양성 / 위음성 | {fp} / {fn} |
| 점수 범위 | {score.min() * 100:.2f}% ~ {score.max() * 100:.2f}% |

## 분포도와 임계값

![실제 양성·음성별 점수 분포](../results/monorouter_holter/{args.run.name}/score_distribution.png)

파랑은 실제 음성 253명, 빨강은 실제 양성 32명이며 각 집단 내 비율을 보여준다. 점선은 50% 점수 기준이다. 5%p 구간으로 묶었다. [구간별 인원 CSV](../results/monorouter_holter/{args.run.name}/score_distribution_bins.csv), [임계값 CSV](../results/monorouter_holter/{args.run.name}/threshold_summary.csv).

민감도 80%·90% 행은 같은 train 환자에서 사후 선택된 기준이며 동점으로 목표를 넘을 수 있다. 별도 검증이나 임상 임계값이 아니다. GPT-6-Luna의 기존 356명 결과는 validation 71명도 포함해 직접 수치를 비교할 수 없다.

## 결과 및 검증

- 환자별 예측 CSV: `{prediction_files[0]}`. 식별자와 라벨이 포함되므로 접근을 제한한다.
- 실행 노트북: `Members/sang/notebooks/monorouter_holter_claude_haiku55_train.ipynb`.
- 원본 환자·예측 중복, 285명, 실제 양성 32명, 점수 범위와 성공 상태, 독립 재계산 AUROC·AP·혼동행렬 일치, 히스토그램 합계를 확인했다.
- 지표는 연구용이며 임상 의사결정에 사용할 수 없다.
"""
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.write_text(report, encoding="utf-8")
    print(
        "REPORT_COMPLETE",
        args.report,
        "AUROC",
        round(auc, 4),
        "AP",
        round(ap, 4),
        "TP/FP",
        int(tp),
        int(fp),
    )


if __name__ == "__main__":
    main()
