"""Build research metrics, plots and demo data from completed patient-level runs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

MODELS = ("gpt_oss", "qwen32", "qwen_moe", "exaone")
CONDITIONS = ("holter", "clinical_only", "holter_clinical", "holter_clinical_cases")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    src = json.loads((a.root / "input_summary30.json").read_text())
    cohort = {r["patient_id"]: r for r in src["patients"]}
    if len(cohort) != 356 or sum(x["label"] for x in cohort.values()) != 40:
        raise ValueError("cohort mismatch")
    candidates = json.loads((a.root / "review_candidates.json").read_text())
    a.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    predictions, metrics = [], []
    for model in MODELS:
        run = a.root / f"run_{model}"
        meta = json.loads((run / "metadata.json").read_text())
        if meta["status"] != "COMPLETE":
            raise ValueError(f"{model} is not complete")
        with (run / "predictions.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        for condition in CONDITIONS:
            group = [r for r in rows if r["condition"] == condition and r["status"] == "ok"]
            if len(group) != 356 or {r["patient_id"] for r in group} != set(cohort):
                raise ValueError("incomplete or duplicate patient predictions")
            y = np.array([int(r["y_true"]) for r in group])
            scores = np.array([float(r["p_positive"]) for r in group])
            if (
                any(int(r["y_true"]) != cohort[r["patient_id"]]["label"] for r in group)
                or not np.isfinite(scores).all()
                or ((scores < 0) | (scores > 1)).any()
            ):
                raise ValueError("invalid label or score")
            positive = scores >= 0.5
            metrics.append(
                {
                    "model": model,
                    "condition": condition,
                    "patients": len(group),
                    "positive": int(y.sum()),
                    "negative": int(len(y) - y.sum()),
                    "auroc": float(roc_auc_score(y, scores)),
                    "ap": float(average_precision_score(y, scores)),
                    "tp": int(((y == 1) & positive).sum()),
                    "fn": int(((y == 1) & ~positive).sum()),
                    "fp": int(((y == 0) & positive).sum()),
                    "tn": int(((y == 0) & ~positive).sum()),
                }
            )
            predictions.extend(
                {
                    "model": model,
                    "condition": condition,
                    "patient_id": r["patient_id"],
                    "fold": int(r["fold"]),
                    "y_true": int(r["y_true"]),
                    "score": float(r["p_positive"]),
                }
                for r in group
            )
    for name, rows in [("model_metrics.csv", metrics), ("patient_predictions.csv", predictions)]:
        with (a.output / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    fig, axes = plt.subplots(4, 4, figsize=(17, 12), sharex=True, constrained_layout=True)
    for ax, (model, condition) in zip(
        axes.flat, ((m, c) for m in MODELS for c in CONDITIONS), strict=False
    ):
        for label, color in ((0, "steelblue"), (1, "tomato")):
            vals = [
                r["score"]
                for r in predictions
                if r["model"] == model and r["condition"] == condition and r["y_true"] == label
            ]
            ax.hist(
                vals,
                bins=np.linspace(0, 1, 21),
                alpha=0.55,
                color=color,
                label=f"label {label}",
                weights=np.ones(len(vals)) / len(vals),
            )
        ax.axvline(0.5, color="black", ls="--", lw=0.8)
        ax.set_title(model + " / " + condition)
        ax.set_xlim(0, 1)
    axes[0, 0].legend()
    fig.suptitle("Patient-level score distributions (class-normalized)")
    fig.savefig(a.output / "score_distributions.png", dpi=150)
    plt.close(fig)
    demo = {}
    for pid in cohort:
        scores = {
            r["model"] + "__" + r["condition"]: r["score"]
            for r in predictions
            if r["patient_id"] == pid
        }
        demo[pid] = {
            "patient_id": pid,
            "scores": scores,
            "review_candidates": candidates[pid]["segments"],
            "explanation": None,
            "explanation_status": "not_generated",
            "waveform": {
                "sample_rate_hz": 200,
                "duration_seconds": 30,
                "source": "original recording; read on demand",
            },
            "caveat": "Research only. Scores are uncalibrated; review candidates are not diagnosed risk or model attribution.",
        }
    (a.output / "demo_outputs.json").write_text(
        json.dumps(demo, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    lines = [
        "# 30초 구간 전체 요약 기반 모델 비교",
        "",
        "356명(양성 40명·음성 316명)의 품질 통과 30초 구간에서 RR 및 심박수 통계를 산출하고 환자별 중앙값·5/95백분위로 요약했다.",
        "환자별 네 가지 입력 조건을 평가했으며 사례는 다른 폴드에서 선택했다. 이미 검토된 개발 데이터이므로 독립적인 시험 결과가 아니다.",
        "점수는 보정된 임상 확률이 아니며 50%는 연구 비교 기준이다. 특징 실패 구간과 품질 제외 구간은 사용하지 않았다.",
        "구간 후보는 환자 내 심박수 최저·최고 및 RR 표준편차 최대의 기술적 극단값이며 SCD 구간 판정이나 모델 기여도가 아니다.",
        "",
        "| 모델 | 입력 | 환자 | AUROC | AP | 양성 탐지 | 놓친 양성 | 위양성 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for x in metrics:
        lines.append(
            f"| {x['model']} | {x['condition']} | {x['patients']} | {x['auroc']:.3f} | {x['ap']:.3f} | {x['tp']}/{x['positive']} | {x['fn']} | {x['fp']}/{x['negative']} |"
        )
    (a.output / "결과_요약.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        "COMPLETE metrics",
        len(metrics),
        "patient_predictions",
        len(predictions),
        "demo_patients",
        len(demo),
    )


if __name__ == "__main__":
    main()
