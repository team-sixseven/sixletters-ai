"""전체 445명 LLM 실험(Furiosa 4모델, Haiku, Terra)의 성능표를 만든다.

입력은 환자별 예측 CSV이며, 성공한 응답만 집계하고 시도·성공 건수를 함께 적는다.
전체 445명 기준과 원본 test 89명 기준을 나눠서 계산한다. 점수는 보정된 확률이 아니다.

사용: python summarize_llm_full445.py --split-csv final_split.csv --furiosa-dir DIR \
    --haiku predictions.csv --terra predictions.csv --output-dir OUT
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

FURIOSA_MODELS = {
    "qwen32": "Qwen3-32B",
    "qwen_moe": "Qwen3-30B-A3B",
    "exaone": "EXAONE-4.0-32B",
    "gpt_oss": "GPT-OSS-20B",
}
FURIOSA_CONDITIONS = ("holter", "clinical_only", "holter_clinical", "holter_clinical_cases")
THRESHOLD = 0.5
FIELDS = [
    "model",
    "input",
    "subset",
    "expected",
    "attempted",
    "succeeded",
    "positives",
    "auroc",
    "average_precision",
    "tn",
    "fp",
    "fn",
    "tp",
    "sensitivity",
    "specificity",
]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_splits(path):
    splits = {}
    for row in read_csv(path):
        name = "validation" if row["split"] == "val" else row["split"]
        splits[row["pid"]] = name
    return splits


def summarize(rows, expected):
    """rows: 환자별 {y_true, p_positive, status}. 성공한 행만 지표에 사용한다."""
    ok = [r for r in rows if r["status"] == "ok" and r["p_positive"] not in ("", None)]
    out = {"expected": expected, "attempted": len(rows), "succeeded": len(ok)}
    if not ok:
        return out
    y = np.array([int(r["y_true"]) for r in ok])
    p = np.array([float(r["p_positive"]) for r in ok])
    predicted = p >= THRESHOLD
    tp = int((predicted & (y == 1)).sum())
    fn = int((~predicted & (y == 1)).sum())
    fp = int((predicted & (y == 0)).sum())
    tn = int((~predicted & (y == 0)).sum())
    out.update(positives=int(y.sum()), tn=tn, fp=fp, fn=fn, tp=tp)
    out["sensitivity"] = tp / (tp + fn) if tp + fn else None
    out["specificity"] = tn / (tn + fp) if tn + fp else None
    if len(set(y.tolist())) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["average_precision"] = float(average_precision_score(y, p))
    return out


def subsets(rows, splits):
    """전체와 test 부분집합으로 나눈다. 환자 수는 코호트 기준(전체 445, test 89)이다."""
    test_rows = [r for r in rows if splits[r["patient_id"]] == "test"]
    return {
        "all": (rows, len(splits)),
        "test": (test_rows, sum(v == "test" for v in splits.values())),
    }


def main(args):
    splits = read_splits(args.split_csv)
    result = []

    def add(model, input_name, rows):
        for subset, (part, expected) in subsets(rows, splits).items():
            result.append(
                {"model": model, "input": input_name, "subset": subset, **summarize(part, expected)}
            )

    for key, label in FURIOSA_MODELS.items():
        rows = read_csv(Path(args.furiosa_dir) / key / "predictions.csv")
        for condition in FURIOSA_CONDITIONS:
            add(label, condition, [r for r in rows if r["condition"] == condition])
    add("Claude Haiku 5.5", "holter", read_csv(args.haiku))
    add("GPT-6.1-Terra", "holter", read_csv(args.terra))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(result)
    (out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"{len(result)}행 저장:", out_dir / "summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--furiosa-dir", type=Path, required=True)
    parser.add_argument("--haiku", type=Path, required=True)
    parser.add_argument("--terra", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args())
