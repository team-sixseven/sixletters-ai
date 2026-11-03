"""Research-only sequential Qwen3-32B Holter/clinical/few-shot evaluation."""

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from furiosa_llm import LLM, SamplingParams
from furiosa_llm.sampling_params import StructuredOutputsParams
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score

MODEL = "furiosa-ai/Qwen3-32B-FP8"
CONDITIONS = ("holter", "holter_clinical", "holter_clinical_cases")
SCHEMA = {
    "type": "object",
    "properties": {"risk_percent": {"type": "number"}},
    "required": ["risk_percent"],
    "additionalProperties": False,
}
SYSTEM = (
    "This is retrospective research, not clinical care. Given experimental Holter-derived RR and HRV estimates, "
    "estimate the percentage probability of sudden cardiac death within four years. "
    "Normal beat classification and these features are not validated; null means unavailable. "
    "Do not infer identities or invent missing measurements. "
    "Example patients, if present, were selected by class and are not representative of outcome prevalence. "
    "Return only a JSON object with the single key risk_percent and a finite number from 0 to 100."
)


def atomic(path, data):
    fd, temp = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def build_messages(rec, condition):
    data = {"ecg_summary": rec["holter"]}
    if condition != "holter":
        data["clinical"] = rec["clinical"]
    if condition == "holter_clinical_cases":
        data["training_examples"] = rec["examples"]
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": json.dumps(data, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
        },
    ]


def parse_result(output):
    if output.finish_reason != "stop":
        raise ValueError("incomplete")
    value = json.loads(output.text)
    if not isinstance(value, dict) or set(value) != {"risk_percent"}:
        raise ValueError("schema")
    n = value["risk_percent"]
    if type(n) not in (float, int) or not math.isfinite(n) or not 0 <= n <= 100:
        raise ValueError("range")
    return float(n) / 100


def write_rows(path, rows):
    fields = ["patient_id", "y_true", "condition", "p_positive", "status", "error_type", "seconds"]
    import io

    s = io.StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
    atomic(path, s.getvalue())


def metric(rows, expected, kind):
    good = [r for r in rows if r["status"] == "ok"]
    result = {
        "population": kind,
        "expected": expected,
        "succeeded": len(good),
        "complete": len(good) == expected,
    }
    if not good:
        return result
    y = np.array([r["y_true"] for r in good])
    p = np.array([r["p_positive"] for r in good])
    predictions = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel().tolist()
    result.update(
        positive=int(y.sum()),
        negative=int(len(y) - y.sum()),
        accuracy=float(accuracy_score(y, predictions)),
        recall=tp / (tp + fn) if tp + fn else None,
        specificity=tn / (tn + fp) if tn + fp else None,
        confusion_matrix=[[tn, fp], [fn, tp]],
        auroc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        average_precision=float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
    )
    return result


def main(args):
    raw = args.input.read_bytes()
    payload = json.loads(raw)
    data = payload["validation"]
    if (
        len(data) != 71
        or len({x["patient_id"] for x in data}) != 71
        or any(x["split"] != "validation" for x in data)
    ):
        raise ValueError("validation cohort mismatch")
    if any(
        len(x["examples"]) != 4 or sum(e["outcome_4y_scd"] for e in x["examples"]) != 2
        for x in data
    ):
        raise ValueError("training examples mismatch")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    meta = {
        "status": "RUNNING",
        "model": MODEL,
        "conditions": CONDITIONS,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "created_utc": datetime.now(UTC).isoformat(),
        "temperature": 0,
        "max_tokens": 1024,
        "thinking": False,
        "structured_output": "json_schema",
        "npu_devices": "npu:4,npu:5,npu:6,npu:7",
        "input_metadata": payload["metadata"],
        "caveat": "research only; uncalibrated scores; validation also used for comparison; not clinical care",
    }
    atomic(args.output / "metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
    print("모델 로딩:", MODEL, flush=True)
    rows = []
    try:
        with LLM(MODEL, devices=meta["npu_devices"]) as llm:
            for condition in CONDITIONS:
                for i, rec in enumerate(data, 1):
                    start = time.monotonic()
                    row = {
                        "patient_id": rec["patient_id"],
                        "y_true": rec["label"],
                        "condition": condition,
                        "p_positive": None,
                        "status": "error",
                        "error_type": None,
                        "seconds": None,
                    }
                    try:
                        prompt = llm.tokenizer.apply_chat_template(
                            build_messages(rec, condition),
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        params = SamplingParams(
                            max_tokens=1024,
                            temperature=0,
                            structured_outputs=StructuredOutputsParams(json=SCHEMA),
                        )
                        result = llm.generate([prompt], params)
                        row.update(p_positive=parse_result(result[0].outputs[0]), status="ok")
                    except (ValueError, TypeError, RuntimeError, KeyError, IndexError) as e:
                        row["error_type"] = type(e).__name__
                    row["seconds"] = round(time.monotonic() - start, 3)
                    rows.append(row)
                    write_rows(args.output / "predictions.csv", rows)
                    print(condition, f"{i}/{len(data)}", row["status"], flush=True)
    except Exception as exc:
        meta["status"] = "FAILED"
        meta["error_type"] = type(exc).__name__
        atomic(args.output / "metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
        raise
    complete_ids = set.intersection(
        *[
            {r["patient_id"] for r in rows if r["condition"] == c and r["status"] == "ok"}
            for c in CONDITIONS
        ]
    )
    report = {}
    for condition in CONDITIONS:
        subset = [r for r in rows if r["condition"] == condition]
        report[condition] = {
            "all_successful": metric(subset, len(data), "all successful"),
            "paired_successful": metric(
                [r for r in subset if r["patient_id"] in complete_ids],
                len(complete_ids),
                "paired only",
            ),
        }
    meta["status"] = (
        "COMPLETE"
        if len(rows) == len(data) * len(CONDITIONS) and all(r["status"] == "ok" for r in rows)
        else "PARTIAL"
    )
    meta["paired_successful"] = len(complete_ids)
    atomic(
        args.output / "metrics.json",
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
    )
    atomic(
        args.output / "metadata.json",
        json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False),
    )
    print("상태:", meta["status"], "동일 성공 표본:", len(complete_ids), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
