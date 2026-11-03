"""Run one Furiosa model on the fixed 356-patient comparison inputs."""

import argparse
import csv
import hashlib
import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from furiosa_llm import LLM, SamplingParams
from furiosa_llm.sampling_params import StructuredOutputsParams
from run_server import SCHEMA, SYSTEM, atomic, build_messages, metric, parse_result

MODELS = {
    "qwen32": {
        "id": "furiosa-ai/Qwen3-32B-FP8",
        "devices": "npu:4,npu:5,npu:6,npu:7",
        "template": "qwen_instruct",
    },
    "qwen_moe": {
        "id": "furiosa-ai/Qwen3-30B-A3B-Instruct-2507-FP8",
        "devices": "npu:4,npu:5,npu:6,npu:7",
        "template": "qwen_instruct",
    },
    "exaone": {
        "id": "furiosa-ai/EXAONE-4.0-32B-FP8",
        "devices": "npu:4,npu:5,npu:6,npu:7",
        "template": "exaone",
    },
    "gpt_oss": {"id": "furiosa-ai/gpt-oss-20b", "devices": "npu:4", "template": "gpt_oss"},
}
CONDITIONS = ("holter", "clinical_only", "holter_clinical", "holter_clinical_cases")


def messages(r, condition):
    if condition == "clinical_only":
        obj = {"ecg_summary": None, "clinical": r["clinical"]}
        return [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    obj, ensure_ascii=True, allow_nan=False, separators=(",", ":")
                ),
            },
        ]
    return build_messages(r, condition)


def write_rows(path, rows):
    fields = [
        "patient_id",
        "original_split",
        "fold",
        "y_true",
        "condition",
        "p_positive",
        "status",
        "error_type",
        "seconds",
    ]
    s = io.StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
    atomic(path, s.getvalue())


def main(args):
    cfg = MODELS[args.model]
    raw = args.input.read_bytes()
    data = json.loads(raw)
    patients = data["patients"]
    by = {r["patient_id"]: r for r in patients}
    if len(patients) != 356 or len(by) != 356:
        raise ValueError("cohort mismatch")
    for r in patients:
        if r["original_split"] not in ("train", "validation") or r["fold"] not in (1, 2, 3, 4, 5):
            raise ValueError("invalid fold")
        if len(r["example_patient_ids"]) != 4 or any(
            i not in by or by[i]["fold"] == r["fold"] for i in r["example_patient_ids"]
        ):
            raise ValueError("case leakage")
        if len(r["examples"]) != 4 or sum(e["outcome_4y_scd"] for e in r["examples"]) != 2:
            raise ValueError("case label mismatch")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    meta = {
        "status": "RUNNING",
        "model": cfg["id"],
        "devices": cfg["devices"],
        "conditions": CONDITIONS,
        "temperature": 0,
        "max_tokens": 1024,
        "template": cfg["template"],
        "structured_output": "json_schema",
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "created_utc": datetime.now(UTC).isoformat(),
        "input_metadata": data["metadata"],
        "caveat": "research only; outputs uncalibrated; comparison uses previously examined development data",
    }
    atomic(args.output / "metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
    rows = []
    try:
        print("모델 로딩:", cfg["id"], flush=True)
        with LLM(cfg["id"], devices=cfg["devices"]) as llm:
            params = SamplingParams(
                max_tokens=1024,
                temperature=0,
                structured_outputs=StructuredOutputsParams(json=SCHEMA),
            )
            for condition in CONDITIONS:
                for i, r in enumerate(patients, 1):
                    start = time.monotonic()
                    row = {
                        "patient_id": r["patient_id"],
                        "original_split": r["original_split"],
                        "fold": r["fold"],
                        "y_true": r["label"],
                        "condition": condition,
                        "p_positive": None,
                        "status": "error",
                        "error_type": None,
                        "seconds": None,
                    }
                    try:
                        options = {"tokenize": False, "add_generation_prompt": True}
                        if cfg["template"] in ("qwen_instruct", "exaone"):
                            options["enable_thinking"] = False
                        if cfg["template"] == "gpt_oss":
                            options["reasoning_effort"] = "low"
                        prompt = llm.tokenizer.apply_chat_template(
                            messages(r, condition), **options
                        )
                        out = llm.generate([prompt], params)[0].outputs[0]
                        row.update(p_positive=parse_result(out), status="ok")
                    except (ValueError, TypeError, RuntimeError, KeyError, IndexError) as e:
                        row["error_type"] = type(e).__name__
                    row["seconds"] = round(time.monotonic() - start, 3)
                    rows.append(row)
                    write_rows(args.output / "predictions.csv", rows)
                    print(
                        condition,
                        f"{i}/{len(patients)}",
                        row["status"],
                        row["error_type"] or "",
                        flush=True,
                    )
                    if i == 1 and row["status"] != "ok":
                        raise RuntimeError("first request failed; stopping condition/model")
    except Exception as e:
        meta["status"] = "FAILED"
        meta["error_type"] = type(e).__name__
        atomic(args.output / "metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
        print("MODEL_FAILED", type(e).__name__, flush=True)
        raise
    paired = set.intersection(
        *[
            {r["patient_id"] for r in rows if r["condition"] == c and r["status"] == "ok"}
            for c in CONDITIONS
        ]
    )
    report = {}
    for condition in CONDITIONS:
        subset = [r for r in rows if r["condition"] == condition]
        report[condition] = {
            "all_successful": metric(subset, len(patients), "all successful"),
            "paired_successful": metric(
                [r for r in subset if r["patient_id"] in paired], len(paired), "paired only"
            ),
            "folds": {
                str(f): metric(
                    [r for r in subset if r["fold"] == f],
                    sum(r["fold"] == f for r in patients),
                    "fold",
                )
                for f in range(1, 6)
            },
        }
    meta["status"] = (
        "COMPLETE"
        if len(rows) == len(patients) * len(CONDITIONS) and all(r["status"] == "ok" for r in rows)
        else "PARTIAL"
    )
    meta["paired_successful"] = len(paired)
    atomic(
        args.output / "metrics.json",
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
    )
    atomic(
        args.output / "metadata.json",
        json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False),
    )
    print("MODEL_DONE", meta["status"], "paired", len(paired), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
