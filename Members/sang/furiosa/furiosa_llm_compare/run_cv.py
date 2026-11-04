"""Run research-only patient-separated 5-fold Qwen3-32B comparisons."""

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
from run_server import CONDITIONS, MODEL, SCHEMA, atomic, build_messages, metric, parse_result


def write_rows(path, rows):
    columns = [
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
    writer = csv.DictWriter(s, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic(path, s.getvalue())


def main(args):
    raw = args.input.read_bytes()
    payload = json.loads(raw)
    data = payload["patients"]
    if len(data) != 356 or len({r["patient_id"] for r in data}) != 356:
        raise ValueError("cohort mismatch")
    assignments = {r["patient_id"]: r["fold"] for r in data}
    for r in data:
        if (
            r["original_split"] not in ("train", "validation")
            or r["fold"] not in (1, 2, 3, 4, 5)
            or len(r["examples"]) != 4
            or len(r["example_patient_ids"]) != 4
        ):
            raise ValueError("bad CV input")
        if sum(e["outcome_4y_scd"] for e in r["examples"]) != 2 or any(
            pid not in assignments or assignments[pid] == r["fold"]
            for pid in r["example_patient_ids"]
        ):
            raise ValueError("example leakage")
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
        "caveat": "research only; scores uncalibrated; no external test",
    }
    atomic(args.output / "metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
    rows = []
    print("모델 로딩:", MODEL, flush=True)
    try:
        with LLM(MODEL, devices=meta["npu_devices"]) as llm:
            params = SamplingParams(
                max_tokens=1024,
                temperature=0,
                structured_outputs=StructuredOutputsParams(json=SCHEMA),
            )
            for condition in CONDITIONS:
                for i, rec in enumerate(data, 1):
                    start = time.monotonic()
                    row = {
                        "patient_id": rec["patient_id"],
                        "original_split": rec["original_split"],
                        "fold": rec["fold"],
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
                        result = llm.generate([prompt], params)
                        row.update(p_positive=parse_result(result[0].outputs[0]), status="ok")
                    except (ValueError, TypeError, RuntimeError, KeyError, IndexError) as e:
                        row["error_type"] = type(e).__name__
                    row["seconds"] = round(time.monotonic() - start, 3)
                    rows.append(row)
                    write_rows(args.output / "predictions.csv", rows)
                    print(condition, f"{i}/{len(data)}", row["status"], flush=True)
    except Exception as e:
        meta["status"] = "FAILED"
        meta["error_type"] = type(e).__name__
        atomic(args.output / "metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
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
            "all_successful": metric(subset, len(data), "all successful"),
            "paired_successful": metric(
                [r for r in subset if r["patient_id"] in paired], len(paired), "paired only"
            ),
            "folds": {
                str(fold): metric(
                    [r for r in subset if r["fold"] == fold],
                    sum(r["fold"] == fold for r in data),
                    "fold",
                )
                for fold in range(1, 6)
            },
        }
    meta["status"] = (
        "COMPLETE"
        if len(rows) == len(data) * 3 and all(r["status"] == "ok" for r in rows)
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
    print("상태:", meta["status"], "동일 성공 표본:", len(paired), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
