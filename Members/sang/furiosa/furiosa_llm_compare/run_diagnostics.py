"""Run controlled Qwen input and retrieval ablations on fixed patient folds."""

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
from run_server import MODEL, SCHEMA, SYSTEM, atomic, metric, parse_result

CONDITIONS = (
    "clinical_only",
    "random_101",
    "random_202",
    "random_303",
    "hybrid",
    "nearest_unbalanced",
)


def messages(rec, condition):
    if condition == "clinical_only":
        data = {"ecg_summary": None, "clinical": rec["clinical"]}
    else:
        data = {
            "ecg_summary": rec["holter"],
            "clinical": rec["clinical"],
            "training_examples": rec["alternatives"][condition]["examples"],
        }
    system = SYSTEM
    if condition == "nearest_unbalanced":
        system = system.replace(
            "Example patients, if present, were selected by class and are not representative of outcome prevalence.",
            "Example patients, if present, were selected by similarity and may not represent outcome prevalence.",
        )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(data, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
        },
    ]


def write_rows(path, rows):
    fields = [
        "patient_id",
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
    raw = args.input.read_bytes()
    p = json.loads(raw)
    rows = p["patients"]
    by = {r["patient_id"]: r for r in rows}
    if len(rows) != 356 or len(by) != 356:
        raise ValueError("cohort mismatch")
    for r in rows:
        for _key, a in r["alternatives"].items():
            if (
                len(a["examples"]) != 4
                or len(a["example_patient_ids"]) != 4
                or any(i not in by or by[i]["fold"] == r["fold"] for i in a["example_patient_ids"])
            ):
                raise ValueError("case leakage")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    meta = {
        "status": "RUNNING",
        "model": MODEL,
        "conditions": CONDITIONS,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "created_utc": datetime.now(UTC).isoformat(),
        "temperature": 0,
        "thinking": False,
        "structured_output": "json_schema",
        "npu_devices": "npu:4,npu:5,npu:6,npu:7",
        "input_metadata": p["metadata"],
        "caveat": "research only; no external validation",
    }
    atomic(args.output / "metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
    results = []
    try:
        with LLM(MODEL, devices=meta["npu_devices"]) as llm:
            params = SamplingParams(
                max_tokens=1024,
                temperature=0,
                structured_outputs=StructuredOutputsParams(json=SCHEMA),
            )
            for cond in CONDITIONS:
                for i, r in enumerate(rows, 1):
                    start = time.monotonic()
                    item = {
                        "patient_id": r["patient_id"],
                        "fold": r["fold"],
                        "y_true": r["label"],
                        "condition": cond,
                        "p_positive": None,
                        "status": "error",
                        "error_type": None,
                        "seconds": None,
                    }
                    try:
                        prompt = llm.tokenizer.apply_chat_template(
                            messages(r, cond),
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        output = llm.generate([prompt], params)[0].outputs[0]
                        item.update(p_positive=parse_result(output), status="ok")
                    except (ValueError, TypeError, RuntimeError, KeyError, IndexError) as e:
                        item["error_type"] = type(e).__name__
                    item["seconds"] = round(time.monotonic() - start, 3)
                    results.append(item)
                    write_rows(args.output / "predictions.csv", results)
                    print(cond, f"{i}/{len(rows)}", item["status"], flush=True)
    except Exception as e:
        meta["status"] = "FAILED"
        meta["error_type"] = type(e).__name__
        atomic(args.output / "metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
        raise
    paired = set.intersection(
        *[
            {r["patient_id"] for r in results if r["condition"] == c and r["status"] == "ok"}
            for c in CONDITIONS
        ]
    )
    report = {}
    for cond in CONDITIONS:
        subset = [r for r in results if r["condition"] == cond]
        report[cond] = {
            "all_successful": metric(subset, len(rows), "all successful"),
            "paired_successful": metric(
                [r for r in subset if r["patient_id"] in paired], len(paired), "paired only"
            ),
        }
    meta["status"] = (
        "COMPLETE"
        if len(results) == len(rows) * len(CONDITIONS) and all(r["status"] == "ok" for r in results)
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
    print("완료", meta["status"], "공통", len(paired), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
