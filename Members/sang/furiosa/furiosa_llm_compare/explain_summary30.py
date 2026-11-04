"""Generate research-only explanations for Qwen3-32B patient-level scores."""

import argparse
import csv
import json
from pathlib import Path

from furiosa_llm import LLM, SamplingParams
from furiosa_llm.sampling_params import StructuredOutputsParams

SCHEMA = {
    "type": "object",
    "properties": {"explanation": {"type": "string"}},
    "required": ["explanation"],
    "additionalProperties": False,
}
SYSTEM = (
    "You explain a retrospective research output. Write a short Korean description using only supplied facts. "
    "The score is uncalibrated and is not a clinical probability. Window candidates are descriptive RR/heart rate "
    "extrema, not diagnosed events and not proven causes of the score. Do not infer QT, arrhythmia, or treatment. "
    "Return one concise Korean sentence in JSON with explanation only."
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = json.loads((a.root / "input_summary30.json").read_text())
    candidates = json.loads((a.root / "review_candidates.json").read_text())
    with (a.root / "run_qwen32/predictions.csv").open(newline="") as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["condition"] == "holter_clinical" and r["status"] == "ok"
        ]
    if len(rows) != 356 or len({r["patient_id"] for r in rows}) != 356:
        raise ValueError("incomplete scores")
    by = {r["patient_id"]: r for r in data["patients"]}
    params = SamplingParams(
        max_tokens=512, temperature=0, structured_outputs=StructuredOutputsParams(json=SCHEMA)
    )
    a.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with LLM("furiosa-ai/Qwen3-32B-FP8", devices="npu:4,npu:5,npu:6,npu:7") as llm:
        for i, row in enumerate(rows, 1):
            pid = row["patient_id"]
            rec = by[pid]
            existing = a.output / f"{pid}.json"
            if (
                existing.exists()
                and json.loads(existing.read_text()).get("status") == "generated_unverified"
            ):
                continue
            facts = {
                "score": float(row["p_positive"]),
                "score_model": "qwen32",
                "score_condition": "holter_clinical",
                "holter_summary": rec["holter"],
                "review_candidates": candidates[pid]["segments"],
            }
            messages = [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        facts, ensure_ascii=True, allow_nan=False, separators=(",", ":")
                    ),
                },
            ]
            try:
                prompt = llm.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                result = llm.generate([prompt], params)[0].outputs[0]
                if result.finish_reason != "stop":
                    raise ValueError("incomplete output: " + str(result.finish_reason))
                obj = json.loads(result.text)
                if (
                    result.finish_reason != "stop"
                    or set(obj) != {"explanation"}
                    or not isinstance(obj["explanation"], str)
                    or not obj["explanation"].strip()
                ):
                    raise ValueError("invalid output")
                output = {
                    "status": "generated_unverified",
                    "text": obj["explanation"],
                    "warning": "LLM output may be incorrect; review required; not medical advice.",
                }
            except (ValueError, TypeError, KeyError, RuntimeError, IndexError) as exc:
                output = {"status": "error", "error_type": type(exc).__name__, "text": None}
                print("explanation_error", type(exc).__name__, flush=True)
            (a.output / f"{pid}.json").write_text(
                json.dumps(output, ensure_ascii=False), encoding="utf-8"
            )
            if i % 25 == 0:
                print("explanations", i, "/ 356", flush=True)
    print("EXPLANATIONS_DONE", flush=True)


if __name__ == "__main__":
    main()
