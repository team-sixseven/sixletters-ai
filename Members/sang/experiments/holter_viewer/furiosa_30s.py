"""Experimental Furiosa inference for one selected 30-second RR summary."""

import argparse
import csv
import io
import json
from pathlib import Path

from furiosa_llm import LLM, SamplingParams
from furiosa_llm.sampling_params import StructuredOutputsParams
from llm_contract import (
    CONTRACT_VERSION,
    SCHEMA,
    build_messages,
    parse_model_output,
    sanitize_features,
)
from run_server import atomic

MODELS = {
    "gpt_oss": ("furiosa-ai/gpt-oss-20b", "npu:4", "gpt_oss"),
    "qwen32": ("furiosa-ai/Qwen3-32B-FP8", "npu:4,npu:5,npu:6,npu:7", "qwen_instruct"),
    "qwen_moe": (
        "furiosa-ai/Qwen3-30B-A3B-Instruct-2507-FP8",
        "npu:4,npu:5,npu:6,npu:7",
        "qwen_instruct",
    ),
    "exaone": ("furiosa-ai/EXAONE-4.0-32B-FP8", "npu:4,npu:5,npu:6,npu:7", "exaone"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    features = sanitize_features(json.loads(args.features.read_text()))
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    meta = {
        "status": "RUNNING",
        "scope": "one 30-second RR summary",
        "contract_version": CONTRACT_VERSION,
        "calibrated": False,
        "limitations": "not SCD risk; no waveform image, no QT or event labels; 1 segment only",
    }
    atomic(args.output / "metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
    messages = build_messages(features)
    rows = []
    for name, (model, device, template) in MODELS.items():
        print("MODEL_START", name, flush=True)
        stage = "load"
        try:
            with LLM(model, devices=device) as llm:
                stage = "template"
                opt = {"tokenize": False, "add_generation_prompt": True}
                if template == "gpt_oss":
                    opt["reasoning_effort"] = "low"
                else:
                    opt["enable_thinking"] = False
                prompt = llm.tokenizer.apply_chat_template(messages, **opt)
                stage = "schema"
                params = SamplingParams(
                    max_tokens=1024,
                    temperature=0,
                    structured_outputs=StructuredOutputsParams(json=SCHEMA),
                )
                stage = "generate"
                result = llm.generate([prompt], params)[0].outputs[0]
                stage = "validate"
                row = {
                    "model": name,
                    "score": parse_model_output(result.text, result.finish_reason),
                    "status": "ok",
                }
        except (ValueError, TypeError, RuntimeError, KeyError, IndexError) as exc:
            row = {"model": name, "score": "", "status": type(exc).__name__}
            safe = {
                "model response did not finish",
                "model response must be JSON",
                "model output schema mismatch",
                "invalid review_score",
            }
            print(
                "MODEL_ERROR",
                name,
                stage,
                type(exc).__name__,
                str(exc) if str(exc) in safe else "",
                flush=True,
            )
        rows.append(row)
        s = io.StringIO(newline="")
        w = csv.DictWriter(s, fieldnames=["model", "score", "status"])
        w.writeheader()
        w.writerows(rows)
        atomic(args.output / "model_scores.csv", s.getvalue())
        print("MODEL_DONE", name, row["status"], flush=True)
    meta["status"] = "COMPLETE" if all(r["status"] == "ok" for r in rows) else "PARTIAL"
    atomic(args.output / "metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
    print("RUN_DONE", meta["status"], flush=True)


if __name__ == "__main__":
    main()
