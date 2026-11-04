"""Provisional model-side adapter; no claim of clinical or web compatibility."""

import argparse
import csv
import json
import re
import secrets
import shlex
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from llm_contract import CONTRACT_VERSION, sanitize_features
from server import RATE, REPO, VIEW_SECONDS, HolterStore, segment_features

REMOTE_ROOT = "/root/.cache/std3_holter/llm_compare"
MODEL_SCRIPT = Path(__file__).with_name("furiosa_30s.py")
CONTRACT_SCRIPT = Path(__file__).with_name("llm_contract.py")
MODEL_NAMES = {"gpt_oss", "qwen32", "qwen_moe", "exaone"}


def event(step, state, **details):
    print(json.dumps({"step": step, "state": state, **details}, ensure_ascii=False), flush=True)


def private_json(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    path.chmod(0o600)


def run_remote(features_path, destination, key_path):
    run = "adapter30_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + secrets.token_hex(3)
    stage = REMOTE_ROOT + "/" + run + "_stage"
    result = REMOTE_ROOT + "/" + run
    host = "std3@eval.nxt.furiosa.ai"
    options = ["-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=10"]
    with tempfile.TemporaryDirectory(prefix="holter-adapter-key-") as directory:
        key = Path(directory) / "key"
        key.write_bytes(key_path.read_bytes())
        key.chmod(0o600)

        def ssh(command, timeout=30):
            return subprocess.run(
                ["ssh", "-i", str(key), "-p", "36040", *options, host, command],
                check=True,
                timeout=timeout,
                capture_output=True,
                text=True,
            )

        def scp(source, target):
            subprocess.run(
                ["scp", "-q", "-i", str(key), "-P", "36040", *options, str(source), str(target)],
                check=True,
                timeout=30,
                capture_output=True,
            )

        ssh("mkdir -m 700 " + shlex.quote(stage))
        scp(MODEL_SCRIPT, host + ":" + stage + "/run.py")
        scp(CONTRACT_SCRIPT, host + ":" + stage + "/llm_contract.py")
        scp(features_path, host + ":" + stage + "/features.json")
        command = " ".join(
            shlex.quote(x)
            for x in [
                "env",
                "HF_HOME=/root/.cache/std3_holter/huggingface",
                "PYTHONPATH=" + REMOTE_ROOT,
                "/root/std3/furiosa_holter_venv/bin/python",
                "-u",
                stage + "/run.py",
                "--features",
                stage + "/features.json",
                "--output",
                result,
            ]
        )
        # A real model run can take minutes; log only model identifiers and statuses locally.
        completed = ssh(command, timeout=900)
        scp(host + ":" + result + "/model_scores.csv", destination / "model_scores.csv")
        scp(host + ":" + result + "/metadata.json", destination / "model_metadata.json")
        for name in ("model_scores.csv", "model_metadata.json"):
            (destination / name).chmod(0o600)
        return result, completed.stdout


def assemble(window, features, model_rows, remote_status):
    if len(window["samples"]) != RATE * VIEW_SECONDS or VIEW_SECONDS != 30:
        raise ValueError("Expected exactly 30 seconds of 200 Hz data")
    if any(flag is not False for flag in window["quality_excluded"]):
        raise ValueError("Selected segment is excluded or has unknown quality")
    if (
        remote_status != "COMPLETE"
        or len(model_rows) != 4
        or {r["model"] for r in model_rows} != MODEL_NAMES
    ):
        raise ValueError("All four model outputs are required")
    scores = []
    for row in model_rows:
        value = float(row["score"])
        if row["status"] != "ok" or not 0 <= value <= 1:
            raise ValueError("Invalid model result")
        scores.append({"model": row["model"], "review_priority_score": value})
    return {
        "schema": "holter-demo-research-provisional-v1",
        "analysis_scope": {
            "kind": "selected_30_second_window",
            "start_seconds": window["start_seconds"],
            "duration_seconds": 30,
            "lead": window["lead"],
            "sample_rate_hz": RATE,
        },
        "record": {"absolute_start_time": None, "quality": "passed_source_30s_index"},
        "features": sanitize_features(features),
        "model": {
            "kind": "experimental_rr_summary_llm",
            "scores": scores,
            "contract_version": CONTRACT_VERSION,
            "is_calibrated": False,
            "is_scd_probability": False,
        },
        # Missing capability is explicit; empty list would incorrectly imply a negative finding.
        "prob": None,
        "factors": None,
        "segments": None,
        "qt_prolongation": None,
        "unavailable_reasons": {
            "prob": "No validated SCD probability model",
            "factors": "No verified SHAP or attribution",
            "segments": "No validated 24-hour dangerous-segment detector",
            "qt_prolongation": "QT candidate method not validated",
        },
        "waveform": {"sample_rate_hz": RATE, "samples": window["samples"]},
        "intended_use": "research demonstration only; not clinical care",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patient", required=True)
    p.add_argument("--start-seconds", type=float, default=0)
    p.add_argument("--lead", choices=["X", "Y", "Z"], default="X")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--key", type=Path, default=Path("/mnt/c/Users/Admin/.ssh/id_ed25519_std3"))
    args = p.parse_args()
    if not re.fullmatch(r"P\d{4}", args.patient):
        raise ValueError("Unsupported patient ID")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    event("record", "started")
    store = HolterStore(REPO / "Members/sang/model/basic.yaml")
    record = store.record(args.patient)
    window = store.window(args.patient, args.start_seconds, args.lead)
    event("record", "completed", duration_seconds=record["duration_seconds"], sample_rate_hz=RATE)
    event("quality", "started")
    if any(x is not False for x in window["quality_excluded"]):
        event("quality", "failed", reason="excluded_or_unknown")
        raise ValueError("Selected segment quality is excluded or unknown")
    event("quality", "completed", windows=window["quality_excluded"])
    event("features", "started")
    features = segment_features(window)
    private_json(args.output / "features.json", features)
    event("features", "completed", rr_count=features["rr_valid_count"])
    event("model", "started", model_count=4, scope="selected_30s_rr_summary")
    remote, _ = run_remote(args.output / "features.json", args.output, args.key)
    with (args.output / "model_scores.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    meta = json.loads((args.output / "model_metadata.json").read_text())
    event(
        "model",
        "completed",
        successful=sum(r.get("status") == "ok" for r in rows),
        status=meta["status"],
    )
    event("output", "started")
    output = assemble(window, features, rows, meta["status"])
    private_json(args.output / "provisional_output.json", output)
    private_json(
        args.output / "provenance.json",
        {
            "local_patient_id": args.patient,
            "remote_run": remote,
            "limitations": "not a trained or validated SCD risk/segment model; no NPZ upload support",
        },
    )
    event("output", "completed", scd_probability_available=False, segments_available=False)
    print("RESULT_DIRECTORY", args.output, flush=True)


if __name__ == "__main__":
    main()
