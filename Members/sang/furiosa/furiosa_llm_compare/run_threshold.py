"""Evaluate threshold selection with separate calibration and outer folds."""

import argparse
import csv
import hashlib
import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from furiosa_llm import LLM, SamplingParams
from furiosa_llm.sampling_params import StructuredOutputsParams
from run_server import MODEL, SCHEMA, atomic, build_messages, parse_result
from sklearn.metrics import confusion_matrix


def write_rows(path, rows):
    fields = [
        "patient_id",
        "fold",
        "outer_fold",
        "role",
        "y_true",
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
    data = p["patients"]
    assignment = {}
    for r in data:
        if r["patient_id"] in assignment and assignment[r["patient_id"]] != r["fold"]:
            raise ValueError("inconsistent fold")
        assignment[r["patient_id"]] = r["fold"]
    if len(data) != 712 or len(assignment) != 356:
        raise ValueError("cohort mismatch")
    for r in data:
        outer = r["outer_fold"]
        cal = outer % 5 + 1
        expect = outer if r["role"] == "evaluation" else cal
        if (
            r["role"] not in ("evaluation", "calibration")
            or r["fold"] != expect
            or len(r["example_patient_ids"]) != 4
        ):
            raise ValueError("role mismatch")
        if any(assignment.get(pid) in (None, outer, cal) for pid in r["example_patient_ids"]):
            raise ValueError("case leakage")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    meta = {
        "status": "RUNNING",
        "model": MODEL,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "created_utc": datetime.now(UTC).isoformat(),
        "temperature": 0,
        "thinking": False,
        "npu_devices": "npu:4,npu:5,npu:6,npu:7",
        "input_metadata": p["metadata"],
        "caveat": "8 positives per calibration fold; research only",
    }
    atomic(args.output / "metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
    rows = []
    try:
        with LLM(MODEL, devices=meta["npu_devices"]) as llm:
            params = SamplingParams(
                max_tokens=1024,
                temperature=0,
                structured_outputs=StructuredOutputsParams(json=SCHEMA),
            )
            for i, r in enumerate(data, 1):
                t = time.monotonic()
                row = {
                    "patient_id": r["patient_id"],
                    "fold": r["fold"],
                    "outer_fold": r["outer_fold"],
                    "role": r["role"],
                    "y_true": r["label"],
                    "p_positive": None,
                    "status": "error",
                    "error_type": None,
                    "seconds": None,
                }
                try:
                    prompt = llm.tokenizer.apply_chat_template(
                        build_messages(r, "holter_clinical_cases"),
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    row.update(
                        p_positive=parse_result(llm.generate([prompt], params)[0].outputs[0]),
                        status="ok",
                    )
                except (ValueError, TypeError, RuntimeError, KeyError, IndexError) as e:
                    row["error_type"] = type(e).__name__
                row["seconds"] = round(time.monotonic() - t, 3)
                rows.append(row)
                write_rows(args.output / "predictions.csv", rows)
                print("threshold", f"{i}/{len(data)}", row["status"], flush=True)
    except Exception as e:
        meta["status"] = "FAILED"
        meta["error_type"] = type(e).__name__
        atomic(args.output / "metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
        raise
    if any(x["status"] != "ok" for x in rows):
        meta["status"] = "PARTIAL"
        atomic(args.output / "metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
        print("부분 실패: 임계값 평가 중지", flush=True)
        return
    df = pd.DataFrame(rows)
    report = {"folds": []}
    all_true = []
    all_pred = []
    for fold in range(1, 6):
        cal = df[(df.outer_fold == fold) & (df.role == "calibration")]
        ev = df[(df.outer_fold == fold) & (df.role == "evaluation")]
        positives = cal[cal.y_true == 1].p_positive.to_numpy()
        if len(positives) != 8:
            raise ValueError("calibration positive count mismatch")
        # Highest threshold retaining >=80% sensitivity in the calibration fold.
        thresholds = sorted(set(cal.p_positive), reverse=True)
        thr = next(float(t) for t in thresholds if (positives >= t).mean() >= 0.8)
        y = ev.y_true.to_numpy()
        pred = (ev.p_positive.to_numpy() >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
        report["folds"].append(
            {
                "fold": fold,
                "threshold": thr,
                "calibration_positive_count": len(positives),
                "calibration_sensitivity": float((positives >= thr).mean()),
                "n": len(ev),
                "confusion_matrix": [[tn, fp], [fn, tp]],
                "sensitivity": tp / (tp + fn) if tp + fn else None,
                "specificity": tn / (tn + fp) if tn + fp else None,
            }
        )
        all_true.extend(y.tolist())
        all_pred.extend(pred.tolist())
    tn, fp, fn, tp = confusion_matrix(all_true, all_pred, labels=[0, 1]).ravel().tolist()
    report["pooled"] = {
        "n": len(all_true),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "target": "calibration fold sensitivity >=0.8; not a clinical threshold",
    }
    meta["status"] = "COMPLETE"
    atomic(
        args.output / "metrics.json",
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
    )
    atomic(args.output / "metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
    print("완료", meta["status"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
