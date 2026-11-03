"""Retry failed train patients into a separate run, retaining every successful prediction."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import nbformat
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.run / "metadata.json").read_text())
    assert meta["models"] == ["~anthropic/claude-haiku-5-5"] and meta["split"] == "train"
    sources = list(args.run.glob("*/predictions.csv"))
    assert len(sources) == 1
    frame = pd.read_csv(sources[0])
    assert len(frame) == 285 and frame.patient_id.nunique() == 285
    features = json.loads(args.features.read_text())
    assert set(frame.patient_id) == set(features)
    notebook = nbformat.read(
        "Members/sang/notebooks/monorouter_holter_claude_haiku55_train.ipynb", 4
    )
    env = {}
    for index in (2, 3, 9):
        exec(compile(notebook.cells[index].source, f"cell {index}", "exec"), env)
    destination = args.run.parent / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8]
    )
    destination.mkdir(mode=0o700)
    subdir = destination / sources[0].parent.name
    subdir.mkdir(mode=0o700)
    records = frame.where(pd.notna(frame), None).to_dict("records")
    metadata = {
        **meta,
        "status": "RUNNING",
        "retry_of": str(args.run),
        "retry_feature_file": str(args.features),
    }
    metadata_path = destination / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def save():
        temp = subdir / "predictions.csv.tmp"
        pd.DataFrame(records).to_csv(temp, index=False)
        os.chmod(temp, 0o600)
        temp.replace(subdir / "predictions.csv")

    save()
    for index, row in enumerate(records):
        if row["status"] == "ok":
            continue
        for _attempt in range(2):
            try:
                probability, usage = env["predict_one"](
                    features[row["patient_id"]], meta["models"][0]
                )
                row.update(
                    p_positive=probability,
                    **usage,
                    status="ok",
                    error_type=None,
                    http_status=None,
                    source="retry",
                )
                break
            except (RuntimeError, ValueError, KeyError, TypeError, TimeoutError) as exc:
                row.update(
                    status="error",
                    error_type=type(exc).__name__,
                    http_status=getattr(exc, "status", None),
                )
                if getattr(exc, "status", None) in (401, 402, 403, 429):
                    break
        save()
        print("RETRY", index + 1, "STATUS", row["status"], flush=True)
        if row.get("http_status") in (401, 402, 403, 429):
            break
    metadata["status"] = "COMPLETE" if all(row["status"] == "ok" for row in records) else "PARTIAL"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    env.update(
        results={meta["models"][0]: {"records": records, "directory": subdir}},
        result_dir=destination,
        prepared=[None] * len(records),
        SPLIT="train",
        THRESHOLD=meta["threshold"],
    )
    exec(compile(notebook.cells[13].source, "evaluation cell", "exec"), env)
    print("RETRY_RESULT", destination, metadata["status"], flush=True)


if __name__ == "__main__":
    main()
