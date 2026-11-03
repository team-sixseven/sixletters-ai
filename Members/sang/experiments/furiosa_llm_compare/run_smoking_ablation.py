"""Ablate one source field with ambiguous semantics without modifying the input."""

import argparse
import json
from pathlib import Path

import run_diagnostics as experiment
from run_server import SYSTEM

experiment.CONDITIONS = (
    "clinical_no_smoking",
    "combined_no_smoking",
    "clinical_only_corrected_prompt",
)


def messages(record, condition):
    clinical = (
        dict(record["clinical"])
        if condition == "clinical_only_corrected_prompt"
        else {k: v for k, v in record["clinical"].items() if k != "cigarettes /year"}
    )
    data = {
        "ecg_summary": record["holter"] if condition == "combined_no_smoking" else None,
        "clinical": clinical,
    }
    system = SYSTEM
    if condition == "clinical_only_corrected_prompt":
        system = system.replace(
            "Given experimental Holter-derived RR and HRV estimates, ",
            "Given the available clinical measurements and no Holter measurements, ",
        ).replace(
            "Normal beat classification and these features are not validated; null means unavailable. ",
            "Null means unavailable. ",
        )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(data, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
        },
    ]


experiment.messages = messages
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    experiment.main(p.parse_args())
