"""Export patient-level scores from the saved partial fine-tuning checkpoint."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from train_adapter import AdapterModel, load_patients
from train_finetune import metrics, predict

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--checkpoint", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
result = json.loads((args.output / "metrics.json").read_text())
records = json.loads((args.data / "manifest.json").read_text())["records"]
ids = {}
for row in records:
    patient = re.sub(r"-\d+$", "", row["record"])
    if patient in ids and ids[patient] != row["split"]:
        raise ValueError("Patient split mismatch")
    ids[patient] = row["split"]
patients = load_patients(args.data)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdapterModel(args.checkpoint).to(device)
state = torch.load(args.output / "best_finetuned.pt", map_location=device, weights_only=True)
model.adapter.load_state_dict(state["adapter"])
model.head.load_state_dict(state["head"])
for idx, weights in state["layers"].items():
    model.encoder.encoder.layers[int(idx)].load_state_dict(weights)
threshold = result["validation_selected_threshold"]
outputs = []
for split, rows in patients.items():
    names = [name for name, value in ids.items() if value == split]
    if len(names) != len(rows):
        raise ValueError("Patient alignment mismatch")
    labels, scores = predict(model, rows, device)
    actual = metrics(labels, scores, threshold)
    expected = result["splits"][split]["validation_f1_threshold"]
    for key in ("auroc", "average_precision", "precision", "recall", "specificity"):
        if actual[key] is None or expected[key] is None:
            if actual[key] != expected[key]:
                raise ValueError(f"{split} {key} mismatch")
        elif not np.isclose(actual[key], expected[key], atol=1e-6):
            raise ValueError(f"{split} {key} mismatch")
    if actual["confusion"] != expected["confusion"]:
        raise ValueError(f"{split} confusion mismatch")
    for name, label, score in zip(names, labels, scores, strict=True):
        outputs.append(
            {
                "patient_id": name,
                "split": split,
                "scd_label": int(label),
                "model_score": float(score),
                "prediction_at_0_5": int(score >= 0.5),
                "validation_selected_threshold": threshold,
                "prediction_at_validation_threshold": int(score >= threshold),
                "y_true": int(label),
                "y_score": float(score),
                "y_pred": int(score >= threshold),
            }
        )
    print(f"{split}: {len(rows)} patients, verified metrics", flush=True)
path = args.output / "patient_predictions.csv"
frame = pd.DataFrame(outputs)
frame.to_csv(path, index=False)
path.chmod(0o600)
validation_path = args.output / "validation.csv"
frame.loc[frame["split"] == "validation"].to_csv(validation_path, index=False)
validation_path.chmod(0o600)
print(f"Saved {len(outputs)} predictions to {path}; validation: {validation_path}", flush=True)
