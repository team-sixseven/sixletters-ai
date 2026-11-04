"""Run existing foundation-model experiments, recording patient-level validation BCE."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[4]
kind = sys.argv.pop(1)
output = Path(sys.argv[sys.argv.index("--output") + 1])
if output.exists():
    raise SystemExit(f"Output already exists: {output}")
paths = {
    "deep": Path("/home/sang/deepecg-3channel/holter_original_sampling.py"),
    "founder": ROOT / "Members/sang/experiments/ecgfounder/train.py",
    "fm": ROOT / "Members/sang/experiments/ecg_fm/train_adapter.py",
}
path = paths[kind]
sys.path.insert(0, str(path.parent))
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("foundation_original", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
weight = None
orig_load = module.load_patients if kind == "fm" else module.load_groups


def load(*a, **kw):
    global weight
    groups = orig_load(*a, **kw)
    labels = [int(r["label"]) for r in groups["train"]]
    weight = (len(labels) - sum(labels)) / sum(labels)
    return groups


def bce(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return float(np.mean(-weight * y * np.log(p) - (1 - y) * np.log1p(-p)))


if kind in ("deep", "founder"):
    module.load_groups = load
    original = module.predict if kind == "deep" else module.evaluate

    def evaluate(*a, **kw):
        rows, metrics = original(*a, **kw)
        metrics["validation_loss"] = bce(
            [r["scd_4year_label"] for r in rows], [r["model_score"] for r in rows]
        )
        return rows, metrics

    if kind == "deep":
        module.predict = evaluate
    else:
        module.evaluate = evaluate
else:
    module.load_patients = load

    def evaluate(model, rows, device):
        labels, scores = [], []
        model.eval()
        with torch.no_grad():
            for patient in rows:
                record_logits = []
                for record in patient["records"]:
                    logits = []
                    for batch in np.array_split(record, max(1, int(np.ceil(len(record) / 4)))):
                        logits.extend(model(torch.from_numpy(batch).to(device)).cpu().tolist())
                    record_logits.append(float(np.mean(logits)))
                labels.append(int(patient["label"]))
                scores.append(float(torch.sigmoid(torch.tensor(np.mean(record_logits))).item()))
        y, p = np.array(labels), np.array(scores)
        tn, fp, fn, tp = confusion_matrix(y, p >= 0.5, labels=[0, 1]).ravel()
        result = {
            "n": len(y),
            "positive": int(y.sum()),
            "auroc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "precision": float(tp / (tp + fp)) if tp + fp else None,
            "sensitivity": float(recall_score(y, p >= 0.5, zero_division=0)),
            "specificity": float(tn / (tn + fp)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "validation_loss": bce(y, p),
        }
        if len(y) == 71 and int(y.sum()) == 8:
            output.mkdir(mode=0o700, parents=True, exist_ok=True)
            with (output / "validation.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["scd_4year_label", "model_score"])
                writer.writerows(zip(y, p, strict=False))
            (output / "validation.csv").chmod(0o600)
        return result

    module.evaluate = evaluate

print(
    json.dumps(
        {
            "model": kind,
            "max_epochs": 100,
            "patience": 10,
            "validation_loss": "patient-level weighted BCE on mean logit; training class weight",
            "output": str(output),
        }
    ),
    flush=True,
)
module.main()
