"""Train a 3->12 adapter and SCD head with a frozen ECG-FM encoder."""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from fairseq_signals.models import build_model_from_checkpoint
from scipy.signal import resample_poly
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)


def load_patients(source):
    records = json.loads((source / "manifest.json").read_text())["records"]
    patients = {}
    for row in records:
        name = re.sub(r"-\d+$", "", row["record"])
        values = np.load(source / (row["record"] + ".npy"), allow_pickle=False).astype(np.float32)
        values = resample_poly(values, 5, 2, axis=-1).astype(np.float32)
        std = values.std(axis=-1, keepdims=True)
        values = (values - values.mean(axis=-1, keepdims=True)) / np.where(std < 1e-8, 1, std)
        values[:, np.logical_not(row["lead_presence"]), :] = 0
        if values.shape[1:] != (3, 2500) or not np.isfinite(values).all():
            raise ValueError(f"Invalid waveform shape or values in {row['record']}")
        entry = patients.setdefault(
            name, {"split": row["split"], "label": row["label"], "records": []}
        )
        if (entry["split"], entry["label"]) != (row["split"], row["label"]):
            raise ValueError("Patient split or label mismatch")
        entry["records"].append(values)
    groups = {
        s: [v for v in patients.values() if v["split"] == s]
        for s in ("train", "validation", "test")
    }
    for split, rows in groups.items():
        if {int(r["label"]) for r in rows} != {0, 1}:
            raise ValueError(f"{split}: both classes are required")
    return groups


class AdapterModel(torch.nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        base = build_model_from_checkpoint(checkpoint_path=str(checkpoint))
        self.encoder = base.encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        # The upstream feature extractor otherwise uses no_grad, blocking gradients to the adapter.
        self.encoder.feature_grad_mult = 1.0
        self.adapter = torch.nn.Conv1d(3, 12, kernel_size=1, bias=False)
        with torch.no_grad():
            self.adapter.weight.zero_()
            for i in range(3):
                self.adapter.weight[i, i, 0] = 1.0
        self.head = torch.nn.Linear(base.proj.in_features, 1)
        self.encoder.eval()

    def forward(self, x):
        features = self.encoder.extract_features(source=self.adapter(x), padding_mask=None)["x"]
        pooled = features.mean(dim=1)
        pooled = torch.nn.functional.layer_norm(pooled, (pooled.shape[-1],))
        return self.head(pooled).squeeze(-1)


def evaluate(model, rows, device):
    labels, scores = [], []
    model.eval()
    with torch.no_grad():
        for patient in rows:
            per_record = []
            for record in patient["records"]:
                logits = []
                for batch in np.array_split(record, max(1, int(np.ceil(len(record) / 4)))):
                    logits.extend(model(torch.from_numpy(batch).to(device)).cpu().tolist())
                per_record.append(float(np.mean(logits)))
            labels.append(patient["label"])
            scores.append(float(torch.sigmoid(torch.tensor(np.mean(per_record))).item()))
    y, p = np.array(labels), np.array(scores)
    prediction = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "n": len(y),
        "positive": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "sensitivity": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This experiment requires local CUDA for a practical runtime")
    groups = load_patients(args.data)
    model = AdapterModel(args.checkpoint).to(device)
    model.encoder.eval()
    optimizer = torch.optim.AdamW(
        list(model.adapter.parameters()) + list(model.head.parameters()),
        lr=0.0003,
        weight_decay=0.001,
    )
    positives = sum(r["label"] for r in groups["train"])
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(groups["train"]) - positives) / positives, device=device)
    )
    initial = model.adapter.weight.detach().clone()
    best = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    print(
        "device",
        torch.cuda.get_device_name(),
        "patients",
        {k: len(v) for k, v in groups.items()},
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.adapter.train()
        model.head.train()
        model.encoder.eval()
        shuffled = random.sample(groups["train"], len(groups["train"]))
        losses = []
        for start in range(0, len(shuffled), args.batch_size):
            batch = shuffled[start : start + args.batch_size]
            x = []
            for row in batch:
                rec = random.choice(row["records"])
                x.append(rec[random.randrange(len(rec))])
            x = torch.from_numpy(np.stack(x)).to(device)
            y = torch.tensor([r["label"] for r in batch], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            if (
                model.adapter.weight.grad is None
                or not torch.isfinite(model.adapter.weight.grad).all()
            ):
                raise RuntimeError("Adapter gradients are missing or non-finite")
            torch.nn.utils.clip_grad_norm_(
                list(model.adapter.parameters()) + list(model.head.parameters()), 1.0
            )
            optimizer.step()
            losses.append(float(loss.item()))
        metrics = evaluate(model, groups["validation"], device)
        delta = float((model.adapter.weight.detach() - initial).norm().item())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation": metrics,
                "adapter_change_l2": delta,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if metrics["average_precision"] > best:
            best = metrics["average_precision"]
            best_epoch = epoch
            stale = 0
            best_state = {
                "adapter": copy.deepcopy(model.adapter.state_dict()),
                "head": copy.deepcopy(model.head.state_dict()),
            }
            torch.save(best_state, args.output / "best_adapter.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.adapter.load_state_dict(best_state["adapter"])
    model.head.load_state_dict(best_state["head"])
    result = {
        "method": "Trainable 3-to-12 linear adapter and SCD head; frozen finetuned ECG-FM encoder",
        "device": torch.cuda.get_device_name(),
        "seed": 42,
        "best_epoch": best_epoch,
        "selection": "validation average precision",
        "threshold": 0.5,
        "history": history,
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        result["splits"][split] = evaluate(model, groups[split], device)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["splits"], indent=2), flush=True)


if __name__ == "__main__":
    main()
