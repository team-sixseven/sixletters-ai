"""Fine tune two encoder blocks preceding two frozen final blocks."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from train_adapter import AdapterModel, load_patients


def predict(model, patients, device):
    y, predictions = [], []
    model.eval()
    with torch.inference_mode():
        for patient in patients:
            record_scores = []
            for record in patient["records"]:
                logits = []
                for x in np.array_split(record, max(1, int(np.ceil(len(record) / 4)))):
                    logits.extend(model(torch.from_numpy(x).to(device)).cpu().tolist())
                record_scores.append(np.mean(logits))
            y.append(int(patient["label"]))
            predictions.append(torch.sigmoid(torch.tensor(np.mean(record_scores))).item())
    return np.asarray(y), np.asarray(predictions)


def metrics(y, p, threshold):
    pred = p >= threshold
    tn, fp, fn, tp = (int(n) for n in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {
        "n": len(y),
        "positive": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def choose_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required")
    groups = load_patients(args.data)
    model = AdapterModel(args.checkpoint).to(device)
    initial = torch.load(args.adapter, map_location=device, weights_only=True)
    model.adapter.load_state_dict(initial["adapter"])
    model.head.load_state_dict(initial["head"])
    layers = model.encoder.encoder.layers
    if len(layers) < 4:
        raise ValueError("Encoder must have at least four transformer layers")
    selected = list(range(len(layers) - 4, len(layers) - 2))
    frozen = list(range(len(layers) - 2, len(layers)))
    for i in selected:
        for param in layers[i].parameters():
            param.requires_grad_(True)
    assert all(not p.requires_grad for i in frozen for p in layers[i].parameters())
    print(
        "device",
        torch.cuda.get_device_name(),
        "trainable layers",
        selected,
        "last frozen layers",
        frozen,
        flush=True,
    )
    trainable = [p for i in selected for p in layers[i].parameters()]
    optim = torch.optim.AdamW(
        [
            {"params": trainable, "lr": 1e-5},
            {
                "params": list(model.adapter.parameters()) + list(model.head.parameters()),
                "lr": 1e-4,
            },
        ],
        weight_decay=0.001,
    )
    positives = sum(v["label"] for v in groups["train"])
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(groups["train"]) - positives) / positives, device=device)
    )
    frozen_initial = [p.detach().cpu().clone() for i in frozen for p in layers[i].parameters()]
    train_initial = [p.detach().cpu().clone() for p in trainable]
    history = []
    best = -1
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.encoder.eval()
        model.adapter.train()
        model.head.train()
        rows = random.sample(groups["train"], len(groups["train"]))
        losses = []
        for start in range(0, len(rows), 4):
            batch = rows[start : start + 4]
            x = []
            for row in batch:
                rec = random.choice(row["records"])
                x.append(rec[random.randrange(len(rec))])
            x = torch.from_numpy(np.stack(x)).to(device)
            y = torch.tensor([r["label"] for r in batch], dtype=torch.float32, device=device)
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
                raise RuntimeError("Trainable encoder gradient missing or nonfinite")
            torch.nn.utils.clip_grad_norm_(
                trainable + list(model.adapter.parameters()) + list(model.head.parameters()), 1.0
            )
            optim.step()
            losses.append(loss.item())
        y, p = predict(model, groups["validation"], device)
        val = metrics(y, p, 0.5)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation": val})
        print(json.dumps(history[-1]), flush=True)
        if val["average_precision"] > best:
            best = val["average_precision"]
            best_epoch = epoch
            stale = 0
            best_state = {
                "adapter": copy.deepcopy(model.adapter.state_dict()),
                "head": copy.deepcopy(model.head.state_dict()),
                "layers": {str(i): copy.deepcopy(layers[i].state_dict()) for i in selected},
            }
            torch.save(best_state, args.output / "best_finetuned.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert all(
        torch.equal(before, p.detach().cpu())
        for before, i in zip(
            frozen_initial, [p for n in frozen for p in layers[n].parameters()], strict=False
        )
        for p in [i]
    )
    delta = float(
        sum(
            (p.detach().cpu() - before).square().sum().item()
            for p, before in zip(trainable, train_initial, strict=False)
        )
        ** 0.5
    )
    if delta <= 0:
        raise RuntimeError("No change in trainable encoder layers")
    model.adapter.load_state_dict(best_state["adapter"])
    model.head.load_state_dict(best_state["head"])
    for i in selected:
        layers[i].load_state_dict(best_state["layers"][str(i)])
    yv, pv = predict(model, groups["validation"], device)
    threshold = choose_threshold(yv, pv)
    result = {
        "max_epochs": args.epochs,
        "patience": args.patience,
        "device": torch.cuda.get_device_name(),
        "seed": 42,
        "trainable_layer_indices": selected,
        "frozen_last_layer_indices": frozen,
        "total_encoder_layers": len(layers),
        "frozen_final_layers_unchanged": True,
        "trainable_layers_change_l2": delta,
        "best_epoch": best_epoch,
        "selection": "validation average precision",
        "threshold_selection": "max validation F1",
        "validation_selected_threshold": threshold,
        "history": history,
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        y, p = predict(model, groups[split], device)
        result["splits"][split] = {
            "fixed_0_5": metrics(y, p, 0.5),
            "validation_f1_threshold": metrics(y, p, threshold),
        }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["splits"], indent=2), flush=True)


if __name__ == "__main__":
    main()
