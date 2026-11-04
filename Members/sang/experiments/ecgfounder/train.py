"""Frozen official ECGFounder with trainable XYZ input and binary output adapters."""

import argparse
import copy
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch, medfilt, resample_poly
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from torch import nn

REPO = Path(__file__).resolve().parents[4]
MODEL_REPO = REPO / ".poc-cache/ECGFounder"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MODEL_REPO))
from Members.sang.model.config import load_basic_config  # noqa: E402
from Members.sang.model.DL.dataloader.raw_waveform import load_raw_cohort  # noqa: E402
from net1d import Net1D  # noqa: E402

CHECKPOINT_SHA256 = "ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997"


def make_backbone(checkpoint):
    h = hashlib.sha256()
    with Path(checkpoint).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Official checkpoint SHA-256 mismatch")
    net = Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=150,
    )
    safe = [
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        type(np.dtype("float64")),
        type(np.dtype("float32")),
    ]
    with torch.serialization.safe_globals(safe):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["state_dict"], strict=True)
    return net


class AdaptedFounder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.input_adapter = nn.Conv1d(3, 12, 1, bias=False)
        with torch.no_grad():
            self.input_adapter.weight.zero_()
            for i in range(3):
                self.input_adapter.weight[i, i, 0] = 1.0
        self.backbone = backbone.requires_grad_(False).eval()
        self.output_adapter = nn.Linear(150, 1)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (3, 5000):
            raise ValueError("Expected (N,3,5000)")
        return self.output_adapter(self.backbone(self.input_adapter(x))).squeeze(-1)


def preprocess(windows, presence):
    # Official util.filter_bandpass: notch 50 Hz, bandpass 0.67-40 Hz, 0.4s median baseline.
    x = resample_poly(windows.astype(np.float32), 5, 2, axis=-1)
    b, a = iirnotch(50, 30, 500)
    x = filtfilt(b, a, x, axis=-1)
    b, a = butter(4, [0.67, 40], btype="bandpass", fs=500)
    x = filtfilt(b, a, x, axis=-1)
    baseline = medfilt(x, kernel_size=(1, 1, 201))
    x = x - baseline
    # The published dataset computes one mean/std across all leads and times.
    x = (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-8)
    x[:, np.logical_not(presence), :] = 0
    x = x.astype(np.float32)
    if x.shape[1:] != (3, 5000) or not np.isfinite(x).all():
        raise ValueError("Invalid preprocessed signal")
    return x


def load_groups(splits=("train", "validation")):
    config_path = REPO / "Members/sang/model/basic.yaml"
    cfg = load_basic_config(config_path)
    raw = cfg["data"]["raw_ecg"]
    if raw["sample_rate_hz"] != 200 or raw["segment_seconds"] != 30:
        raise ValueError("Expected 200 Hz, 30-second quality windows")
    frames, indices, _ = load_raw_cohort(cfg, config_path)
    groups = {}
    for split in splits:
        for idx, row in enumerate(frames[split].itertuples(), 1):
            pid = re.sub(r"-\d+$", "", row.patient_id)
            usable = np.flatnonzero(indices[row.patient_id])
            chosen = usable[np.linspace(0, len(usable) - 1, min(8, len(usable)), dtype=int)]
            signal = np.load(row.ecg_path, mmap_mode="r", allow_pickle=False)
            windows = np.stack(
                [
                    np.asarray(signal[int(i) * 6000 : int(i) * 6000 + 2000], dtype=np.float32).T
                    for i in chosen
                ]
            )
            windows = preprocess(windows, row.lead_presence)
            item = groups.setdefault(
                pid, {"pid": pid, "split": split, "label": int(row.y_true), "records": []}
            )
            if (item["split"], item["label"]) != (split, int(row.y_true)):
                raise ValueError("Inconsistent patient split/label")
            item["records"].append(
                {"id": row.patient_id, "windows": windows, "has_z": bool(row.lead_presence[2])}
            )
            if idx % 50 == 0:
                print("preprocessed", split, idx, flush=True)
    return {s: [v for v in groups.values() if v["split"] == s] for s in splits}


def evaluate(model, rows, device, batch_size):
    model.eval()
    result = []
    with torch.no_grad():
        for r in rows:
            record_logits = []
            for record in r["records"]:
                scores = []
                for start in range(0, len(record["windows"]), batch_size):
                    logits = model(
                        torch.from_numpy(record["windows"][start : start + batch_size]).to(device)
                    )
                    if not torch.isfinite(logits).all():
                        raise RuntimeError("Non-finite logits")
                    scores.extend(logits.cpu().tolist())
                record_logits.append(float(np.mean(scores)))
            probability = torch.sigmoid(torch.tensor(np.mean(record_logits))).item()
            result.append(
                {
                    "patient_id": r["pid"],
                    "split": r["split"],
                    "scd_4year_label": r["label"],
                    "model_score": probability,
                    "prediction_0_5": int(probability >= 0.5),
                    "record_count": len(r["records"]),
                    "segment_count": sum(len(t["windows"]) for t in r["records"]),
                    "all_records_have_z": all(t["has_z"] for t in r["records"]),
                }
            )
    y = np.array([r["scd_4year_label"] for r in result])
    p = np.array([r["model_score"] for r in result])
    pred = p >= 0.5
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    metrics = {
        "n": len(y),
        "positive": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(np.mean(pred == y)),
        "precision": tp / (tp + fp) if tp + fp else None,
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "threshold": 0.5,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    return result, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if min(a.epochs, a.patience, a.batch_size) < 1 or a.output.exists():
        p.error("positive settings and new output directory required")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    model = AdaptedFounder(make_backbone(a.checkpoint)).to(a.device)
    groups = load_groups()
    for split, rows in groups.items():
        if {r["label"] for r in rows} != {0, 1}:
            raise ValueError(f"{split}: both classes required")
    print(
        "patients/records",
        {s: (len(v), sum(len(r["records"]) for r in v)) for s, v in groups.items()},
        flush=True,
    )
    before = {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()}
    initial_adapter = model.input_adapter.weight.detach().clone()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0003, weight_decay=0.001)
    pos = sum(r["label"] for r in groups["train"])
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(groups["train"]) - pos) / pos, device=a.device)
    )
    history = []
    best = -1.0
    stale = 0
    selected = None
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = random.sample(groups["train"], len(groups["train"]))
        losses = []
        for start in range(0, len(order), a.batch_size):
            batch = order[start : start + a.batch_size]
            values = []
            for patient in batch:
                rec = random.choice(patient["records"])
                values.append(rec["windows"][random.randrange(len(rec["windows"]))])
            x = torch.from_numpy(np.stack(values)).to(a.device)
            y = torch.tensor([r["label"] for r in batch], device=a.device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            if (
                model.input_adapter.weight.grad is None
                or not torch.isfinite(model.input_adapter.weight.grad).all()
            ):
                raise RuntimeError("Invalid input gradient")
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(loss.item())
        _, m = evaluate(model, groups["validation"], a.device, a.batch_size)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": m})
        print(json.dumps(history[-1]), flush=True)
        if m["average_precision"] > best:
            best = m["average_precision"]
            stale = 0
            selected = {
                "epoch": epoch,
                "input_adapter": copy.deepcopy(model.input_adapter.state_dict()),
                "output_adapter": copy.deepcopy(model.output_adapter.state_dict()),
            }
        else:
            stale += 1
            if stale >= a.patience:
                break
    if any(not torch.equal(v, model.backbone.state_dict()[k].cpu()) for k, v in before.items()):
        raise RuntimeError("Frozen backbone changed")
    changed = bool(not torch.equal(initial_adapter, model.input_adapter.weight.detach()))
    if not changed:
        raise RuntimeError("Input adapter did not change")
    model.input_adapter.load_state_dict(selected["input_adapter"])
    model.output_adapter.load_state_dict(selected["output_adapter"])
    rows, m = evaluate(model, groups["validation"], a.device, a.batch_size)
    a.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    with (a.output / "validation.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (a.output / "validation.csv").chmod(0o600)
    selected.update(
        {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "source_commit": "04edac702b61c91face519774ddcc0cd712fef23",
        }
    )
    torch.save(selected, a.output / "best_adapter.pt")
    summary = {
        "model": "ECGFounder 12-lead 150-class frozen, 3->12 and 150->1 adapters",
        "best_epoch": selected["epoch"],
        "epochs_run": len(history),
        "max_epochs": a.epochs,
        "patience": a.patience,
        "seed": a.seed,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_commit": selected["source_commit"],
        "selection": "validation average precision",
        "counts": {
            s: {
                "patients": len(v),
                "records": sum(len(r["records"]) for r in v),
                "positive": sum(r["label"] for r in v),
            }
            for s, v in groups.items()
        },
        "frozen_backbone_unchanged": True,
        "input_adapter_changed": changed,
        "validation": m,
        "history": history,
    }
    (a.output / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("saved", a.output, flush=True)


if __name__ == "__main__":
    main()
