"""MOMENT foundation-model probe for patient-level SCD from raw Holter ECG.

The pretrained MOMENT encoder is frozen; only the new MIL classifier head is trained.
Because the frozen encoder is deterministic in eval mode, window embeddings are computed
once and cached, and the head is trained on the cache (same result, far faster).
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from ..evaluation.core import binary_metrics

MOMENT_SEQ_LEN = 512
SPLIT_NAMES = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- cohort / windows


def load_cohort(
    split_path: Path, quality_path: Path, wave_directory: Path, lead_metadata_path: Path
) -> pd.DataFrame:
    """One row per NPY file in the split: patient_id, split, y_true, file, has_z, usable window starts."""
    lead_info = pd.read_csv(lead_metadata_path, usecols=["file", "has_z"])
    has_z = dict(
        zip(
            lead_info["file"].str.removesuffix(".npy"),
            lead_info["has_z"].astype(bool),
            strict=False,
        )
    )
    split = pd.read_csv(split_path, dtype={"pid": str})
    split["split"] = split["split"].map(SPLIT_NAMES)
    if split["split"].isna().any():
        raise ValueError("Unknown split value in split file")
    quality = pd.read_csv(quality_path, usecols=["file", "window_idx", "start_idx", "exclude"])
    quality = quality.loc[~quality["exclude"].astype(bool)]
    quality["stem"] = quality["file"].str.removesuffix(".npy")
    starts = quality.sort_values("window_idx").groupby("stem")["start_idx"].apply(list)

    rows, missing = [], []
    for record in split.itertuples(index=False):
        path = wave_directory / f"{record.pid}.npy"
        usable = starts.get(record.pid, [])
        if not path.is_file() or not usable or record.pid not in has_z:
            missing.append(record.pid)
            continue
        rows.append(
            {
                "patient_id": record.pid,
                "split": record.split,
                "y_true": int(record.SCD),
                "file": str(path),
                "has_z": has_z[record.pid],
                "usable_starts": usable,
            }
        )
    if missing:
        print(f"Skipped {len(missing)} patients without NPY or usable windows: {missing[:10]}")
    return pd.DataFrame(rows)


def select_starts(usable_starts: list[int], n_windows: int) -> list[int]:
    """Evenly spaced over the usable 30 s windows, chronological (same idea as the raw DL run)."""
    if len(usable_starts) <= n_windows:
        return list(usable_starts)
    index = np.unique(np.linspace(0, len(usable_starts) - 1, n_windows).round().astype(int))
    return [usable_starts[i] for i in index]


def window_chunks(
    signal: np.ndarray, start: int, window_samples: int, decimation: int, chunks_per_window: int
) -> np.ndarray:
    """30 s window (samples, leads) -> (chunks, leads, 512) float32 for MOMENT."""
    window = np.asarray(signal[start : start + window_samples], dtype=np.float32)
    window = np.nan_to_num(window)[::decimation]  # signal is already 0.5-40 Hz band-limited
    used = chunks_per_window * MOMENT_SEQ_LEN
    if window.shape[0] < used:
        raise ValueError("Window too short for the requested chunks; lower chunks_per_window")
    offset = (window.shape[0] - used) // 2
    window = window[offset : offset + used]
    return window.reshape(chunks_per_window, MOMENT_SEQ_LEN, -1).transpose(0, 2, 1)


# ---------------------------------------------------------------- frozen backbone


def load_moment(model_name: str, device: torch.device) -> nn.Module:
    from momentfm import MOMENTPipeline

    backbone = MOMENTPipeline.from_pretrained(model_name, model_kwargs={"task_name": "embedding"})
    backbone.init()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone.to(device).eval()


@torch.no_grad()
def embed_sequences(
    backbone: nn.Module, sequences: np.ndarray, device, batch_size: int
) -> np.ndarray:
    """(n, 512) univariate sequences -> (n, d_model) mean-pooled MOMENT embeddings."""
    outputs = []
    for begin in range(0, len(sequences), batch_size):
        x = torch.from_numpy(sequences[begin : begin + batch_size]).to(device).unsqueeze(1)
        mask = torch.ones(x.shape[0], MOMENT_SEQ_LEN, device=device)
        outputs.append(backbone(x_enc=x, input_mask=mask).embeddings.float().cpu().numpy())
    return np.concatenate(outputs)


def extract_embeddings(cohort: pd.DataFrame, backbone: nn.Module, settings: dict, device) -> Path:
    """Cache one (windows, leads * d_model) float16 array per patient; existing files are reused."""
    cache_dir = Path(settings["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    window_samples = settings["sample_rate_hz"] * settings["segment_seconds"]
    for record in tqdm(cohort.itertuples(index=False), total=len(cohort), desc="MOMENT embeddings"):
        target = cache_dir / f"{record.patient_id}.npy"
        if target.is_file():
            continue
        signal = np.load(record.file, mmap_mode="r")
        starts = select_starts(record.usable_starts, settings["n_windows"])
        chunks = np.stack(
            [
                window_chunks(
                    signal,
                    start,
                    window_samples,
                    settings["decimation"],
                    settings["chunks_per_window"],
                )
                for start in starts
            ]
        )  # (windows, chunks, leads, 512); leads = X, Y, Z
        if not record.has_z:
            chunks = chunks[:, :, :2]  # Z column is all zeros for XY-only recordings
        n_windows, n_chunks, n_leads, _ = chunks.shape
        embedded = embed_sequences(
            backbone, chunks.reshape(-1, MOMENT_SEQ_LEN), device, settings["embed_batch_size"]
        ).reshape(n_windows, n_chunks, n_leads, -1)
        embedded = embedded.mean(axis=1)  # average chunks inside a 30 s window
        if not record.has_z:
            # missing Z -> mean of X and Y, so a flat zero lead never becomes a feature
            embedded = np.concatenate([embedded, embedded.mean(axis=1, keepdims=True)], axis=1)
        np.save(target, embedded.reshape(n_windows, -1).astype(np.float16))
    return cache_dir


def load_bags(cohort: pd.DataFrame, cache_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad patient bags to the longest one; returns features (P, N, D) and mask (P, N)."""
    arrays = [np.load(Path(cache_dir) / f"{pid}.npy") for pid in cohort["patient_id"]]
    longest = max(array.shape[0] for array in arrays)
    features = np.zeros((len(arrays), longest, arrays[0].shape[1]), dtype=np.float32)
    mask = np.zeros((len(arrays), longest), dtype=bool)
    for i, array in enumerate(arrays):
        features[i, : len(array)] = array
        mask[i, : len(array)] = True
    return torch.from_numpy(features), torch.from_numpy(mask)


# ---------------------------------------------------------------- trainable head


class MILClassifierHead(nn.Module):
    """New layers: window projection -> gated attention MIL -> patient logit."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1)
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.project(features)
        scores = self.attention(hidden).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return self.classifier((weights * hidden).sum(dim=1)).squeeze(-1)


class MomentSCDClassifier(nn.Module):
    """Full model for reference: frozen MOMENT backbone + trainable head."""

    def __init__(self, backbone: nn.Module, head: MILClassifierHead):
        super().__init__()
        self.backbone = backbone
        self.head = head
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # keep frozen backbone in eval mode (no dropout)
        return self


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


@torch.no_grad()
def predict(head: MILClassifierHead, features, mask, device, batch_size: int = 64) -> np.ndarray:
    head.eval()
    probabilities = []
    for begin in range(0, len(features), batch_size):
        logits = head(
            features[begin : begin + batch_size].to(device),
            mask[begin : begin + batch_size].to(device),
        )
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def train_head(
    head, train_bags, train_y, val_bags, val_y, settings: dict, device
) -> tuple[dict, pd.DataFrame]:
    """Only the head's parameters go to the optimizer. Best epoch by validation AP or loss."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    head.to(device)
    positives = float(train_y.sum())
    pos_weight = torch.tensor((len(train_y) - positives) / positives, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    features, mask = train_bags
    targets = torch.from_numpy(train_y.astype(np.float32))
    generator = torch.Generator().manual_seed(settings["seed"])
    history, best, best_state, stale = [], None, None, 0

    for epoch in range(1, settings["epochs"] + 1):
        head.train()
        order = torch.randperm(len(features), generator=generator)
        epoch_loss = 0.0
        for begin in range(0, len(order), settings["batch_size"]):
            batch = order[begin : begin + settings["batch_size"]]
            logits = head(features[batch].to(device), mask[batch].to(device))
            loss = loss_fn(logits, targets[batch].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        val_p = predict(head, *val_bags, device)
        val_logits = torch.logit(torch.from_numpy(val_p).clamp(1e-6, 1 - 1e-6))
        val_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.cpu())(
            val_logits, torch.from_numpy(val_y.astype(np.float32))
        ).item()
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / len(order),
            "val_loss": val_loss,
            "val_auroc": roc_auc_score(val_y, val_p),
            "val_ap": average_precision_score(val_y, val_p),
        }
        history.append(row)
        score = -row["val_loss"] if settings["selection_metric"] == "val_loss" else row["val_ap"]
        if best is None or score > best + settings["min_delta"]:
            best, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_row = row
        else:
            stale += 1
            if stale >= settings["patience"]:
                break

    head.load_state_dict(best_state)
    return best_row, pd.DataFrame(history)


def recall_threshold(y_true: np.ndarray, probability: np.ndarray, target_recall: float) -> float:
    """Highest threshold whose validation recall reaches target_recall."""
    positives = np.sort(probability[y_true == 1])[::-1]
    needed = int(np.ceil(target_recall * len(positives)))
    return float(positives[max(needed, 1) - 1])


def evaluate(
    cohort: pd.DataFrame, probability: np.ndarray, threshold: float, run_id: str
) -> tuple[dict, pd.DataFrame]:
    metrics = binary_metrics(cohort["y_true"].to_numpy(), probability, threshold)
    frame = cohort[["patient_id", "split", "y_true"]].copy()
    frame["p_positive"] = probability
    frame["threshold"] = threshold
    frame["y_pred"] = (probability >= threshold).astype(int)
    frame["model"] = "MOMENT+MIL"
    frame["run_id"] = run_id
    return metrics, frame


def make_run_dir(output_root: Path, model_name: str) -> tuple[str, Path]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}_moment_{model_name.split('/')[-1].lower()}"
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def save_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
