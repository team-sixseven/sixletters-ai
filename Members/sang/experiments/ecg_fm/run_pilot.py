"""Frozen ECG-FM representations and a patient-level SCD linear probe.

XYZ-to-first-three-input-slots is arbitrary, not a 12-lead reconstruction.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from fairseq_signals.models import build_model_from_checkpoint
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

source, ckpt, dest = map(Path, sys.argv[1:4])
dest.mkdir(parents=True, exist_ok=True)
records = json.loads((source / "manifest.json").read_text())["records"]
model = build_model_from_checkpoint(checkpoint_path=str(ckpt)).eval().cuda()
groups = {}
with torch.inference_mode():
    for i, row in enumerate(records):
        sig = np.load(source / (row["record"] + ".npy"), allow_pickle=False).astype(np.float32)
        sig = resample_poly(sig, 5, 2, axis=-1).astype(np.float32)
        mean = sig.mean(axis=-1, keepdims=True)
        std = sig.std(axis=-1, keepdims=True)
        sig = (sig - mean) / np.where(std < 1e-8, 1, std)
        sig[:, np.logical_not(row["lead_presence"]), :] = 0
        x = np.zeros((len(sig), 12, 2500), dtype=np.float32)
        x[:, :3, :] = sig
        out = model(source=torch.from_numpy(x).cuda())
        emb = out["encoder_out"].mean(dim=1).mean(dim=0).cpu().numpy()
        if not np.isfinite(emb).all():
            raise ValueError("Non-finite embeddings")
        base = re.sub(r"-\d+$", "", row["record"])
        item = groups.setdefault(
            base, {"split": row["split"], "label": row["label"], "embeddings": []}
        )
        if (item["split"], item["label"]) != (row["split"], row["label"]):
            raise ValueError("Inconsistent patient split or label")
        item["embeddings"].append(emb)
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(records)} records", flush=True)
patients = [
    {
        **{"patient": k, "split": v["split"], "label": v["label"]},
        "embedding": np.mean(v["embeddings"], axis=0),
    }
    for k, v in groups.items()
]


def dataset(split):
    rows = [r for r in patients if r["split"] == split]
    return np.stack([r["embedding"] for r in rows]), np.array([r["label"] for r in rows]), rows


Xtr, ytr, _ = dataset("train")
Xv, yv, _ = dataset("validation")
Xt, yt, testrows = dataset("test")
if any(len(np.unique(y)) != 2 for y in (ytr, yv, yt)):
    raise ValueError("Both classes are required in each split")
results = []
for c in (0.001, 0.01, 0.1, 1.0, 10.0):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=42),
    )
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xv)[:, 1]
    results.append((float(average_precision_score(yv, p)), c, clf))
best_ap, best_c, clf = max(results, key=lambda x: x[0])
result = {
    "method": "frozen finetuned ECG-FM, arbitrary XYZ-to-first-three slots, other leads zero, 5s 200Hz resampled to 500Hz, mean segment and record embeddings, logistic regression",
    "selected_C": best_c,
    "selection": "validation average precision",
    "validation_candidates": [{"C": c, "AP": ap} for ap, c, _ in results],
    "splits": {},
}
for split, X, y in [("train", Xtr, ytr), ("validation", Xv, yv), ("test", Xt, yt)]:
    p = clf.predict_proba(X)[:, 1]
    result["splits"][split] = {
        "n": len(y),
        "positive": int(y.sum()),
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "sensitivity_at_0.5": float(((p >= 0.5) & (y == 1)).sum() / y.sum()),
        "specificity_at_0.5": float(((p < 0.5) & (y == 0)).sum() / (y == 0).sum()),
    }
(dest / "metrics.json").write_text(json.dumps(result, indent=2))
np.savez_compressed(
    dest / "embeddings.npz",
    X_train=Xtr,
    y_train=ytr,
    X_validation=Xv,
    y_validation=yv,
    X_test=Xt,
    y_test=yt,
)
print(json.dumps(result, indent=2), flush=True)
