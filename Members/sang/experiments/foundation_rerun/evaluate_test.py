"""Evaluate the selected adapter checkpoint on the previously used test split."""

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "Members/sang/results/foundation_rerun_20261103"
DEEP = Path("/home/sang/deepecg-3channel")


def load_module(path):
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("evaluation_original", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def deep_test_groups(module):
    from scipy.signal import resample_poly
    from Members.sang.model.config import load_basic_config
    from Members.sang.model.DL.dataloader.raw_waveform import load_raw_cohort

    config_path = ROOT / "Members/sang/model/basic.yaml"
    config = load_basic_config(config_path)
    frames, indices, _ = load_raw_cohort(config, config_path)
    patients = {}
    for row in frames["test"].itertuples():
        pid = re.sub(r"-\d+$", "", row.patient_id)
        chosen = np.flatnonzero(indices[row.patient_id])
        chosen = chosen[np.linspace(0, len(chosen) - 1, min(8, len(chosen)), dtype=int)]
        source = np.load(row.ecg_path, mmap_mode="r", allow_pickle=False)
        windows = np.stack(
            [
                np.asarray(
                    source[int(i) * 6000 : int(i) * 6000 + 2000], dtype=np.float32
                ).T
                for i in chosen
            ]
        )
        windows[:, np.logical_not(row.lead_presence), :] = 0
        windows = resample_poly(windows, 5, 4, axis=-1).astype(np.float32)
        if windows.shape[1:] != (3, 2500) or not np.isfinite(windows).all():
            raise ValueError("invalid waveform")
        patient = patients.setdefault(
            pid, {"pid": pid, "split": "test", "label": int(row.y_true), "records": []}
        )
        if patient["label"] != int(row.y_true):
            raise ValueError("inconsistent patient label")
        patient["records"].append(
            {
                "id": row.patient_id,
                "windows": windows,
                "has_z": bool(row.lead_presence[2]),
            }
        )
    return list(patients.values())


def fm_predict(model, patients, device):
    rows = []
    model.eval()
    with torch.inference_mode():
        for patient in patients:
            per_record = []
            for record in patient["records"]:
                logits = []
                for batch in np.array_split(
                    record, max(1, int(np.ceil(len(record) / 4)))
                ):
                    logits.extend(
                        model(torch.from_numpy(batch).to(device)).cpu().tolist()
                    )
                per_record.append(float(np.mean(logits)))
            score = float(torch.sigmoid(torch.tensor(np.mean(per_record))).item())
            rows.append(
                {"scd_4year_label": int(patient["label"]), "model_score": score}
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["fm", "deep", "founder"])
    args = ap.parse_args()
    kind = args.model
    name = {"fm": "ecg_fm", "deep": "deepecg", "founder": "ecgfounder"}[kind]
    out = BASE / name
    dest = out / "test.csv"
    metrics_path = out / "test_metrics.json"
    if dest.exists() or metrics_path.exists():
        raise SystemExit("Output exists; preserve prior test results")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA required")
    checkpoint = torch.load(
        out / "best_adapter.pt", map_location=device, weights_only=True
    )
    if kind == "fm":
        mod = load_module(ROOT / "Members/sang/experiments/ecg_fm/train_adapter.py")
        model = mod.AdapterModel(ROOT / ".poc-cache/ecg-fm/finetuned.pt").to(device)
        model.adapter.load_state_dict(checkpoint["adapter"])
        model.head.load_state_dict(checkpoint["head"])
        groups = mod.load_patients(Path("/tmp/sang_ecg_fm_poc_20261103"))
        rows = fm_predict(model, groups["test"], device)
    elif kind == "founder":
        mod = load_module(ROOT / "Members/sang/experiments/ecgfounder/train.py")
        model = mod.AdaptedFounder(
            mod.make_backbone(ROOT / ".poc-cache/ecgfounder/12_lead_ECGFounder.pth")
        ).to(device)
        model.input_adapter.load_state_dict(checkpoint["input_adapter"])
        model.output_adapter.load_state_dict(checkpoint["output_adapter"])
        groups = mod.load_groups(splits=("test",))
        rows, _ = mod.evaluate(model, groups["test"], device, 4)
    else:
        mod = load_module(DEEP / "holter_original_sampling.py")
        model = mod.AdaptedECG(
            mod.load_backbone(
                DEEP / "checkpoints/SSL_pretrained.pt",
                DEEP / "weights/deepecg_ssl_clsa_cvd_fold0.safetensors",
                0,
                public_encoder=True,
            )
        ).to(device)
        model.input_adapter.load_state_dict(checkpoint["input_adapter"])
        model.output_adapter.load_state_dict(checkpoint["output_adapter"])
        rows, _ = mod.predict(model, deep_test_groups(mod), device, 4)
    y = np.array([int(r["scd_4year_label"]) for r in rows])
    p = np.array([float(r["model_score"]) for r in rows])
    if (
        len(y) != 89
        or sum(y) != 10
        or not np.isfinite(p).all()
        or not ((p >= 0) & (p <= 1)).all()
    ):
        raise ValueError("invalid test data or predictions")
    tn, fp, fn, tp = map(int, confusion_matrix(y, p >= 0.5, labels=[0, 1]).ravel())
    m = {
        "n": len(y),
        "positive": int(sum(y)),
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": 0.5,
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    if kind == "fm":
        prior = json.loads((out / "metrics.json").read_text())["splits"]["test"]
        if (
            any(
                not np.isclose(m[k], prior[k], atol=1e-7)
                for k in ("auroc", "average_precision", "brier")
            )
            or m["confusion"] != prior["confusion"]
        ):
            raise ValueError("test metrics disagree with original run")
    with dest.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dest.chmod(0o600)
    metrics_path.write_text(json.dumps(m, indent=2) + "\n")
    metrics_path.chmod(0o600)
    print(name, json.dumps(m), flush=True)


if __name__ == "__main__":
    main()
