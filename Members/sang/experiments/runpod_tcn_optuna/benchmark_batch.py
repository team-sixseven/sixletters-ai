"""Measure actual-waveform GPU training steps without modifying study artifacts."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from Members.sang.model.DL.composed import build_composed_model_from_specs

torch.set_num_threads(4)
root = Path("/workspace/sang_tcn_20261101/data/Holter/windows")
files = sorted(root.glob("*.npy"))[:6]
waves = []
for f in files:
    arr = np.load(f, mmap_mode="r")
    if arr.shape[0] < 128 * 6000 or arr.shape[1] != 3:
        continue
    wave = (
        np.asarray(arr[: 128 * 6000], dtype=np.float32)
        .reshape(128, 6000, 3)
        .transpose(0, 2, 1)
        .copy()
    )
    waves.append(torch.from_numpy(wave))
if len(waves) < 6:
    raise RuntimeError("Need six sample waveforms")
training = {
    "classifier": {"hidden_dim": 32, "activation": "GELU", "dropout": 0.1},
    "mil": {"pooling": "attention", "hidden_dim": 32, "activation": "GELU", "dropout": 0.0},
}
basic = {}
for cnn, cnn_spec in [
    ("TCN", {"activation": "GELU", "hidden_dim": 64, "kernel_size": 5}),
    ("ResNet", {"activation": "GELU"}),
]:
    estimates = {}
    for batch in (2, 4, 6):
        free, _ = torch.cuda.mem_get_info()
        if batch == 6 and (4 not in estimates or estimates[4] * 1.5 > free / (1024**3) - 10):
            print(
                json.dumps(
                    {"model": cnn, "batch": batch, "status": "skipped_insufficient_headroom"}
                ),
                flush=True,
            )
            continue
        try:
            torch.cuda.reset_peak_memory_stats()
            model = build_composed_model_from_specs(
                {"cnn_model": cnn, "sequence_model": "transformer_enc"},
                basic,
                cnn_spec,
                {"activation": "GELU", "dropout": 0.1, "layers": 2},
                training,
            ).cuda()
            x = torch.stack(waves[:batch]).cuda()
            mask = torch.ones(batch, 128, dtype=torch.bool, device="cuda")
            positions = torch.arange(128, device="cuda").repeat(batch, 1)
            leads = torch.ones(batch, 3, dtype=torch.bool, device="cuda")
            torch.cuda.synchronize()
            start = time.monotonic()
            y = model(x, mask, positions, leads)
            y.sum().backward()
            torch.cuda.synchronize()
            sec = time.monotonic() - start
            peak = torch.cuda.max_memory_allocated() / 1024**3
            estimates[batch] = peak
            print(
                json.dumps(
                    {
                        "model": cnn,
                        "batch": batch,
                        "seconds": round(sec, 3),
                        "patients_per_second": round(batch / sec, 3),
                        "peak_gib": round(peak, 2),
                        "finite": bool(torch.isfinite(y).all()),
                    }
                ),
                flush=True,
            )
            del model, x, mask, positions, leads, y
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(json.dumps({"model": cnn, "batch": batch, "status": "oom"}), flush=True)
            break
