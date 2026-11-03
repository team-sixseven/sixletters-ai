"""Small exploratory QT check; not a population or clinical validation."""

import numpy as np
import wfdb
from scipy.signal import resample_poly
from server import experimental_qt

ann = wfdb.rdann("sel100", "q1c", pn_dir="qtdb")
start = 149800
signal, meta = wfdb.rdsamp(
    "sel100", pn_dir="qtdb", sampfrom=start, sampto=start + 2500, channels=[0]
)
assert meta["fs"] == 250
samples = resample_poly(signal[:, 0], 4, 5)
assert len(samples) == 2000
result = experimental_qt({"samples": samples.tolist(), "quality_excluded": [False]})
reference = []
for i, symbol in enumerate(ann.symbol):
    if symbol != "N" or not start < ann.sample[i] < start + 2500:
        continue
    onset = next(
        (ann.sample[k] for k in range(i - 1, max(-1, i - 5), -1) if ann.symbol[k] == "("), None
    )
    t = next((k for k in range(i + 1, min(len(ann.sample), i + 7)) if ann.symbol[k] == "t"), None)
    end = (
        ann.sample[t + 1]
        if t is not None and t + 1 < len(ann.sample) and ann.symbol[t + 1] == ")"
        else None
    )
    if onset is not None and end is not None and start < onset < ann.sample[i] < end < start + 2500:
        reference.append(((int(ann.sample[i]) - start) / 250, (int(end) - int(onset)) * 4))
errors = []
for r, qt in reference:
    nearest = (
        min(result["beats"], key=lambda b: abs(b["r_peak_s"] - r)) if result["beats"] else None
    )
    if nearest and abs(nearest["r_peak_s"] - r) < 0.15:
        errors.append(abs(nearest["qt_ms"] - qt))
print(
    "reference_beats:",
    len(reference),
    "matched:",
    len(errors),
    "median_absolute_error_ms:",
    float(np.median(errors)) if errors else None,
    "max_absolute_error_ms:",
    max(errors) if errors else None,
)
print("Single public 10s strip only; not validated on the local XYZ recordings.")
