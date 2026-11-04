import numpy as np
import scipy.signal as sig
import wfdb
from Members.sang.experiments.monorouter_holter.clinical_features import (
    detect_peaks_by_lead,
    filtered_rr,
)

beat_symbols = set("NLRBAaJjSVF/feQ") | {"E", "n", "r"}
for name in ["100", "105", "200", "207"]:
    rec = wfdb.rdrecord(name, pn_dir="mitdb")
    ann = wfdb.rdann(name, "atr", pn_dir="mitdb")
    signals = sig.resample_poly(rec.p_signal, 5, 9, axis=0)
    sos = sig.butter(4, [0.5, 40], btype="bandpass", fs=200, output="sos")
    signals = sig.sosfiltfilt(sos, signals, axis=0)
    length = len(signals) // 6000 * 6000
    wave = np.column_stack((signals[:length], np.zeros(length)))
    out = detect_peaks_by_lead(
        wave, np.ones(length // 6000, dtype=bool), 200, 30, [True, True, False]
    )
    samples = np.asarray(ann.sample)
    symbols = np.asarray(ann.symbol)
    mask = np.array([x in beat_symbols for x in symbols]) & (samples * 5 / 9 < length)
    ref = samples[mask] * 5 / 9
    symbol = symbols[mask]
    pred = out["peaks"]
    matched = {}
    i = j = 0
    tolerance = 20
    while i < len(ref) and j < len(pred):
        if pred[j] < ref[i] - tolerance:
            j += 1
        elif pred[j] > ref[i] + tolerance:
            i += 1
        else:
            matched[j] = i
            i += 1
            j += 1
    tp = len(matched)
    fp = len(pred) - tp
    fn = len(ref) - tp
    _, accepted, _ = filtered_rr(pred, out["block_ids"], 200)
    nn_tp = nn_fp = nn_fn = 0
    for k, good in enumerate(accepted):
        if k not in matched or k + 1 not in matched:
            continue
        a, b = matched[k], matched[k + 1]
        if b != a + 1:
            continue
        true = symbol[a] == "N" and symbol[b] == "N"
        if good and true:
            nn_tp += 1
        elif good and not true:
            nn_fp += 1
        elif not good and true:
            nn_fn += 1
    print(
        name,
        "lead",
        out["lead"],
        "beats",
        len(ref),
        "sensitivity",
        round(tp / (tp + fn), 4),
        "precision",
        round(tp / (tp + fp), 4),
        "rr_accepted_not_NN",
        nn_fp,
        "rr_accepted",
        nn_tp + nn_fp,
        "NN_rejected",
        nn_fn,
        "NN_total",
        nn_tp + nn_fn,
        flush=True,
    )
