"""Prepare a small patient-split Holter ECG-FM pilot from validated quality segments."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Members.sang.model.config import load_basic_config
from Members.sang.model.DL.dataloader.raw_waveform import load_raw_cohort

basic_path = Path("Members/sang/model/basic.yaml").resolve()
basic = load_basic_config(basic_path)
frames, indices, audit = load_raw_cohort(basic, basic_path)
output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
manifest = []
for split, frame in frames.items():
    for row in frame.itertuples():
        valid = np.flatnonzero(indices[row.patient_id])
        chosen = valid[np.linspace(0, len(valid) - 1, min(8, len(valid)), dtype=int)]
        data = np.load(row.ecg_path, mmap_mode="r", allow_pickle=False)
        windows = np.stack(
            [
                np.asarray(data[int(i) * 6000 : int(i) * 6000 + 1000], dtype=np.float32).T
                for i in chosen
            ]
        )
        if not np.isfinite(windows).all():
            raise ValueError(f"Non-finite signal in {row.patient_id}")
        np.save(output / (row.patient_id + ".npy"), windows, allow_pickle=False)
        manifest.append(
            {
                "record": row.patient_id,
                "split": split,
                "label": int(row.y_true),
                "lead_presence": list(map(bool, row.lead_presence)),
                "indices": chosen.tolist(),
            }
        )
(output / "manifest.json").write_text(
    json.dumps({"records": manifest, "audit_summary": audit["split_summary"]}, indent=2)
)
print("Prepared", len(manifest), "records:", audit["split_summary"], flush=True)
