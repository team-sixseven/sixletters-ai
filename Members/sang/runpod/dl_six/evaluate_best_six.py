import os
import sys
import json
import tarfile
import traceback
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

base = Path("/workspace/dl_six")
repo = base / "repo"
sys.path.insert(0, str(repo))
from Members.sang.model.evaluation.evaluation import evaluate_saved_model  # noqa: E402

os.umask(0o077)
source = base / "best_previous"
source.mkdir(exist_ok=True)
with tarfile.open(base / "dl_six_selected.tar.gz") as tar:
    tar.extractall(source, filter="data")
selected = [
    "resnet_attention_lstm_fcn",
    "tcn_attention_lstm_fcn",
    "resnet_bilstm",
    "tcn_bilstm",
    "resnet_transformer",
]
status = {}
output = base / "best_test_evaluation"
output.mkdir(exist_ok=True)
for name in selected:
    try:
        artifact = source / name / "models"
        resolved_path = artifact / "resolved_config.yaml"
        resolved = yaml.safe_load(resolved_path.read_text())
        basic = resolved["basic"]
        raw = basic["data"]["raw_ecg"]
        data = base / "snapshot/Holter"
        for key, value in {
            "split_path": data / "final_split_augmented.csv",
            "quality_index_path": data / "segment_rpeaks.csv",
            "lead_metadata_path": data / "ECG_tem_clean.csv",
            "wave_directory": data / "windows",
        }.items():
            raw[key] = str(value)
        basic_path = output / f"{name}_basic.yaml"
        basic_path.write_text(yaml.safe_dump(basic, sort_keys=False))
        resolved["basic_path"] = str(basic_path)
        resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
        e = {
            "basic_config": str(basic_path),
            "evaluation": {
                "target": "dl",
                "artifact_dir": str(artifact),
                "split": "validation",
                "threshold_override": None,
                "results_directory": str(output / name),
                "mlflow_logging": False,
                "metrics": [
                    "auroc",
                    "average_precision",
                    "accuracy",
                    "precision",
                    "recall",
                    "sensitivity",
                    "specificity",
                    "f1",
                    "confusion_matrix",
                ],
            },
        }
        cfg = output / f"{name}_evaluation.yaml"
        cfg.write_text(yaml.safe_dump(e, sort_keys=False))
        val = evaluate_saved_model(cfg, run_id="validation_check")
        old = pd.read_csv(
            artifact / "validation_predictions.csv", dtype={"patient_id": str}
        )
        new = pd.read_csv(val / "predictions.csv", dtype={"patient_id": str})
        both = old.merge(
            new, on="patient_id", validate="one_to_one", suffixes=("_old", "_new")
        )
        if len(both) != len(old) or len(both) != len(new):
            raise RuntimeError("validation cohort mismatch")
        delta = float(np.max(np.abs(both.p_positive_old - both.p_positive_new)))
        print(
            "VALIDATION_REPRO", name, "n", len(both), "max_abs_delta", delta, flush=True
        )
        if delta > 1e-3:
            raise RuntimeError("validation scores did not reproduce within 0.001")
        test = evaluate_saved_model(cfg, split_override="test", run_id="test")
        metrics = json.loads((test / "metrics.json").read_text())
        status[name] = {
            "validation_max_abs_delta": delta,
            "test_dir": str(test),
            "test_metrics": metrics,
            "artifact": str(artifact),
        }
        (output / "status.json").write_text(json.dumps(status, indent=2))
        print(
            "TEST_DONE",
            name,
            "AP",
            metrics["average_precision"],
            "AUROC",
            metrics["auroc"],
            flush=True,
        )
    except Exception as exc:
        status[name] = {"error": type(exc).__name__, "message": str(exc)}
        (output / "status.json").write_text(json.dumps(status, indent=2))
        traceback.print_exc()
        print("ERROR", name, flush=True)
print("FIVE_EVALUATIONS_FINISHED", flush=True)
