"""Isolated Optuna study of the repository's raw TCN + transformer pipeline.

This does not train clinical ML versions or fuse modalities; those are separate decisions.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import optuna
import yaml

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from Members.sang.model.DL.raw_pipeline import train_raw_ecg  # noqa: E402
from Members.sang.model.evaluation.evaluation import evaluate_saved_model  # noqa: E402


def write_trial_configs(trial, output, data_root, epochs, patience, search):
    def suggest(name):
        spec = search[name]
        if spec["type"] == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        if spec["type"] == "float":
            return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        raise ValueError(f"Unsupported search type for {name}")

    source = REPO / "Members/sang/model"
    stage = output / "configs" / f"trial_{trial.number:03d}"
    dl = stage / "DL"
    dl.mkdir(mode=0o700, parents=True, exist_ok=False)
    (dl / "CNN").mkdir()
    (dl / "Transformer").mkdir()
    basic = yaml.safe_load((source / "basic.yaml").read_text())
    raw = basic["data"]["raw_ecg"]
    raw.update(
        {
            "split_path": str(data_root / "Holter/final_split_augmented.csv"),
            "wave_directory": str(data_root / "Holter/windows"),
            "quality_index_path": str(data_root / "Holter/segment_rpeaks.csv"),
            "lead_metadata_path": str(data_root / "Holter/ECG_tem_clean.csv"),
        }
    )
    # Do not load or upload clinical data during the ECG-only study.
    basic["outputs"]["dl_model_directory"] = str(output / "models")
    train = yaml.safe_load((source / "DL/training.yaml").read_text())
    train["epochs"] = epochs
    train["early_stopping"]["patience"] = patience
    train["device"] = "cuda"
    train["learning_rate"] = suggest("learning_rate")
    train["weight_decay"] = suggest("weight_decay")
    train["classifier"]["dropout"] = suggest("classifier_dropout")
    cnn = copy.deepcopy(yaml.safe_load((source / "DL/CNN/model_set.yaml").read_text()))
    cnn["models"]["TCN"].update(
        {"hidden_dim": suggest("tcn_hidden_dim"), "kernel_size": suggest("tcn_kernel_size")}
    )
    trans = copy.deepcopy(yaml.safe_load((source / "DL/Transformer/transformer.yaml").read_text()))
    trans["models"]["transformer_enc"].update(
        {"dropout": suggest("transformer_dropout"), "layers": suggest("transformer_layers")}
    )
    for path, data in [
        (stage / "basic.yaml", basic),
        (dl / "training.yaml", train),
        (dl / "CNN/model_set.yaml", cnn),
        (dl / "Transformer/transformer.yaml", trans),
        (dl / "model_choice.yaml", {"cnn_model": "TCN", "sequence_model": "transformer_enc"}),
    ]:
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return dl / "model_choice.yaml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True, help="directory containing Holter")
    p.add_argument("--output", type=Path, required=True, help="new private output directory")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    a = p.parse_args()
    if min(a.trials, a.epochs, a.patience) < 1:
        p.error("positive trials, epochs and patience required")
    data = a.data_root.resolve()
    out = a.output.resolve()
    required = [
        data / "Holter/final_split_augmented.csv",
        data / "Holter/windows",
        data / "Holter/segment_rpeaks.csv",
        data / "Holter/ECG_tem_clean.csv",
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Holter inputs: {missing}")
    os.umask(0o077)
    search_path = Path(__file__).with_name("search_space.yaml")
    search = yaml.safe_load(search_path.read_text())["search"]
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    (out / "search_space.yaml").write_text(search_path.read_text())
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage="sqlite:///" + str(out / "study.sqlite3"),
        study_name="tcn_transformer_raw_scd",
    )

    def objective(trial):
        choice = write_trial_configs(trial, out, data, a.epochs, a.patience, search)
        run = train_raw_ecg(
            choice, run_id=f"trial_{trial.number:03d}", device_override="cuda", mlflow_logging=False
        )
        metrics = json.loads((run / "metrics.json").read_text())["validation"]
        trial.set_user_attr("run_dir", str(run))
        trial.set_user_attr("validation_auroc", metrics["auroc"])
        # Trainer picks the epoch by validation loss; Optuna ranks resulting checkpoints by validation AP.
        return float(metrics["average_precision"])

    study.optimize(objective, n_trials=a.trials, n_jobs=1)
    study.trials_dataframe().to_csv(out / "optuna_trials.csv", index=False)
    best_run = Path(study.best_trial.user_attrs["run_dir"])
    best_basic = out / "configs" / f"trial_{study.best_trial.number:03d}" / "basic.yaml"
    evaluation = {
        "basic_config": str(best_basic),
        "evaluation": {
            "target": "dl",
            "artifact_dir": str(best_run),
            "split": "test",
            "threshold_override": None,
            "results_directory": str(out / "evaluation"),
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
    eval_yaml = out / "selected_test.yaml"
    eval_yaml.write_text(yaml.safe_dump(evaluation, sort_keys=False))
    result = evaluate_saved_model(eval_yaml, run_id="selected_test")
    (out / "best.json").write_text(
        json.dumps(
            {
                "trial": study.best_trial.number,
                "params": study.best_params,
                "validation_ap": study.best_value,
                "run_dir": str(best_run),
                "epoch_selection": "validation_loss",
                "trial_selection": "validation_average_precision",
                "test_evaluated": True,
                "test_dir": str(result),
                "test_metrics": json.loads((result / "metrics.json").read_text()),
            },
            indent=2,
        )
    )
    print("Completed study:", out, flush=True)


if __name__ == "__main__":
    main()
