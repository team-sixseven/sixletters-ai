"""Run ten validation-selected trials per model; test only 60-epoch-eligible models."""

import argparse
import json
import os
from pathlib import Path

import optuna
import yaml
from Members.sang.model.DL.raw_pipeline import train_raw_ecg
from Members.sang.model.evaluation.evaluation import evaluate_saved_model
from optuna.trial import TrialState
from run import REPO, write_trial_configs

COMBINATIONS = [
    ("tcn_transformer", "TCN", "transformer_enc"),
    ("resnet_attention_lstm_fcn", "ResNet", "Attention LSTM FCN"),
    ("tcn_attention_lstm_fcn", "TCN", "Attention LSTM FCN"),
    ("resnet_bilstm", "ResNet", "BiLSTM"),
    ("tcn_bilstm", "TCN", "BiLSTM"),
    ("resnet_transformer", "ResNet", "transformer_enc"),
]


def status(out, stage, **extra):
    path = out / "multi_status.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps({"stage": stage, **extra}, indent=2))
    temp.replace(path)
    print(stage, extra, flush=True)


def write_configs(trial, out, data, cnn, sequence, search):
    def suggest(name):
        spec = search[name]
        if spec["type"] == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))

    src = REPO / "Members/sang/model"
    stage = out / "configs" / f"trial_{trial.number:03d}"
    dl = stage / "DL"
    for folder in ("CNN", "LSTM", "Transformer"):
        (dl / folder).mkdir(mode=0o700, parents=True, exist_ok=False)
    basic = yaml.safe_load((src / "basic.yaml").read_text())
    basic["data"]["raw_ecg"].update(
        {
            "split_path": str(data / "Holter/final_split_augmented.csv"),
            "wave_directory": str(data / "Holter/windows"),
            "quality_index_path": str(data / "Holter/segment_rpeaks.csv"),
            "lead_metadata_path": str(data / "Holter/ECG_tem_clean.csv"),
        }
    )
    basic["outputs"]["dl_model_directory"] = str(out / "models")
    training = yaml.safe_load((src / "DL/training.yaml").read_text())
    training.update(
        epochs=100,
        device="cuda",
        learning_rate=suggest("learning_rate"),
        weight_decay=suggest("weight_decay"),
    )
    training["early_stopping"]["patience"] = 10
    training["classifier"]["dropout"] = suggest("classifier_dropout")
    cnn_cfg = yaml.safe_load((src / "DL/CNN/model_set.yaml").read_text())
    lstm_cfg = yaml.safe_load((src / "DL/LSTM/model_set.yaml").read_text())
    transformer_cfg = yaml.safe_load((src / "DL/Transformer/transformer.yaml").read_text())
    if cnn == "TCN":
        cnn_cfg["models"]["TCN"].update(
            hidden_dim=suggest("tcn_hidden_dim"), kernel_size=suggest("tcn_kernel_size")
        )
    seq_cfg = transformer_cfg if sequence == "transformer_enc" else lstm_cfg
    seq_cfg["models"][sequence]["dropout"] = suggest("sequence_dropout")
    if sequence == "transformer_enc":
        seq_cfg["models"][sequence]["layers"] = suggest("transformer_layers")
    for path, value in [
        (stage / "basic.yaml", basic),
        (dl / "training.yaml", training),
        (dl / "model_choice.yaml", {"cnn_model": cnn, "sequence_model": sequence}),
        (dl / "CNN/model_set.yaml", cnn_cfg),
        (dl / "LSTM/model_set.yaml", lstm_cfg),
        (dl / "Transformer/transformer.yaml", transformer_cfg),
    ]:
        path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
    return dl / "model_choice.yaml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    data = args.data_root.resolve()
    output = args.output.resolve()
    os.umask(0o077)
    output.mkdir(parents=True, exist_ok=True)
    search = yaml.safe_load(Path(__file__).with_name("multi_search.yaml").read_text())["search"]
    original_search = yaml.safe_load(Path(__file__).with_name("search_space.yaml").read_text())[
        "search"
    ]
    selected = []
    for index, (name, cnn, sequence) in enumerate(COMBINATIONS):
        folder = output / ("study" if index == 0 else "study_" + name)
        folder.mkdir(parents=True, exist_ok=True)
        study_name = "tcn_transformer_until_60" if index == 0 else name
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            storage="sqlite:///" + str(folder / "study.sqlite3"),
            study_name=study_name,
            load_if_exists=True,
        )
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if len(completed) > 10:
            raise RuntimeError(f"{name} has more than 10 completed trials")

        def objective(trial):
            status(
                output,
                "training",
                combination=name,  # noqa: B023
                trial=trial.number,
                completed_trials=len(completed),  # noqa: B023
            )
            choice = (
                write_trial_configs(trial, folder, data, 100, 10, original_search)  # noqa: B023
                if index == 0  # noqa: B023
                else write_configs(trial, folder, data, cnn, sequence, search)  # noqa: B023
            )
            run = train_raw_ecg(
                choice,
                run_id=f"trial_{trial.number:03d}",
                device_override="cuda",
                mlflow_logging=False,
            )
            meta = json.loads((run / "metadata.json").read_text())
            metrics = json.loads((run / "metrics.json").read_text())["validation"]
            trial.set_user_attr("run_dir", str(run))
            trial.set_user_attr("completed_epochs", meta["completed_epochs"])
            trial.set_user_attr("best_epoch", meta["best_epoch"])
            return float(metrics["average_precision"])

        while len(completed) < 10:
            study.optimize(objective, n_trials=1, n_jobs=1)
            completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
            study.trials_dataframe().to_csv(folder / "optuna_trials.csv", index=False)
        eligible = [t for t in completed if t.user_attrs["completed_epochs"] >= 60]
        best = max(eligible or completed, key=lambda t: t.value)
        selected.append(
            {
                "combination": name,
                "study_dir": str(folder),
                "trial": best.number,
                "params": best.params,
                "validation_ap": best.value,
                "completed_epochs": best.user_attrs["completed_epochs"],
                "best_epoch": best.user_attrs["best_epoch"],
                "run_dir": best.user_attrs["run_dir"],
                "eligible_for_test": bool(eligible),
                "test_status": "pending" if eligible else "held_no_60_epoch_trial",
                "completed_trials": len(completed),
            }
        )
        (output / "multi_selected.json").write_text(json.dumps(selected, indent=2))
    status(output, "evaluating", combinations=len(selected))
    for item in selected:
        if not item["eligible_for_test"]:
            continue
        folder = Path(item["study_dir"])
        run = Path(item["run_dir"])
        cfg = {
            "basic_config": str(folder / "configs" / f"trial_{item['trial']:03d}" / "basic.yaml"),
            "evaluation": {
                "target": "dl",
                "artifact_dir": str(run),
                "split": "test",
                "threshold_override": None,
                "results_directory": str(folder / "evaluation"),
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
        path = folder / "selected_test.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        result = evaluate_saved_model(path, run_id="multi_selected_test")
        item.update(test_status="complete", test_dir=str(result))
        (output / "multi_selected.json").write_text(json.dumps(selected, indent=2))
    status(output, "complete", combinations=len(selected))


if __name__ == "__main__":
    main()
