"""Run fixed-patience trials until one reaches 60 epochs, then evaluate it once."""

import argparse
import json
import os
from pathlib import Path

import optuna
import yaml
from Members.sang.model.DL.raw_pipeline import train_raw_ecg
from Members.sang.model.evaluation.evaluation import evaluate_saved_model
from run import write_trial_configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, out = args.data_root.resolve(), args.output.resolve()
    os.umask(0o077)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    search_path = Path(__file__).with_name("search_space.yaml")
    search = yaml.safe_load(search_path.read_text())["search"]
    (out / "search_space.yaml").write_text(search_path.read_text())
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage="sqlite:///" + str(out / "study.sqlite3"),
        study_name="tcn_transformer_until_60",
    )

    def objective(trial):
        choice = write_trial_configs(trial, out, data, 100, 10, search)
        run = train_raw_ecg(
            choice, run_id=f"trial_{trial.number:03d}", device_override="cuda", mlflow_logging=False
        )
        metadata = json.loads((run / "metadata.json").read_text())
        metrics = json.loads((run / "metrics.json").read_text())["validation"]
        trial.set_user_attr("run_dir", str(run))
        trial.set_user_attr("completed_epochs", metadata["completed_epochs"])
        trial.set_user_attr("best_epoch", metadata["best_epoch"])
        return float(metrics["average_precision"])

    while True:
        study.optimize(objective, n_trials=1, n_jobs=1)
        study.trials_dataframe().to_csv(out / "optuna_trials.csv", index=False)
        trial = study.trials[-1]
        if trial.user_attrs["completed_epochs"] >= 60:
            break
    run = Path(trial.user_attrs["run_dir"])
    config = {
        "basic_config": str(out / "configs" / f"trial_{trial.number:03d}" / "basic.yaml"),
        "evaluation": {
            "target": "dl",
            "artifact_dir": str(run),
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
    eval_path = out / "selected_test.yaml"
    eval_path.write_text(yaml.safe_dump(config, sort_keys=False))
    result = evaluate_saved_model(eval_path, run_id="selected_test")
    (out / "selected.json").write_text(
        json.dumps(
            {
                "trial": trial.number,
                "params": trial.params,
                "completed_epochs": trial.user_attrs["completed_epochs"],
                "best_epoch": trial.user_attrs["best_epoch"],
                "validation_ap": trial.value,
                "run_dir": str(run),
                "test_dir": str(result),
                "test_metrics": json.loads((result / "metrics.json").read_text()),
                "selection": "first trial completing at least 60 epochs; checkpoint by validation loss",
            },
            indent=2,
        )
    )
    print("Completed:", out, flush=True)


if __name__ == "__main__":
    main()
