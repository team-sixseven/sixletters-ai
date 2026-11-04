"""Every fixed choice of the clinical ML cross-validation, in one place.

Rules agreed with the team on 2026-11-03:
- Repeated stratified k-fold (5 folds x 3 repeats) on the 705 development patients.
  A method trained on only445 is also tuned and evaluated within the original 445 (356 patients);
  the folds are the same, restricted to those patients.
- One hyperparameter setting per method, chosen by the log loss of its out-of-fold predictions.
- Missing values filled by iterative imputation fitted on each fold's training patients.
- Threshold fixed at 0.5, no probability calibration.
- Class weights fixed from each training set's SCD ratio (Settings.class_weight), not tuned (2026-11-03).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...config import REPO_ROOT
from ..feature_versions import get_feature_columns

# ---------- data files (relative to drive_root in basic.yaml) ----------
CHOICE_PATH = (
    Path(__file__).resolve().parents[1] / "model_choice.yaml"
)  # team YAML: basic.yaml + model parameter YAMLs
RAW_CLINICAL_FILE = "clincal/clinical_clean_4year_holter_max_raw.csv"  # blanks kept (scripts/data_prep/build_ml_max_raw.py)
PATIENT_FLAGS_FILE = "clincal/ml_max_patient_flags.csv"  # added / device / holter_available / any_missing per patient

# ---------- inputs ----------
ALL_FEATURES = get_feature_columns(
    5
)  # 23 clinical + 4 Holter; also the inputs of every imputation model

FEATURE_SETS = {
    "v5": ALL_FEATURES,
    "v4": get_feature_columns(4),  # team default, 23 clinical
    "core": [
        "Age",
        "Gender (male=1)",
        "NYHA class",
        "LVEF (%)",
        "Left atrial size (mm)",
        "eGFR",
        "Pro-BNP (ng/L)",
        "Ischemic etiology",
        "Prior Myocardial Infarction (yes=1)",
        "Holter ventricular premature beats in 24h",
        "Holter ventricular couplets",
        "Holter NSVT (yes=1)",
    ],
    # MUSIC sudden-death score items available here (Vazquez 2009): LA size in mm, prior MI, no LBBB.
    "MUSIC_SCD": [
        "Pro-BNP (ng/L)",
        "Left atrial size (mm)",
        "Holter NSVT (yes=1)",
        "Holter ventricular premature beats in 24h",
        "Prior Myocardial Infarction (yes=1)",
    ],
    "ProBNP": ["Pro-BNP (ng/L)"],  # reference model
}

# Skewed counts/concentrations: log(x + 1) before imputation and before logistic regression.
LOG_COLUMNS = [
    "Pro-BNP (ng/L)",
    "Holter ventricular premature beats in 24h",
    "Holter ventricular couplets",
    "cigarettes /year",
    "alcohol consumption (standard units)",
]

# ---------- methods ----------
MODELS = {
    "Logistic": "LR",
    "XGBoost": "XGBoost",
    "Random Forest": "RandomForest",
    "CatBoost": "CatBoost",
}
TRAIN_FILTERS = {
    "all": "all development patients",
    "only445": "patients from the original 445 only",
    "no_device": "patients without an implanted device",
}
LOGISTIC_C_GRID = np.logspace(-3, 1, 9)  # 0.001 ... 10, every value is tried
THRESHOLD = 0.5


@dataclass(frozen=True)
class Settings:
    n_splits: int = 5
    n_repeats: int = 3
    split_seed: int = 2026  # same folds as the earlier (10/31) results
    n_trials: int = 100  # Optuna trials per tree method (team default)
    optuna_seed: int = 42
    n_jobs: int = 15  # folds fitted in parallel within one trial
    n_boot: int = 1000  # paired bootstrap resamples for comparisons
    # "balanced": class weights fixed from each training set's SCD ratio (XGBoost scale_pos_weight = negatives /
    # positives, RF class_weight="balanced", CatBoost auto_class_weights="Balanced", logistic class_weight="balanced"),
    # not tuned. None: no fixed weights; tree models tune them with Optuna (the earlier kfold_logloss run).
    class_weight: str | None = "balanced"
    results_root: Path = REPO_ROOT / "Members/mingyu/results/ml_kfold"
    run_tag: str = "kfold_logloss_balanced"  # folder under results_root; change it when data, fold or weight settings change

    @property
    def result_dir(self) -> Path:
        return Path(self.results_root) / self.run_tag
