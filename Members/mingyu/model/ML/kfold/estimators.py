"""Model building, fitting and prediction for the four model types.

Tree models reuse the team code (ML/models.py) and the team parameter YAMLs; the Optuna search space is
the same as the `objective` in the team notebook (model_experiment_only_ML.ipynb).
Logistic regression gets its own preprocessing: log(x + 1) for skewed columns, standardization,
and one-hot encoding of categorical inputs with more than two levels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from ..models import build_estimator, make_preprocessor, prepare_categorical_frame
from .settings import LOG_COLUMNS


def suggest_parameters(
    model: str, trial: optuna.Trial, tune_class_weight: bool = True
) -> dict[str, Any]:
    """One Optuna proposal; ranges copied from the team notebook objective.

    tune_class_weight=False leaves the class-weight setting out of the search (it is then fixed by
    with_class_weight from the training labels).
    """
    if model == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0, step=0.5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
            "reg_alpha": trial.suggest_categorical("reg_alpha", [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
        }
        if tune_class_weight:
            params["scale_pos_weight"] = trial.suggest_categorical(
                "scale_pos_weight", [1.0, 2.0, 4.0, 8.0]
            )
        return params
    if model == "CatBoost":
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1500, step=100),
            "depth": trial.suggest_int("depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0, step=0.5),
            "rsm": trial.suggest_float("rsm", 0.6, 1.0, step=0.1),
            "border_count": trial.suggest_categorical("border_count", [16, 32, 64, 128]),
        }
        if tune_class_weight:
            params["auto_class_weights"] = trial.suggest_categorical(
                "auto_class_weights", ["None", "Balanced", "SqrtBalanced"]
            )
        params["bootstrap_type"] = trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli"]
        )
        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0.0, 5.0, step=0.5
            )
        else:
            params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0, step=0.1)
        return params
    if model == "Random Forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 2, 3, 4, 5, 6, 8, 10, 15]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, None]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }
        if tune_class_weight:
            params["class_weight"] = trial.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]
            )
        params["ccp_alpha"] = trial.suggest_categorical(
            "ccp_alpha", [0.0, 0.0001, 0.001, 0.005, 0.01]
        )
        params["bootstrap"] = trial.suggest_categorical("bootstrap", [True, False])
        params["max_samples"] = (
            trial.suggest_float("max_samples", 0.6, 1.0, step=0.1) if params["bootstrap"] else None
        )
        return params
    raise ValueError(f"Optuna 탐색 범위가 없는 모델입니다: {model}")


def with_class_weight(model: str, params: dict, y: np.ndarray, class_weight: str | None) -> dict:
    """params plus class weights fixed from the training labels y (class_weight="balanced"); None keeps params."""
    if class_weight is None:
        return params
    if class_weight != "balanced":
        raise ValueError(f"알 수 없는 class_weight: {class_weight} (balanced 또는 None)")
    if model == "XGBoost":
        return {**params, "scale_pos_weight": float((y == 0).sum() / (y == 1).sum())}
    if model == "CatBoost":
        return {**params, "auto_class_weights": "Balanced"}
    return {**params, "class_weight": "balanced"}  # Random Forest, Logistic


def loss_weights(y: np.ndarray, class_weight: str | None) -> np.ndarray | None:
    """Per-patient weights for log loss, matching training (class_weight="balanced"); None = unweighted."""
    return compute_sample_weight("balanced", y) if class_weight == "balanced" else None


def fit_model(
    model: str,
    params: dict,
    x: pd.DataFrame,
    y: np.ndarray,
    categorical: list[str],
    base_parameters: dict[str, dict],
    class_weight: str | None = None,
):
    """Fit one model on complete (already imputed) inputs; returns an object for predict_model."""
    categorical = [c for c in categorical if c in x.columns]
    params = with_class_weight(model, params, y, class_weight)
    if model == "Logistic":
        return logistic_pipeline(x, categorical, params["C"], params.get("class_weight")).fit(x, y)

    estimator = build_estimator(model, tree_parameters(model, params, base_parameters))
    if model == "CatBoost":
        estimator.fit(
            prepare_categorical_frame(x, categorical), y, cat_features=categorical, verbose=False
        )
        return estimator
    preprocessor = make_preprocessor(list(x.columns), categorical)
    return Pipeline([("preprocess", preprocessor), ("model", estimator)]).fit(x, y)


def predict_model(model: str, fitted, x: pd.DataFrame, categorical: list[str]) -> np.ndarray:
    """Probability of SCD for each row of x."""
    if model == "CatBoost":
        x = prepare_categorical_frame(x, [c for c in categorical if c in x.columns])
    positive = int(np.flatnonzero(np.asarray(fitted.classes_) == 1)[0])
    return np.asarray(fitted.predict_proba(x), dtype=np.float64)[:, positive]


def tree_parameters(model: str, params: dict, base_parameters: dict[str, dict]) -> dict:
    """Team YAML defaults overridden by the trial, without incompatible CatBoost leftovers."""
    merged = {**base_parameters[model], **params}
    if model == "CatBoost":
        if params.get("bootstrap_type") == "Bayesian":
            merged.pop("subsample", None)
        elif params.get("bootstrap_type") == "Bernoulli":
            merged.pop("bagging_temperature", None)
        if "auto_class_weights" in params:
            merged.pop("class_weights", None)
            merged.pop("scale_pos_weight", None)
    return merged


def logistic_pipeline(
    x: pd.DataFrame, categorical: list[str], c: float, class_weight: str | None = None
) -> Pipeline:
    columns = list(x.columns)
    one_hot = [col for col in categorical if x[col].nunique() > 2]
    logged = [col for col in columns if col in LOG_COLUMNS]
    plain = [col for col in columns if col not in one_hot and col not in logged]
    log_then_scale = make_pipeline(
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler()
    )
    preprocessor = ColumnTransformer(
        [
            ("log", log_then_scale, logged),
            ("numeric", StandardScaler(), plain),
            (
                "one_hot",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                one_hot,
            ),
        ],
        verbose_feature_names_out=False,
    )
    model = LogisticRegression(C=c, class_weight=class_weight, max_iter=5000)
    return Pipeline([("preprocess", preprocessor), ("model", model)])
