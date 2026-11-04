"""Train and validation log loss of the selected setting, recorded on every fold.

"Validation" is the held-out fold of the cross-validation. The x-axis depends on the model type:

| model         | x-axis (step)          | how                                                         |
|---------------|------------------------|-------------------------------------------------------------|
| XGBoost       | boosting round         | eval_set on training and held-out patients                  |
| CatBoost      | boosting round         | staged probabilities (CatBoost's own history is class-weighted) |
| Random Forest | number of trees used   | average of the first k trees' probabilities, k = 1..n_trees |
| Logistic      | C (inverse regularization) | every value of the C grid refitted                      |

The selected setting is refitted once per fold for this record (tuning itself keeps no curves).
With fixed class weights the losses are weighted like training and tuning (estimators.loss_weights).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import log_loss

from ..models import build_estimator, make_preprocessor, prepare_categorical_frame
from .estimators import logistic_pipeline, loss_weights, tree_parameters, with_class_weight
from .settings import LOGISTIC_C_GRID, Settings

if TYPE_CHECKING:  # experiment.py imports this module
    from .experiment import PreparedFold

STEP_LABEL = {
    "XGBoost": "boosting 반복 (트리 수)",
    "CatBoost": "boosting 반복 (트리 수)",
    "Random Forest": "사용한 트리 수",
    "Logistic": "C (클수록 규제가 약함)",
}


def record_loss_curves(
    model: str,
    params: dict,
    prepared: list[PreparedFold],
    categorical: list[str],
    base_parameters: dict[str, dict],
    settings: Settings,
) -> pd.DataFrame:
    """Long table: repeat, fold, step, train_loss, validation_loss (one row per fold and step)."""
    curves = Parallel(n_jobs=settings.n_jobs)(
        delayed(_fold_curve)(
            model,
            with_class_weight(model, params, fold.y_train, settings.class_weight),
            fold,
            categorical,
            base_parameters,
            settings.class_weight,
        )
        for fold in prepared
    )
    for fold, curve in zip(prepared, curves, strict=False):
        curve.insert(0, "fold", fold.fold)
        curve.insert(0, "repeat", fold.repeat)
    return pd.concat(curves, ignore_index=True)


def _fold_curve(
    model, params, fold: PreparedFold, categorical, base_parameters, class_weight
) -> pd.DataFrame:
    categorical = [c for c in categorical if c in fold.x_train.columns]
    if model == "Logistic":
        return _logistic_curve(fold, categorical, params.get("class_weight"), class_weight)
    estimator = build_estimator(model, tree_parameters(model, params, base_parameters))
    if model == "XGBoost":
        return _xgboost_curve(estimator, fold, categorical, class_weight)
    if model == "CatBoost":
        return _catboost_curve(estimator, fold, categorical, class_weight)
    return _random_forest_curve(estimator, fold, categorical, class_weight)


def _xgboost_curve(estimator, fold: PreparedFold, categorical, class_weight) -> pd.DataFrame:
    preprocessor = make_preprocessor(list(fold.x_train.columns), categorical)
    x_train, x_eval = preprocessor.fit_transform(fold.x_train), preprocessor.transform(fold.x_eval)
    estimator.set_params(eval_metric="logloss")
    weights = [loss_weights(fold.y_train, class_weight), loss_weights(fold.y_eval, class_weight)]
    estimator.fit(
        x_train,
        fold.y_train,
        eval_set=[(x_train, fold.y_train), (x_eval, fold.y_eval)],
        sample_weight_eval_set=weights if class_weight else None,
        verbose=False,
    )
    history = estimator.evals_result()
    train, validation = history["validation_0"]["logloss"], history["validation_1"]["logloss"]
    return pd.DataFrame(
        {"step": np.arange(1, len(train) + 1), "train_loss": train, "validation_loss": validation}
    )


def _catboost_curve(estimator, fold: PreparedFold, categorical, class_weight) -> pd.DataFrame:
    # The loss is recomputed from the staged probabilities so it is weighted exactly as for the other models
    # (loss_weights), whatever CatBoost's own history does with class weights.
    x_train = prepare_categorical_frame(fold.x_train, categorical)
    x_eval = prepare_categorical_frame(fold.x_eval, categorical)
    estimator.fit(x_train, fold.y_train, cat_features=categorical, verbose=False)
    positive = int(np.flatnonzero(np.asarray(estimator.classes_) == 1)[0])

    def staged_loss(x, y):
        weights = loss_weights(y, class_weight)
        return [
            log_loss(y, p[:, positive], labels=[0, 1], sample_weight=weights)
            for p in estimator.staged_predict_proba(x)
        ]

    train = staged_loss(x_train, fold.y_train)
    return pd.DataFrame(
        {
            "step": np.arange(1, len(train) + 1),
            "train_loss": train,
            "validation_loss": staged_loss(x_eval, fold.y_eval),
        }
    )


def _random_forest_curve(estimator, fold: PreparedFold, categorical, class_weight) -> pd.DataFrame:
    preprocessor = make_preprocessor(list(fold.x_train.columns), categorical)
    x_train, x_eval = preprocessor.fit_transform(fold.x_train), preprocessor.transform(fold.x_eval)
    estimator.fit(x_train, fold.y_train)
    positive = int(np.flatnonzero(estimator.classes_ == 1)[0])
    n_trees = np.arange(1, len(estimator.estimators_) + 1)

    def cumulative_loss(x, y):
        per_tree = np.array([tree.predict_proba(x)[:, positive] for tree in estimator.estimators_])
        running_mean = np.cumsum(per_tree, axis=0) / n_trees[:, None]
        weights = loss_weights(y, class_weight)
        return [log_loss(y, p, labels=[0, 1], sample_weight=weights) for p in running_mean]

    return pd.DataFrame(
        {
            "step": n_trees,
            "train_loss": cumulative_loss(x_train, fold.y_train),
            "validation_loss": cumulative_loss(x_eval, fold.y_eval),
        }
    )


def _logistic_curve(
    fold: PreparedFold, categorical, model_class_weight, class_weight
) -> pd.DataFrame:
    rows = []
    for c in LOGISTIC_C_GRID:
        pipeline = logistic_pipeline(fold.x_train, categorical, c, model_class_weight).fit(
            fold.x_train, fold.y_train
        )
        rows.append(
            {
                "step": float(c),
                "train_loss": log_loss(
                    fold.y_train,
                    pipeline.predict_proba(fold.x_train)[:, 1],
                    labels=[0, 1],
                    sample_weight=loss_weights(fold.y_train, class_weight),
                ),
                "validation_loss": log_loss(
                    fold.y_eval,
                    pipeline.predict_proba(fold.x_eval)[:, 1],
                    labels=[0, 1],
                    sample_weight=loss_weights(fold.y_eval, class_weight),
                ),
            }
        )
    return pd.DataFrame(rows)
