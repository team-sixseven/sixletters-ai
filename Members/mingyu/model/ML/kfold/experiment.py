"""One method = model x feature set x training-patient filter, tuned and evaluated on the fixed folds.

How a method is run:
1. For every fold, fit the imputer on that fold's training patients and fill training and held-out inputs.
   This does not depend on the hyperparameters, so it is done once per method.
2. Each candidate setting (an Optuna trial for trees, one C value for logistic regression) is fitted on
   every fold's training patients and predicts its held-out patients -> out-of-fold (OOF) predictions
   for the method's evaluation patients (Cohort.eval_mask: 705, or the 356 from the original 445 for
   only445) in each of the 3 repeats.
3. The candidate with the lowest OOF log loss on those patients (mean over repeats) is kept; its OOF
   predictions are the method's result. With fixed class weights (Settings.class_weight="balanced") the
   log loss is weighted the same way as training, so SCD and non-SCD patients count equally.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import log_loss

from .cohort import Cohort
from .curves import record_loss_curves
from .estimators import fit_model, loss_weights, predict_model, suggest_parameters
from .imputation import fit_imputer
from .plots import save_loss_curve, save_probability_hist
from .results import result_exists, save_result
from .settings import FEATURE_SETS, LOGISTIC_C_GRID, MODELS, TRAIN_FILTERS, Settings


@dataclass(frozen=True)
class Method:
    model: str  # Logistic / XGBoost / Random Forest / CatBoost
    features: str  # key of FEATURE_SETS
    train_filter: str = "all"  # key of TRAIN_FILTERS

    def __post_init__(self):
        if (
            self.model not in MODELS
            or self.features not in FEATURE_SETS
            or self.train_filter not in TRAIN_FILTERS
        ):
            raise ValueError(f"알 수 없는 설정: {self}")

    @property
    def name(self) -> str:
        """e.g. LR_MUSIC_SCD_iterative, XGBoost_v5_iterative_only445"""
        parts = [MODELS[self.model], self.features, "iterative"]
        if self.train_filter != "all":
            parts.append(self.train_filter)
        return "_".join(parts)


@dataclass
class PreparedFold:
    repeat: int
    fold: int
    eval_index: np.ndarray
    x_train: pd.DataFrame  # imputed, method's feature set only
    y_train: np.ndarray
    x_eval: pd.DataFrame
    y_eval: np.ndarray


def run_method(cohort: Cohort, method: Method, settings: Settings = Settings()) -> str:  # noqa: B008
    """Tune and evaluate one method; skipped when its result already exists."""
    if result_exists(settings, method.name):
        print(f"{method.name}: 결과 있음, 건너뜀")
        return method.name
    started = datetime.now()
    prepared = prepare_folds(cohort, method)
    evaluated = cohort.eval_mask(method.train_filter)

    with Parallel(n_jobs=settings.n_jobs) as parallel:

        def evaluate(params: dict) -> tuple[float, np.ndarray]:
            p = cross_validated_predictions(cohort, method, params, prepared, settings, parallel)
            return mean_log_loss(cohort.y[evaluated], p[evaluated], settings.class_weight), p

        if method.model == "Logistic":
            candidates = [{"C": float(c)} for c in LOGISTIC_C_GRID]
            scored = [(params, *evaluate(params)) for params in candidates]
        else:
            scored = _optuna_search(method.model, evaluate, settings)

    best_params, best_loss, best_p = min(scored, key=lambda row: row[1])
    trials = pd.DataFrame(
        [{"candidate": i, "log_loss": loss, **params} for i, (params, loss, _) in enumerate(scored)]
    )
    minutes = round((datetime.now() - started).total_seconds() / 60, 1)
    save_result(
        settings,
        cohort,
        method.name,
        best_p,
        trials,
        info={
            "method": {
                "model": method.model,
                "features": method.features,
                "train_filter": method.train_filter,
            },
            "input_columns": FEATURE_SETS[method.features],
            "missing": "iterative imputation from all 27 inputs, fitted on each fold's training patients",
            "class_weight": settings.class_weight,
            "selection": f"lowest OOF log loss over {len(scored)} candidates (mean of {settings.n_repeats} repeats)"
            + (", weighted like training (class_weight=balanced)" if settings.class_weight else ""),
            "best_parameters": best_params,
            "best_log_loss": best_loss,
            "n_train_per_fold": sorted({len(f.y_train) for f in prepared}),
            "n_evaluated": int(evaluated.sum()),
            "minutes": minutes,
        },
    )
    save_probability_hist(settings, cohort, method.name)
    curves = record_loss_curves(
        method.model, best_params, prepared, cohort.categorical, cohort.base_parameters, settings
    )
    save_loss_curve(settings, method.name, method.model, curves, best_params.get("C"))
    print(f"{method.name}: 저장 ({minutes}분, log loss {best_loss:.4f})")
    return method.name


def prepare_folds(cohort: Cohort, method: Method) -> list[PreparedFold]:
    columns = FEATURE_SETS[method.features]
    allowed = cohort.train_mask(method.train_filter)
    evaluated = cohort.eval_mask(method.train_filter)
    prepared = []
    for fold in cohort.folds:
        train_index = fold.train_index[allowed[fold.train_index]]
        eval_index = fold.eval_index[evaluated[fold.eval_index]]
        imputer = fit_imputer(cohort.x.iloc[train_index], cohort.categorical)
        prepared.append(
            PreparedFold(
                repeat=fold.repeat,
                fold=fold.fold,
                eval_index=eval_index,
                x_train=imputer.transform(cohort.x.iloc[train_index])[columns],
                y_train=cohort.y[train_index],
                x_eval=imputer.transform(cohort.x.iloc[eval_index])[columns],
                y_eval=cohort.y[eval_index],
            )
        )
    return prepared


def cross_validated_predictions(
    cohort: Cohort,
    method: Method,
    params: dict,
    prepared: list[PreparedFold],
    settings: Settings,
    parallel: Parallel,
) -> np.ndarray:
    """OOF probabilities, shape (patients, repeats); NaN for patients outside the method's evaluation set."""
    held_out = parallel(
        delayed(_fit_and_predict)(
            method.model,
            params,
            f,
            cohort.categorical,
            cohort.base_parameters,
            settings.class_weight,
        )
        for f in prepared
    )
    p = np.full((len(cohort.ids), settings.n_repeats), np.nan)
    for fold, probabilities in zip(prepared, held_out, strict=False):
        p[fold.eval_index, fold.repeat] = probabilities
    return p


def mean_log_loss(y: np.ndarray, p: np.ndarray, class_weight: str | None = None) -> float:
    """Mean over repeats; with class_weight="balanced" SCD and non-SCD patients count equally, as in training."""
    weights = loss_weights(y, class_weight)
    return float(
        np.mean(
            [log_loss(y, p[:, r], labels=[0, 1], sample_weight=weights) for r in range(p.shape[1])]
        )
    )


def _fit_and_predict(
    model, params, fold: PreparedFold, categorical, base_parameters, class_weight
) -> np.ndarray:
    fitted = fit_model(
        model, params, fold.x_train, fold.y_train, categorical, base_parameters, class_weight
    )
    return predict_model(model, fitted, fold.x_eval, categorical)


def _optuna_search(
    model: str, evaluate, settings: Settings
) -> list[tuple[dict, float, np.ndarray]]:
    scored = []

    def objective(trial: optuna.Trial) -> float:
        params = suggest_parameters(model, trial, tune_class_weight=settings.class_weight is None)
        loss, p = evaluate(params)
        scored.append((params, loss, p))
        return loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=settings.optuna_seed)
    )
    study.optimize(objective, n_trials=settings.n_trials, show_progress_bar=True)
    return scored
