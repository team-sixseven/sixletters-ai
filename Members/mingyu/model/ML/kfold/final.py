"""Final model (all development patients of its dataset) and the single test evaluation (89 patients).

The final model refits the hyperparameters chosen in cross-validation (no new search) on every development
patient allowed by the method's training filter (705, or the 356 from the original 445 for only445); its
imputer is fitted on the same patients. The Pro-BNP baseline is refitted on the same patients.
Test probabilities are used as they are (no calibration) with the fixed threshold 0.5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ...evaluation.core import binary_metrics
from .cohort import Cohort
from .estimators import fit_model, predict_model
from .experiment import Method
from .imputation import FittedImputer, fit_imputer
from .results import load_info
from .settings import FEATURE_SETS, THRESHOLD, Settings


@dataclass
class FinalMember:
    method: Method
    parameters: dict
    imputer: FittedImputer
    model: object
    categorical: list[str]
    n_train: int

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """x: raw inputs with columns ALL_FEATURES (blanks allowed)."""
        filled = self.imputer.transform(x)[FEATURE_SETS[self.method.features]]
        return predict_model(self.method.model, self.model, filled, self.categorical)


def fit_final(settings: Settings, cohort: Cohort, name: str) -> list[FinalMember]:
    """Refit a method (or every member of a mean ensemble) on all its development patients."""
    info = load_info(settings, name)
    if info.get("combine") == "rank":
        raise NotImplementedError("순위 평균 앙상블은 확률이 아니라서 최종 모델로 쓰지 않습니다")
    member_names = info.get("ensemble", [name])
    members = []
    for member in member_names:
        member_info = load_info(settings, member)
        method = Method(**member_info["method"])
        train = cohort.train_mask(method.train_filter)
        imputer = fit_imputer(cohort.x[train], cohort.categorical)
        x_train = imputer.transform(cohort.x[train])[FEATURE_SETS[method.features]]
        params = member_info["best_parameters"]
        model = fit_model(
            method.model,
            params,
            x_train,
            cohort.y[train],
            cohort.categorical,
            cohort.base_parameters,
            member_info.get("class_weight"),
        )  # same fixed weights as in cross-validation
        members.append(
            FinalMember(method, params, imputer, model, cohort.categorical, int(train.sum()))
        )
    joblib.dump(members, settings.result_dir / name / "final_model.joblib")
    return members


def predict(members: list[FinalMember], x: pd.DataFrame) -> np.ndarray:
    """Average probability of the members (one member for a single method)."""
    return np.mean([m.predict(x) for m in members], axis=0)


def evaluate_on_test(
    settings: Settings,
    cohort: Cohort,
    name: str,
    members: list[FinalMember],
    allow_retest: bool = False,
) -> pd.Series:
    """The single test evaluation of the chosen final model (from fit_final), with two baselines on the
    same 89 patients: Pro-BNP logistic regression (refit on the final model's training patients) and the
    guideline rule LVEF <= 35%."""
    test_dir = settings.result_dir / "test"
    done = sorted(test_dir.glob("*/metrics.json"))
    if done and not allow_retest:
        raise RuntimeError(f"이미 test 평가한 결과가 있습니다: {[p.parent.name for p in done]}")
    train_filters = {m.method.train_filter for m in members}
    if len(train_filters) != 1:
        raise ValueError(f"최종 모델 구성원의 학습 환자가 다릅니다: {train_filters}")
    reference = Method("Logistic", "ProBNP", train_filters.pop()).name

    y = cohort.y_test
    p_model = predict(members, cohort.x_test)
    p_reference = predict(fit_final(settings, cohort, reference), cohort.x_test)
    lvef_rule = (cohort.x_test["LVEF (%)"].to_numpy() <= 35).astype(float)

    rng = np.random.default_rng(0)
    boots = [
        i for i in (rng.integers(0, len(y), len(y)) for _ in range(3000)) if 0 < y[i].sum() < len(i)
    ][:2000]
    model_auroc = np.array([roc_auc_score(y[i], p_model[i]) for i in boots])
    reference_auroc = np.array([roc_auc_score(y[i], p_reference[i]) for i in boots])

    metrics = binary_metrics(y, p_model, THRESHOLD)
    lvef = binary_metrics(y, lvef_rule, 0.5)
    metrics.update(
        auroc_ci=_ci(model_auroc),
        reference=reference,
        reference_auroc=roc_auc_score(y, p_reference),
        delta_auroc_vs_reference=metrics["auroc"] - roc_auc_score(y, p_reference),
        delta_auroc_vs_reference_ci=_ci(model_auroc - reference_auroc),
        lvef35_sensitivity=lvef["sensitivity"],
        lvef35_specificity=lvef["specificity"],
        threshold=THRESHOLD,
        probability="uncalibrated model output",
        model=name,
    )

    folder = test_dir / f"{datetime.now():%Y%m%d-%H%M}_{name}"
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str)
    )
    pd.DataFrame(
        {
            "patient_id": cohort.test_ids,
            "y_true": y,
            "p_positive": p_model,
            "y_pred": (p_model >= THRESHOLD).astype(int),
            f"p_{reference}": p_reference,
            "lvef_le_35": lvef_rule.astype(int),
        }
    ).to_csv(folder / "test_predictions.csv", index=False)
    print("test 평가 저장:", folder)
    keep = [
        "n",
        "scd",
        "auroc",
        "auroc_ci",
        "average_precision",
        "brier",
        "sensitivity",
        "specificity",
        "reference_auroc",
        "delta_auroc_vs_reference",
        "delta_auroc_vs_reference_ci",
        "lvef35_sensitivity",
        "lvef35_specificity",
    ]
    return pd.Series({k: metrics[k] for k in keep}, name="test", dtype=object)


def _ci(values: np.ndarray) -> list[float]:
    return [round(float(v), 3) for v in np.percentile(values, [2.5, 97.5])]
