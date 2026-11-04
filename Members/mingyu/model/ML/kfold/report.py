"""Comparison tables of OOF results on the same patients.

population="all"      : all 705 development patients
population="original" : the 356 development patients from the original 445 (same population as test);
                        use this to compare training on 705 vs on the original 445 only, and for every
                        only445 method (it has predictions for these 356 patients only).

Each metric is computed per repeat and averaged (± SD over the 3 repeats). Δ columns are paired bootstrap
differences against the reference method: patients are resampled together for both methods.
Threshold-dependent columns use the fixed threshold 0.5 on the probability averaged over repeats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .cohort import Cohort
from .experiment import Method
from .results import evaluated_mask, list_results, load_info, load_oof, result_exists
from .settings import THRESHOLD, Settings

METRICS = {"AUROC": roc_auc_score, "AP": average_precision_score}


def compare(
    settings: Settings,
    cohort: Cohort,
    names: list[str] | None = None,
    reference: str = "LR_ProBNP_iterative",
    population: str = "all",
    save_as: str | None = None,
) -> pd.DataFrame:
    """One row per method, best AUROC first."""
    names = list(names or list_results(settings))
    if reference not in names:
        names = [reference, *names]
    patients = _population(cohort, population)
    oof = {name: load_oof(settings, cohort, name) for name in names}
    missing = [name for name, p in oof.items() if not evaluated_mask(p)[patients].all()]
    if missing:
        raise ValueError(
            f"평가 환자 중 예측이 없는 방법이 있습니다: {missing}. "
            "only445 방법은 population='original'(445명 출신 356명)로 비교하세요"
        )
    y = cohort.y[patients]
    boots = _bootstrap_samples(y, settings.n_boot)
    p_ref = oof[reference][patients]
    ref_boot = {
        m: np.array([_mean_over_repeats(f, y[i], p_ref[i]) for i in boots])
        for m, f in METRICS.items()
    }

    rows = []
    for name in names:
        p = oof[name][patients]
        is_probability = load_info(settings, name).get("is_probability", True)
        row = {"방법": name}
        for metric, function in METRICS.items():
            values = [function(y, p[:, r]) for r in range(p.shape[1])]
            row[metric] = f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"
            row[f"_{metric}"] = np.mean(values)
        if is_probability:
            row["Brier"] = round(_mean_over_repeats(brier_score_loss, y, p), 3)
            row["log loss"] = round(
                _mean_over_repeats(lambda t, s: log_loss(t, s, labels=[0, 1]), y, p), 3
            )
            predicted = p.mean(axis=1) >= THRESHOLD
            row["민감도@0.5"] = round(predicted[y == 1].mean(), 3)
            row["특이도@0.5"] = round((~predicted)[y == 0].mean(), 3)
        if population == "all":
            original = cohort.original
            row["AUROC(445출신)"] = round(
                _mean_over_repeats(roc_auc_score, cohort.y[original], oof[name][original]), 3
            )
        for metric, function in METRICS.items():
            if name == reference:
                row[f"Δ{metric} (95% CI)"] = "기준"
                continue
            diff = (
                np.array([_mean_over_repeats(function, y[i], p[i]) for i in boots])
                - ref_boot[metric]
            )
            low, high = np.percentile(diff, [2.5, 97.5])
            row[f"Δ{metric} (95% CI)"] = (
                f"{row[f'_{metric}'] - _mean_over_repeats(function, y, p_ref):+.3f} ({low:+.3f} ~ {high:+.3f})"
            )
        rows.append(row)

    table = (
        pd.DataFrame(rows)
        .sort_values("_AUROC", ascending=False)
        .drop(columns=["_AUROC", "_AP"])
        .reset_index(drop=True)
    )
    label = "705명 전체" if population == "all" else "445명 출신 356명"
    print(
        f"평가 환자: {label} ({len(y)}명, SCD {y.sum()}), 반복 {p_ref.shape[1]}번, 기준: {reference}"
    )
    if save_as:
        table.to_csv(settings.result_dir / f"comparison_{save_as}_{population}.csv", index=False)
    return table


def describe(settings: Settings, cohort: Cohort, names: list[str]) -> pd.DataFrame:
    """What each method chose: training size, selected hyperparameters, and OOF log loss and AUROC on the
    method's own evaluation patients (705, or 356 for only445)."""
    rows = []
    for name in names:
        info = load_info(settings, name)
        p = load_oof(settings, cohort, name)
        evaluated = evaluated_mask(p)
        rows.append(
            {
                "방법": name,
                "fold당 학습 환자": "~".join(
                    str(n) for n in (min(info["n_train_per_fold"]), max(info["n_train_per_fold"]))
                )
                if len(info["n_train_per_fold"]) > 1
                else info["n_train_per_fold"][0],
                "평가 환자": int(evaluated.sum()),
                "선택된 설정": ", ".join(
                    f"{k}={_short(v)}" for k, v in info["best_parameters"].items()
                ),
                "class weight": info.get("class_weight") or "없음(튜닝)",
                "튜닝 log loss": round(
                    info["best_log_loss"], 4
                ),  # weighted like training when class weight is fixed
                "AUROC": round(
                    _mean_over_repeats(roc_auc_score, cohort.y[evaluated], p[evaluated]), 3
                ),
                "소요(분)": info["minutes"],
            }
        )
    return pd.DataFrame(rows)


def _short(value) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def auroc_grid(
    settings: Settings,
    cohort: Cohort,
    models: list[str],
    feature_sets: list[str],
    train_filter: str = "all",
) -> pd.DataFrame:
    """Overview: mean OOF AUROC on the evaluation patients of train_filter (705, or 356 for only445),
    models as rows and feature sets as columns."""
    evaluated = cohort.eval_mask(train_filter)
    grid = pd.DataFrame(index=models, columns=feature_sets, dtype=float)
    for model in models:
        for features in feature_sets:
            name = Method(model, features, train_filter).name
            if result_exists(settings, name):
                p = load_oof(settings, cohort, name)
                grid.loc[model, features] = _mean_over_repeats(
                    roc_auc_score, cohort.y[evaluated], p[evaluated]
                )
    return grid.round(3)


def _population(cohort: Cohort, population: str) -> np.ndarray:
    if population == "all":
        return np.arange(len(cohort.ids))
    if population == "original":
        return np.flatnonzero(cohort.original)
    raise ValueError("population은 all 또는 original입니다")


def _bootstrap_samples(y: np.ndarray, n_boot: int) -> list[np.ndarray]:
    """Resampled patient positions that contain both classes."""
    rng = np.random.default_rng(0)
    samples = []
    while len(samples) < n_boot:
        index = rng.integers(0, len(y), len(y))
        if 0 < y[index].sum() < len(index):
            samples.append(index)
    return samples


def _mean_over_repeats(function, y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([function(y, p[:, r]) for r in range(p.shape[1])]))
