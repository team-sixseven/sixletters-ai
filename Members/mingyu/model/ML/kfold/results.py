"""Result folders: one per method or ensemble, under Settings.result_dir.

<result_dir>/<name>/oof.csv     patient_id, repeat, fold, y_true, p_positive (evaluation patients x 3 repeats:
                                705, or the 356 from the original 445 for only445)
<result_dir>/<name>/info.json   what was run and the chosen hyperparameters
<result_dir>/<name>/trials.csv  every candidate setting and its OOF log loss (methods only)
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .cohort import Cohort
from .settings import Settings


def result_exists(settings: Settings, name: str) -> bool:
    return (settings.result_dir / name / "oof.csv").exists()


def list_results(settings: Settings) -> list[str]:
    return sorted(path.parent.name for path in settings.result_dir.glob("*/oof.csv"))


def save_result(
    settings: Settings,
    cohort: Cohort,
    name: str,
    p: np.ndarray,
    trials: pd.DataFrame | None,
    info: dict,
) -> None:
    """p: OOF probabilities, shape (patients, repeats), rows in cohort.ids order; NaN rows are not saved."""
    folder = settings.result_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    rows = [
        pd.DataFrame(
            {
                "patient_id": cohort.ids[f.eval_index],
                "repeat": f.repeat,
                "fold": f.fold,
                "y_true": cohort.y[f.eval_index],
                "p_positive": p[f.eval_index, f.repeat],
            }
        )
        for f in cohort.folds
    ]
    oof = pd.concat(rows).dropna(subset=["p_positive"]).sort_values(["repeat", "patient_id"])
    oof.to_csv(folder / "oof.csv", index=False)
    if trials is not None:
        trials.to_csv(folder / "trials.csv", index=False)
    info = {
        "name": name,
        "created": datetime.now().isoformat(timespec="minutes"),
        "data_md5": cohort.data_md5,
        **info,
    }
    (folder / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str))


def load_info(settings: Settings, name: str) -> dict:
    return json.loads((settings.result_dir / name / "info.json").read_text())


def load_oof(settings: Settings, cohort: Cohort, name: str) -> np.ndarray:
    """OOF probabilities, shape (patients, repeats), rows in cohort.ids order; NaN for patients the method
    was not evaluated on (see evaluated_mask)."""
    frame = pd.read_csv(settings.result_dir / name / "oof.csv", dtype={"patient_id": str})
    p = frame.pivot(index="patient_id", columns="repeat", values="p_positive")
    if not set(p.index) <= set(cohort.ids):
        raise ValueError(f"{name}: 환자 목록이 현재 데이터와 다릅니다")
    p = p.reindex(cohort.ids).to_numpy()
    if np.isnan(p).any(axis=1).sum() != np.isnan(p).all(axis=1).sum():
        raise ValueError(f"{name}: 일부 반복에만 예측이 있는 환자가 있습니다")
    return p


def evaluated_mask(p: np.ndarray) -> np.ndarray:
    """Patients with OOF predictions (rows of load_oof that are not NaN)."""
    return ~np.isnan(p).any(axis=1)


def make_ensemble(
    settings: Settings, cohort: Cohort, members: list[str], method: str = "mean"
) -> str:
    """Combine member OOF predictions (no new training; all members share the same folds).

    mean: average probability.
    rank: average percentile rank within each repeat, so members count equally whatever their probability
          scale. The result is a score from 0 to 1, not a probability.
    """
    if method not in ("mean", "rank"):
        raise ValueError(f"알 수 없는 앙상블 방법: {method} (mean 또는 rank)")
    name = "ENS_" + ("rank_" if method == "rank" else "") + "+".join(members)
    scores = [load_oof(settings, cohort, m) for m in members]
    evaluated = evaluated_mask(scores[0])
    if any(not np.array_equal(evaluated_mask(s), evaluated) for s in scores):
        raise ValueError("앙상블 구성원의 평가 환자가 다릅니다 (예: 705명 학습과 only445를 섞음)")
    if method == "rank":
        scores = [
            np.column_stack([_percentile_rank(s[:, r]) for r in range(s.shape[1])]) for s in scores
        ]
    save_result(
        settings,
        cohort,
        name,
        np.mean(scores, axis=0),
        None,
        info={"ensemble": members, "combine": method, "is_probability": method == "mean"},
    )
    if method == "mean":
        from .plots import (
            save_probability_hist,
        )  # plots imports results; imported here to avoid a cycle

        save_probability_hist(settings, cohort, name)
    print(f"{name}: 저장")
    return name


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """Rank / n among the evaluated patients; NaN stays NaN."""
    out = np.full(len(values), np.nan)
    present = ~np.isnan(values)
    out[present] = rankdata(values[present]) / present.sum()
    return out
