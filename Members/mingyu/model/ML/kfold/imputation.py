"""Iterative imputation fitted on training patients only.

Every feature set is imputed from all 27 inputs (ALL_FEATURES), so a 5-variable model still borrows
information from LVEF, NYHA, age, ... when filling its blanks. The outcome (SCD) is never an input,
because held-out and new patients are imputed without knowing it.

- 0/1 and categorical columns: most frequent training value.
- Continuous columns: IterativeImputer (Bayesian ridge, 10 rounds) on log(x + 1) for skewed columns.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from .settings import ALL_FEATURES, LOG_COLUMNS


@dataclass
class FittedImputer:
    modes: dict[str, float]  # discrete column -> fill value
    continuous: list[str]
    imputer: IterativeImputer

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Fill the blanks of x (columns = ALL_FEATURES); observed values are kept as they are."""
        out = x[ALL_FEATURES].fillna(self.modes)
        filled = pd.DataFrame(
            self.imputer.transform(_to_log(out)), columns=ALL_FEATURES, index=out.index
        )
        filled = _from_log(filled)
        out[self.continuous] = out[self.continuous].fillna(filled[self.continuous])
        return out


def fit_imputer(x_train: pd.DataFrame, categorical: list[str]) -> FittedImputer:
    x_train = x_train[ALL_FEATURES]
    discrete = [
        c
        for c in ALL_FEATURES
        if c in categorical or set(x_train[c].dropna().unique()) <= {0.0, 1.0}
    ]
    modes = {c: x_train[c].mode().iloc[0] for c in discrete}
    imputer = IterativeImputer(max_iter=10, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", ConvergenceWarning
        )  # 10 rounds is enough; the tolerance is rarely met
        imputer.fit(_to_log(x_train.fillna(modes)))
    return FittedImputer(
        modes=modes, continuous=[c for c in ALL_FEATURES if c not in discrete], imputer=imputer
    )


def _to_log(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame[LOG_COLUMNS] = np.log1p(frame[LOG_COLUMNS])
    return frame


def _from_log(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame[LOG_COLUMNS] = np.expm1(frame[LOG_COLUMNS])
    return frame
