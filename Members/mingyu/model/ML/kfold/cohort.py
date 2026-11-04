"""Development (705) and test (89) patients, their raw inputs, and the fixed cross-validation folds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from ...config import load_experiment_config, load_yaml, resolve_config_path
from ...data import load_cohorts
from .settings import ALL_FEATURES, CHOICE_PATH, PATIENT_FLAGS_FILE, RAW_CLINICAL_FILE, Settings


@dataclass
class Fold:
    repeat: int
    fold: int
    train_index: np.ndarray  # positions in Cohort.ids
    eval_index: np.ndarray


@dataclass
class Cohort:
    ids: np.ndarray  # development patient IDs, sorted
    y: np.ndarray
    x: pd.DataFrame  # raw inputs (blanks kept), columns = ALL_FEATURES
    flags: pd.DataFrame  # added / device / holter_available / any_missing
    test_ids: np.ndarray
    y_test: np.ndarray
    x_test: pd.DataFrame
    categorical: list[str]  # categorical inputs as declared in basic.yaml
    base_parameters: dict[str, dict]  # team parameter YAML "model" section per tree model
    data_md5: dict[str, str]
    folds: list[Fold]

    @property
    def original(self) -> np.ndarray:
        """True for development patients from the original 445 (same population as test)."""
        return self.flags.added.eq(0).to_numpy()

    def train_mask(self, train_filter: str) -> np.ndarray:
        """Patients allowed in training."""
        return {
            "all": np.ones(len(self.ids), bool),
            "only445": self.original,
            "no_device": self.flags.device.eq(0).to_numpy(),
        }[train_filter]

    def eval_mask(self, train_filter: str) -> np.ndarray:
        """Patients a method is tuned and evaluated on (its OOF predictions exist only for them).

        only445 is a dataset of its own: trained, tuned and evaluated within the original 445 (356 patients).
        no_device only removes patients from training, so it is still evaluated on everyone.
        """
        return self.original if train_filter == "only445" else np.ones(len(self.ids), bool)


def load_cohort(settings: Settings = Settings()) -> Cohort:  # noqa: B008
    choice, basic, _, basic_path = load_experiment_config(CHOICE_PATH)
    _, cohorts, _, _ = load_cohorts(basic, basic_path)
    development = pd.concat([cohorts["ml"]["train"], cohorts["ml"]["validation"]]).sort_values(
        "patient_id"
    )
    test = cohorts["ml"]["test"].sort_values("patient_id")
    if set(development.patient_id) & set(test.patient_id):
        raise ValueError("test 환자가 개발 데이터에 섞였습니다")

    drive = Path(basic["drive_root"])
    raw = pd.read_csv(drive / RAW_CLINICAL_FILE, dtype={"Patient ID": str})
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.set_index("Patient ID")
    flags = pd.read_csv(drive / PATIENT_FLAGS_FILE, dtype={"patient_id": str}).set_index(
        "patient_id"
    )

    ids = development.patient_id.astype(str).to_numpy()
    test_ids = test.patient_id.astype(str).to_numpy()
    y = development.y_true.astype(int).to_numpy()
    if not np.array_equal(raw.loc[ids, "SCD"].to_numpy(), y):
        raise ValueError("빈칸 파일과 팀 로더의 SCD 라벨이 다릅니다")

    x_test = raw.loc[test_ids, ALL_FEATURES].astype(float).reset_index(drop=True)
    if x_test.isna().any().any():
        raise ValueError("test 환자에 빈칸이 있습니다")

    data_files = [
        drive / RAW_CLINICAL_FILE,
        drive / PATIENT_FLAGS_FILE,
        Path(basic["data"]["split"]["path"]),
    ]
    return Cohort(
        ids=ids,
        y=y,
        x=raw.loc[ids, ALL_FEATURES].astype(float).reset_index(drop=True),
        flags=flags.loc[ids].reset_index(drop=True),
        test_ids=test_ids,
        y_test=test.y_true.astype(int).to_numpy(),
        x_test=x_test,
        categorical=list(basic["data"]["clinical"]["categorical_features"]),
        base_parameters={
            name: load_yaml(resolve_config_path(path, CHOICE_PATH))["model"]
            for name, path in choice["model_configs"].items()
        },
        data_md5={path.name: hashlib.md5(path.read_bytes()).hexdigest() for path in data_files},
        folds=_make_folds(y, settings),
    )


def _make_folds(y: np.ndarray, settings: Settings) -> list[Fold]:
    splitter = RepeatedStratifiedKFold(
        n_splits=settings.n_splits, n_repeats=settings.n_repeats, random_state=settings.split_seed
    )
    return [
        Fold(
            repeat=task // settings.n_splits,
            fold=task % settings.n_splits,
            train_index=train,
            eval_index=held,
        )
        for task, (train, held) in enumerate(splitter.split(np.zeros(len(y)), y))
    ]


def save_or_check_folds(cohort: Cohort, settings: Settings) -> None:
    """Write the fold assignment once; refuse to continue if data or fold settings changed since."""
    folds = (
        pd.concat(
            [
                pd.DataFrame(
                    {"patient_id": cohort.ids[f.eval_index], "repeat": f.repeat, "fold": f.fold}
                )
                for f in cohort.folds
            ]
        )
        .sort_values(["repeat", "patient_id"])
        .reset_index(drop=True)
    )
    info = {
        "data_md5": cohort.data_md5,
        "patients": len(cohort.ids),
        "scd": int(cohort.y.sum()),
        "n_splits": settings.n_splits,
        "n_repeats": settings.n_repeats,
        "split_seed": settings.split_seed,
    }

    settings.result_dir.mkdir(parents=True, exist_ok=True)
    fold_path, info_path = (
        settings.result_dir / "folds.csv",
        settings.result_dir / "folds_info.json",
    )
    if not fold_path.exists():
        folds.to_csv(fold_path, index=False)
        info_path.write_text(json.dumps(info, indent=2))
        return
    saved = pd.read_csv(fold_path, dtype={"patient_id": str})
    if json.loads(info_path.read_text()) != info or not saved.equals(folds):
        raise ValueError(
            f"{settings.result_dir}에 저장된 fold와 다릅니다. "
            "데이터나 fold 설정을 바꿨다면 Settings.run_tag를 새로 정하세요"
        )
