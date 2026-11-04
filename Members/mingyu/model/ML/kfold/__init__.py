"""Clinical ML with repeated stratified k-fold cross-validation (rules in settings.py).

cohort = load_cohort()
save_or_check_folds(cohort, Settings())
run_method(cohort, Method("Logistic", "MUSIC_SCD"))
compare(Settings(), cohort, population="original")
"""

from .cohort import Cohort, load_cohort, save_or_check_folds
from .experiment import Method, run_method
from .final import evaluate_on_test, fit_final, predict
from .plots import plot_probability_hist
from .report import auroc_grid, compare, describe
from .results import list_results, load_info, load_oof, make_ensemble
from .settings import FEATURE_SETS, MODELS, TRAIN_FILTERS, Settings

__all__ = [
    "Cohort",
    "load_cohort",
    "save_or_check_folds",
    "Method",
    "run_method",
    "auroc_grid",
    "compare",
    "describe",
    "plot_probability_hist",
    "list_results",
    "load_info",
    "load_oof",
    "make_ensemble",
    "fit_final",
    "predict",
    "evaluate_on_test",
    "FEATURE_SETS",
    "MODELS",
    "TRAIN_FILTERS",
    "Settings",
]
