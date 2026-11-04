"""Figures saved after every training run.

- Predicted-probability histogram: x = OOF probability of SCD (mean of the 3 repeats per patient),
  y = number of patients, colour = actual outcome (y_true), dashed line = threshold 0.5.
  Only the method's evaluation patients are drawn (705, or the 356 from the original 445 for only445).
- Loss curve: train and validation (held-out fold) log loss of the selected setting,
  mean over the 15 folds with the min-max range shaded; the x-axis depends on the model (see curves.py).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_auc_score

from .cohort import Cohort
from .curves import STEP_LABEL
from .results import evaluated_mask, load_oof
from .settings import THRESHOLD, Settings

# Colours validated for colour-vision deficiency (blue / orange); text stays in neutral ink.
BLUE, ORANGE = "#2a78d6", "#eb6834"
OUTCOME_STYLE = {
    0: {"color": BLUE, "label": "SCD 없음 (y_true = 0)"},
    1: {"color": ORANGE, "label": "SCD (y_true = 1)"},
}
LOSS_STYLE = {
    "train_loss": {"color": BLUE, "label": "train (학습 fold)"},
    "validation_loss": {"color": ORANGE, "label": "validation (평가 fold)"},
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BINS = np.linspace(0, 1, 41)  # 0.025 wide


def save_loss_curve(
    settings: Settings, name: str, model: str, curves: pd.DataFrame, chosen_c: float | None
) -> Path:
    """Called after every training run: <result_dir>/<name>/loss_curve.png (+ loss_curve.csv)."""
    _use_korean_font()
    folder = settings.result_dir / name
    curves.to_csv(folder / "loss_curve.csv", index=False)
    summary = curves.groupby("step")[list(LOSS_STYLE)].agg(["mean", "min", "max"])
    steps = summary.index.to_numpy()

    fig, ax = plt.subplots(figsize=(6.4, 4))
    marker = "o" if model == "Logistic" else None  # only 9 C values: show each point
    for column, style in LOSS_STYLE.items():
        ax.fill_between(
            steps,
            summary[column]["min"],
            summary[column]["max"],
            color=style["color"],
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(
            steps,
            summary[column]["mean"],
            color=style["color"],
            linewidth=2,
            marker=marker,
            markersize=5,
            label=f"{style['label']}, {len(curves.groupby(['repeat', 'fold']))}개 fold 평균",
        )
    if chosen_c is not None:
        ax.axvline(
            chosen_c,
            color=MUTED,
            linestyle="--",
            linewidth=1.2,
            label=f"선택된 C ({_short(chosen_c)})",
        )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    else:  # trees: the selected setting is the whole curve; mark where validation loss was lowest
        best = summary["validation_loss"]["mean"].idxmin()
        ax.axvline(
            best,
            color=MUTED,
            linestyle=":",
            linewidth=1.2,
            label=f"validation 최소 ({_short(best)})",
        )
    ax.set_title(
        f"{name}\n선택된 설정의 train / validation loss (음영: fold 간 최소~최대)",
        fontsize=10,
        color=INK,
    )
    ax.set_xlabel(STEP_LABEL[model], color=MUTED)
    ax.set_ylabel("log loss", color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    _style_axes(ax, x_range=None)
    fig.tight_layout()
    path = folder / "loss_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _short(value: float) -> str:
    return f"{value:.4g}"


def save_probability_hist(settings: Settings, cohort: Cohort, name: str) -> Path:
    """Called after every training run: <result_dir>/<name>/probability_hist.png."""
    _use_korean_font()
    fig, ax = plt.subplots(figsize=(6.4, 4))
    _draw_hist(ax, *_evaluated(settings, cohort, name), name)
    ax.set_xlabel("예측 확률 (SCD), 3반복 평균", color=MUTED)
    ax.set_ylabel("환자 수", color=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    path = settings.result_dir / name / "probability_hist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_probability_hist(
    settings: Settings, cohort: Cohort, names: list[str], ncols: int = 4, save_as: str | None = None
) -> plt.Figure:
    """Several methods side by side; panels share both axes so they can be compared directly."""
    _use_korean_font()
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(
        nrows,
        min(ncols, len(names)),
        figsize=(4 * min(ncols, len(names)), 3.2 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    sizes = set()
    for ax, name in zip(axes.flat, names, strict=False):
        y, p = _evaluated(settings, cohort, name)
        sizes.add(len(y))
        _draw_hist(ax, y, p, name)
    for ax in list(axes.flat)[len(names) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("예측 확률 (SCD)", color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("환자 수", color=MUTED)
    axes.flat[0].legend(frameon=False, fontsize=8.5, labelcolor=INK)
    population = {
        len(cohort.y): f"개발 {len(cohort.y)}명",
        int(cohort.original.sum()): f"원래 445명 출신 {int(cohort.original.sum())}명",
    }
    label = (
        population.get(next(iter(sizes)), f"{next(iter(sizes))}명")
        if len(sizes) == 1
        else "방법별 평가 환자"
    )
    fig.suptitle(f"{label} 교차검증 예측 확률 분포 (3반복 평균)", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    if save_as:
        path = settings.result_dir / f"probability_hist_{save_as}.png"
        fig.savefig(path, dpi=150)
        print("저장:", path)
    return fig


def _evaluated(settings: Settings, cohort: Cohort, name: str) -> tuple[np.ndarray, np.ndarray]:
    """y and the repeat-averaged OOF probability of the patients the method was evaluated on."""
    p = load_oof(settings, cohort, name)
    keep = evaluated_mask(p)
    return cohort.y[keep], p[keep].mean(axis=1)


def _draw_hist(ax: plt.Axes, y: np.ndarray, p: np.ndarray, name: str) -> None:
    for outcome in (0, 1):
        values = p[y == outcome]
        style = OUTCOME_STYLE[outcome]
        ax.hist(
            values,
            bins=BINS,
            color=style["color"],
            alpha=0.6,
            edgecolor="white",
            linewidth=0.8,
            label=f"{style['label']}, n = {len(values)}",
        )
    ax.axvline(
        THRESHOLD, color=MUTED, linestyle="--", linewidth=1.2, label=f"threshold {THRESHOLD}"
    )
    ax.set_title(f"{name}\nAUROC {roc_auc_score(y, p):.3f}", fontsize=10, color=INK)
    _style_axes(ax)


def _style_axes(ax: plt.Axes, x_range: tuple[float, float] | None = (0, 1)) -> None:
    if x_range:
        ax.set_xlim(*x_range)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _use_korean_font() -> None:
    for font in Path("/usr/share/fonts/truetype/nanum").glob("NanumGothic*.ttf"):
        font_manager.fontManager.addfont(str(font))
    if any(f.name == "NanumGothic" for f in font_manager.fontManager.ttflist):
        plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False})
