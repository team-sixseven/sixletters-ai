"""Fixed-hyperparameter clinical logistic reference on the existing five patient folds."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main(args):
    d = json.loads(args.input.read_text())
    rows = d["patients"]
    cols = d["metadata"]["features"]
    n = len(rows)
    if n != 356 or len({r["patient_id"] for r in rows}) != 356:
        raise ValueError("cohort mismatch")
    X = np.array([[r["clinical"][k] for k in cols] for r in rows])
    y = np.array([r["label"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    out = np.full(n, np.nan)
    for f in range(1, 6):
        train = fold != f
        test = fold == f
        if np.any(train & test):
            raise ValueError("fold overlap")
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=42)
        )
        model.fit(X[train], y[train])
        out[test] = model.predict_proba(X[test])[:, 1]
    if not np.isfinite(out).all():
        raise ValueError("missing OOF prediction")
    tn, fp, fn, tp = confusion_matrix(y, out >= 0.5, labels=[0, 1]).ravel().tolist()
    metrics = {
        "n": n,
        "positive": int(y.sum()),
        "model": "StandardScaler + LogisticRegression C=1, no hyperparameter search",
        "auroc": float(roc_auc_score(y, out)),
        "ap": float(average_precision_score(y, out)),
        "brier": float(brier_score_loss(y, out)),
        "confusion": [[tn, fp], [fn, tp]],
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "fold_auroc": [float(roc_auc_score(y[fold == f], out[fold == f])) for f in range(1, 6)],
        "caveat": "same development folds; not a final independent comparison; 23 clinical variables and preprocessing differ from prior baselines",
    }
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    import pandas as pd

    frame = pd.DataFrame(
        {
            "patient_id": [r["patient_id"] for r in rows],
            "fold": fold,
            "y_true": y,
            "p_positive": out,
        }
    )
    pred = args.output / "predictions.csv"
    frame.to_csv(pred, index=False)
    pred.chmod(0o600)
    p = args.output / "metrics.json"
    p.write_text(json.dumps(metrics, indent=2))
    p.chmod(0o600)
    print(json.dumps(metrics))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
