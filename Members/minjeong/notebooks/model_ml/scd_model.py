"""SCD risk predictor (logistic regression, 12 inputs). Pure Python: no numpy / sklearn needed.

    from scd_model import SCDModel
    model = SCDModel("model.json")
    model.predict({"Age": 65, "Gender (male=1)": 1, ...})
    # {'probability': 0.62, 'high_risk': True, 'threshold': 0.5}
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


class SCDModel:
    def __init__(self, path: str | Path = HERE / "model.json"):
        self.spec = json.loads(Path(path).read_text(encoding="utf-8"))
        self.features: list[str] = self.spec["features"]
        self.threshold: float = self.spec["threshold"]

    def _validate(self, x: dict) -> None:
        missing = [f for f in self.features if x.get(f) is None or (isinstance(x[f], float) and math.isnan(x[f]))]
        if missing:
            raise ValueError(f"missing inputs: {missing}")
        for f in self.spec["log1p_features"]:
            if float(x[f]) < 0:
                raise ValueError(f"{f} must be >= 0")

    def predict_proba(self, x: dict) -> float:
        """Probability of SCD for one patient (dict keyed by the 12 feature names)."""
        self._validate(x)
        z = self.spec["intercept"]
        for t in self.spec["terms"]:
            v = float(x[t["feature"]])
            if t["log1p"]:
                v = math.log1p(v)
            z += t["coef"] * (v - t["mean"]) / t["scale"]
        return 1.0 / (1.0 + math.exp(-z))

    def predict(self, x: dict) -> dict:
        p = self.predict_proba(x)
        return {"probability": p, "high_risk": p >= self.threshold, "threshold": self.threshold}

    def contributions(self, x: dict) -> dict[str, float]:
        """Interventional local SHAP values in log-odds units for this linear model.

        The reference is the training mean of each transformed feature. These
        values add to the intercept to reconstruct the prediction's log-odds.
        """
        self._validate(x)
        out = {}
        for t in self.spec["terms"]:
            v = float(x[t["feature"]])
            v = math.log1p(v) if t["log1p"] else v
            out[t["feature"]] = t["coef"] * (v - t["mean"]) / t["scale"]
        return out

    def predict_explained(self, x: dict, top_k: int = 3) -> dict:
        """Predict one patient and return local SHAP values and top positive factors."""
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        result = self.predict(x)
        shap_values = self.contributions(x)
        ranked = sorted(shap_values.items(), key=lambda item: item[1], reverse=True)
        result["contributions"] = shap_values  # Keep the existing API field.
        result["shap"] = {
            "base_value_log_odds": self.spec["intercept"],
            "values_log_odds": shap_values,
        }
        result["risk_factors"] = [
            {
                "feature": feature,
                "input_value": x[feature],
                "shap_value_log_odds": value,
            }
            for feature, value in ranked
            if value > 0
        ][:top_k]
        return result
