"""Save the final model in its library's native format, and load it back for the app."""
import json
from pathlib import Path

import lightgbm
import numpy as np
import xgboost

from .evaluate import prediction_interval
from .features import time_band

MODELS = Path(__file__).resolve().parents[2] / "models"


def _plain(value):
    return value.item() if hasattr(value, "item") else str(value)


def save(model, card: dict, directory=MODELS):
    """Save a fitted `models.sqrt_target(...)` model plus a JSON model card."""
    directory.mkdir(parents=True, exist_ok=True)
    if card["library"] == "lightgbm":
        model.regressor_.booster_.save_model(str(directory / "model.txt"))
    else:
        model.regressor_.save_model(directory / "model.json")
    (directory / "model_card.json").write_text(json.dumps(card, indent=2, default=_plain))


class Forecaster:
    """The saved model: forecasts in bikes, 90% intervals, and the booster for SHAP."""

    def __init__(self, directory=MODELS):
        self.card = json.loads((directory / "model_card.json").read_text())
        self.features = self.card["features"]
        if self.card["library"] == "lightgbm":
            self.booster = lightgbm.Booster(model_file=str(directory / "model.txt"))
        else:
            self.booster = xgboost.XGBRegressor()
            self.booster.load_model(directory / "model.json")

    def predict(self, X):
        root = np.asarray(self.booster.predict(X[self.features]), dtype=float)
        return np.square(np.clip(root, 0, None))

    def interval(self, X):
        q = time_band(X).map(self.card["conformal_q_by_band"]).to_numpy()
        return prediction_interval(self.predict(X), q)
