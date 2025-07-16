# ml_scheduler_predictor.py

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import joblib
import os

MODEL_PATH = "modules/q_energy/model/ml_energy_predictor.pkl"

class MLEnergyPredictor:
    def __init__(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        else:
            self.model = GradientBoostingRegressor()

    def _is_model_fitted(self) -> bool:
        """Return True if the underlying model has been fitted."""
        try:
            check_is_fitted(self.model)
            return True
        except NotFittedError:
            return False

    def train(self, features: list[list[float]], targets: list[float]):
        self.model.fit(features, targets)
        joblib.dump(self.model, MODEL_PATH)

    def predict_energy_cost(self, feature: list[float]) -> float:
        """Predict energy cost for the provided feature vector.

        If the model hasn't been trained yet, a simple heuristic based on the
        sum of the feature values is used instead. This avoids errors when the
        model file is missing or the model hasn't been fitted.
        """
        if not self._is_model_fitted():
            return float(np.sum(feature))
        return float(self.model.predict([feature])[0])

    def suggest_reschedule(self, schedule: dict, energy_profile: dict) -> dict:
        rescheduled = schedule.copy()
        jobs = list(schedule.keys())
        for job in jobs:
            trial = schedule.copy()
            trial[job] = max(0, schedule[job] - 1)
            feat = self.schedule_to_features(trial, energy_profile)
            cost = self.predict_energy_cost(feat)
            if cost < self.predict_energy_cost(self.schedule_to_features(schedule, energy_profile)):
                rescheduled[job] = trial[job]
        return rescheduled

    def schedule_to_features(self, schedule: dict, energy_profile: dict) -> list[float]:
        return [schedule[j] * energy_profile[j] for j in schedule]
    def fine_tune_on_new_data(self, dataset_path):
        import json
        with open(dataset_path) as f:
            data = json.load(f)
        X = [d["features"] for d in data]
        y = [d["cost"] for d in data]
        self.train(X, y)
