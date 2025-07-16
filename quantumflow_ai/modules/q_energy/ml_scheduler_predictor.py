import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import joblib
import os
import json

MODEL_PATH = "modules/q_energy/model/ml_energy_predictor.pkl"

class MLEnergyPredictor:
    def __init__(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        else:
            self.model = GradientBoostingRegressor()

    def _is_model_fitted(self) -> bool:
        try:
            check_is_fitted(self.model)
            return True
        except NotFittedError:
            return False

    def train(self, features: list[list[float]], targets: list[float]):
        self.model.fit(features, targets)
        joblib.dump(self.model, MODEL_PATH)

    def predict_energy_cost(self, feature: list[float]) -> float:
        if not self._is_model_fitted():
            return float(np.sum(feature))  # fallback: sum of weighted jobs
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

    def fine_tune_on_new_data(self, dataset_path: str):
        with open(dataset_path) as f:
            data = json.load(f)
        X = [d["features"] for d in data]
        y = [d["cost"] for d in data]
        self.train(X, y)

    def fine_tune_on_profile(
        self, profile_name: str, data_dir: str = "quantumflow_ai/notebooks/profiles"
    ):
        file_path = os.path.join(data_dir, f"{profile_name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Profile data not found: {file_path}")
        with open(file_path, "r") as f:
            data = json.load(f)
        features = [d["features"] for d in data]
        targets = [d["cost"] for d in data]
        self.train(features, targets)
        print(f"[✓] Model fine-tuned on {profile_name} profile")
