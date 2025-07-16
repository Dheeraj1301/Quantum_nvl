# modules/q_energy/meta_scheduler.py

from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class MetaScheduler:
    def __init__(self):
        self.model_path = "modules/q_energy/model/meta_strategy_selector.pkl"
        self.model = self.load_or_initialize()

    def load_or_initialize(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        return RandomForestClassifier(n_estimators=100)

    def train(self, X: list[list[float]], y: list[str]):
        self.model.fit(X, y)
        # Ensure the target directory exists before attempting to save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def recommend(self, feature: list[float]) -> str:
        return self.model.predict([feature])[0]
