from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import joblib
import os
from pathlib import Path

class MetaScheduler:
    def __init__(self):
        self.model_path = Path(__file__).resolve().parent / "model" / "meta_strategy_selector.pkl"
        self.model = self.load_or_initialize()

    def load_or_initialize(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        return RandomForestClassifier(n_estimators=100)

    def _is_model_fitted(self) -> bool:
        try:
            check_is_fitted(self.model)
            return True
        except NotFittedError:
            return False

    def train(self, X: list[list[float]], y: list[str]):
        self.model.fit(X, y)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def recommend(self, feature: list[float]) -> str:
        if not self._is_model_fitted():
            return "classical"
        return self.model.predict([feature])[0]
