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

    def _prepare_features(self, features: list[list[float]] | list[float]) -> list[list[float]] | list[float]:
        """Pad or truncate features to match the trained model's expected dimension."""
        if isinstance(features[0], list):
            # Training data: list of samples
            expected = max(len(f) for f in features)
            return [f + [0.0] * (expected - len(f)) for f in features]
        else:
            # Single feature vector for prediction
            expected = getattr(self.model, "n_features_in_", len(features))
            if len(features) < expected:
                return features + [0.0] * (expected - len(features))
            return features[:expected]

    def train(self, X: list[list[float]], y: list[str]):
        X_prep = self._prepare_features(X)
        self.model.fit(X_prep, y)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def recommend(self, feature: list[float]) -> str:
        if not self._is_model_fitted():
            return "classical"
        prepared = self._prepare_features(feature)
        return self.model.predict([prepared])[0]
