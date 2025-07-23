import os

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Masking
    TENSORFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Sequential = None
    LSTM = None
    Dense = None
    Masking = None
    load_model = None
    TENSORFLOW_AVAILABLE = False

# Save the model next to this file for stability across runs
MODEL_PATH = os.path.join(os.path.dirname(__file__), "lstm_model.keras")

def build_lstm_model(input_dim: int, output_dim: int = 1):
    """Create a simple LSTM model if TensorFlow is installed."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required to build the LSTM model")
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(None, input_dim)))
    model.add(LSTM(64))
    model.add(Dense(output_dim))
    model.compile(optimizer="adam", loss="mse")
    return model

def load_or_train_model(X, y=None):
    """Load an existing model or train a new one if possible."""
    if not TENSORFLOW_AVAILABLE:
        class Dummy:
            def predict(self, _):
                return [[0.0]]
        return Dummy()

    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)

    model = build_lstm_model(X.shape[2])
    if y is not None:
        model.fit(X, y, epochs=50, verbose=0)
        model.save(MODEL_PATH)
    return model
