from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
import os

MODEL_PATH = "lstm_model.keras"

def build_lstm_model(input_dim, output_dim=1):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(None, input_dim)))
    model.add(LSTM(64))
    model.add(Dense(output_dim))
    model.compile(optimizer='adam', loss='mse')
    return model

def load_or_train_model(X, y=None):
    from tensorflow.keras.models import load_model
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    model = build_lstm_model(X.shape[2])
    if y is not None:
        model.fit(X, y, epochs=50, verbose=0)
        model.save(MODEL_PATH)
    return model
