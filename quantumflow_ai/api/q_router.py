# quantumflow_ai/api/q_router.py

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, ConfigDict
from typing import List
import pandas as pd

from quantumflow_ai.modules.q_routing import optimize_routing, classical_route
from quantumflow_ai.lstm.routing_log import (
    save_log_entry,
    get_last_logs,
    prepare_lstm_input,
    NUMPY_AVAILABLE,
    np,
)
from quantumflow_ai.lstm.lstm_model import load_or_train_model
from quantumflow_ai.core.logger import get_logger

logger = get_logger("QRoutingAPI")
router = APIRouter()

class RoutingInput(BaseModel):
    model_graph: dict
    token_stream: List[dict]
    use_quantum: bool = True
    model_config = ConfigDict(protected_namespaces=())

@router.post("/q-routing")
def route_tokens(data: RoutingInput):
    try:
        method = "qaoa" if data.use_quantum else "classical"
        result = optimize_routing(data.model_graph, data.token_stream) if data.use_quantum else classical_route(data.model_graph, data.token_stream)

        save_log_entry({
            "input_matrix": data.model_graph,
            "output_matrix": result.get("assignments"),
            "energy": result.get("routing_score"),
            "method": method,
        })

        return {
            "status": "success",
            "mode": method,
            "routing_score": result.get("routing_score"),
            "optimized_params": result.get("optimized_params"),
            "assignments": result.get("assignments", []),
            "results": result,
        }
    except Exception as e:
        logger.exception("Routing failed")
        raise HTTPException(status_code=500, detail=f"Routing failed: {str(e)}")

@router.post("/q-routing/file-upload")
def route_from_file(file: UploadFile = File(...), use_quantum: bool = True):
    try:
        df = pd.read_csv(file.file)
        if "token_id" not in df.columns:
            raise ValueError("File must include a 'token_id' column.")
        token_stream = [{"token_id": int(row["token_id"])} for _, row in df.iterrows()]
        expert_columns = [col for col in df.columns if col != "token_id"]
        model_graph = {"experts": list(range(len(expert_columns)))}

        result = optimize_routing(model_graph, token_stream) if use_quantum else classical_route(model_graph, token_stream)

        save_log_entry({
            "input_matrix": model_graph,
            "output_matrix": result.get("assignments"),
            "energy": result.get("routing_score"),
            "method": "qaoa" if use_quantum else "classical",
        })

        return {
            "status": "success",
            "mode": "qaoa" if use_quantum else "classical",
            "routing_score": result.get("routing_score"),
            "optimized_params": result.get("optimized_params"),
            "assignments": result.get("assignments", []),
            "results": result,
        }
    except Exception as e:
        logger.exception("File routing failed")
        raise HTTPException(status_code=400, detail=f"File processing failed: {e}")

@router.get("/q-routing/suggest")
def suggest_optimized_routing():
    try:
        logs = get_last_logs()
        if len(logs) < 2:
            raise HTTPException(status_code=400, detail="Not enough logs to make a prediction.")

        X = prepare_lstm_input(logs)
        if (
            X is None
            or (isinstance(X, list) and len(X) == 0)
            or (hasattr(X, "size") and X.size == 0)
        ):
            return {
                "status": "unavailable",
                "note": "LSTM features unavailable; install numpy to enable predictions",
            }

        y = None
        if NUMPY_AVAILABLE:
            y = np.array([log.get("energy", 0.0) for log in logs[-X.shape[1]:]], dtype=float)
            y = y.reshape(1, -1, 1)

        model = load_or_train_model(X, y)
        prediction = model.predict(X)

        return {
            "status": "success",
            "suggested_energy": float(prediction[0][0]),
            "note": "Prediction based on last 10 routes"
        }

    except Exception as e:
        logger.exception("Suggestion failed")
        raise HTTPException(status_code=500, detail=f"Suggestion failed: {str(e)}")
