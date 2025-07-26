from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from quantumflow_ai.api.q_router import router as q_router
from quantumflow_ai.api.energy import router as energy_router
from quantumflow_ai.api.decompressor import router as decompressor_router
from quantumflow_ai.api.hpo import router as hpo_router  # ✅ NEW: Q-HPO backend

from quantumflow_ai.api.compressor import (
    read_csv_as_array,
    run_compression,
    validate_compression_config,
)

app = FastAPI()

# ✅ CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Include all module routers
app.include_router(q_router)
app.include_router(energy_router)
app.include_router(decompressor_router)
app.include_router(hpo_router)  # ✅ Q-HPO optimizer endpoint

# ✅ Compression API Endpoint
@app.post("/q-compression/upload")
async def compress_upload(
    file: UploadFile = File(...),
    use_quantum: bool = True,
    use_denoiser: bool = False,
    noise: bool = False,
    noise_level: float = 0.0,
    use_dropout: bool = False,
    dropout_prob: float = 0.0,
    enable_pruning: bool = False,
    pruning_threshold: float = 0.01,
    predict_first: bool = False,
    compression_mode: str = "qml",
    predict_compressibility: bool = False,
):
    try:
        data = read_csv_as_array(file)
        config = validate_compression_config(
            {
                "use_quantum": use_quantum,
                "use_denoiser": use_denoiser,
                "noise": noise,
                "noise_level": noise_level,
                "use_dropout": use_dropout,
                "dropout_prob": dropout_prob,
                "enable_pruning": enable_pruning,
                "pruning_threshold": pruning_threshold,
                "predict_first": predict_first,
                "compression_mode": compression_mode,
                "predict_compressibility": predict_compressibility,
            }
        )
        result = run_compression(data, config=config)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

# ✅ Serve frontend (after all API routes)
app.mount("/", StaticFiles(directory="quantumflow_ai/frontend/public", html=True), name="static")
