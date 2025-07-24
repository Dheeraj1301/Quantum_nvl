from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from quantumflow_ai.api.q_router import router as q_router
from quantumflow_ai.api.energy import router as energy_router
from quantumflow_ai.api.compressor import read_csv_as_array, run_compression

app = FastAPI()

# CORS Middleware for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Add routers
app.include_router(q_router)
app.include_router(energy_router)

# ✅ New: Q-Compression Upload Endpoint
@app.post("/q-compression/upload")
async def compress_upload(file: UploadFile = File(...), use_quantum: bool = True):
    try:
        data = read_csv_as_array(file)
        result = run_compression(data, use_quantum=use_quantum)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# ✅ Serve frontend (mounted last so API routes take precedence)
app.mount("/", StaticFiles(directory="quantumflow_ai/frontend/public", html=True), name="static")
