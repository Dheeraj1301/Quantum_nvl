from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from quantumflow_ai.api.q_router import router as q_router
from quantumflow_ai.api.energy import router as energy_router
from quantumflow_ai.api.decompressor import router as decompressor_router
from quantumflow_ai.api.hpo import router as hpo_router  # ✅ NEW: Q-HPO backend
from quantumflow_ai.api.compressor import router as compressor_router

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
app.include_router(compressor_router)

# ✅ Serve frontend (after all API routes)
app.mount("/", StaticFiles(directory="quantumflow_ai/frontend/public", html=True), name="static")
