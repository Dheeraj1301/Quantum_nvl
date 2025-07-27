from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from quantumflow_ai.api.q_router import router as q_router
from quantumflow_ai.api.energy import router as energy_router
from quantumflow_ai.api.decompressor import router as decompressor_router
from quantumflow_ai.api.hpo import router as hpo_router
from quantumflow_ai.api.compressor import router as compressor_router
from quantumflow_ai.api.nvlinkopt import router as nvlinkopt_router  # ✅ Q-NVLinkOpt module
from quantumflow_ai.api.attention import router as q_attention_router
app = FastAPI()

# ✅ CORS Middleware for cross-domain frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Register API routers for all modules
app.include_router(q_router)
app.include_router(energy_router)
app.include_router(decompressor_router)
app.include_router(hpo_router)
app.include_router(compressor_router)
app.include_router(nvlinkopt_router)
app.include_router(q_attention_router)    # ✅ Added Q-NVLinkOpt endpoint

# ✅ Serve frontend (public dashboard/static site)
app.mount("/", StaticFiles(directory="quantumflow_ai/frontend/public", html=True), name="static")
