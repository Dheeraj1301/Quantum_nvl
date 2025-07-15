# quantumflow_ai/api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from quantumflow_ai.api.q_router import router as q_router
from quantumflow_ai.api.energy import router as energy_router

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

# ✅ Serve frontend
app.mount("/", StaticFiles(directory="quantumflow_ai/frontend/public", html=True), name="static")
