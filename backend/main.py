"""
SilentBridge Backend — FastAPI Application Entry Point.

Serves:
    - REST API endpoints at /api/*
    - WebSocket endpoints at /ws/*
    - Frontend static files at /
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.api.routes import router as api_router
from backend.api.websocket import router as ws_router


# ── Create FastAPI app ────────────────────────────────────────────────────

app = FastAPI(
    title="SilentBridge",
    description="Real-time sign language ↔ speech communication bridge",
    version="1.0.0",
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ─────────────────────────────────────────────────────

app.include_router(api_router, prefix="/api", tags=["API"])
app.include_router(ws_router, tags=["WebSocket"])


# ── Health check ──────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from backend.pipelines.sign_inference import inference_engine
    from backend.pipelines.speech_to_text import is_model_available

    return {
        "status": "healthy",
        "service": "SilentBridge",
        "sign_model_loaded": inference_engine.is_loaded,
        "whisper_available": is_model_available(),
    }


# ── Serve frontend static files ──────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.exists(FRONTEND_DIR):
    # Serve index.html and all static files at root
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    @app.get("/")
    async def no_frontend():
        return {
            "message": "SilentBridge API is running. Frontend not found.",
            "docs": "/docs",
        }
