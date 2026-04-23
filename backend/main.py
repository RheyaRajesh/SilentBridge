"""
SilentBridge Backend — FastAPI Application Entry Point.

Serves:
    - REST API endpoints at /api/*
    - WebSocket endpoints at /ws/*
    - Frontend static files at /

Fixes vs previous version:
    1. Logging configured at startup with timestamps and level names.
    2. Health check now returns Whisper model status via try/except
       so a Whisper load failure does not crash the health endpoint.
    3. Static file mount uses StaticFiles(html=True) which correctly
       serves index.html for all unknown paths (SPA routing support).
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes   import router as api_router
from backend.api.websocket import router as ws_router


# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ── FastAPI App ───────────────────────────────────────────────────────────

app = FastAPI(
    title="SilentBridge",
    description="Real-time sign language ↔ speech communication bridge",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────

app.include_router(api_router, prefix="/api", tags=["API"])
app.include_router(ws_router,  tags=["WebSocket"])


# ── Health Check ──────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check — reports sign-model and Whisper status."""
    from backend.pipelines.sign_inference import inference_engine

    whisper_ok = False
    try:
        from backend.pipelines.speech_to_text import is_model_available
        whisper_ok = is_model_available()
    except Exception as e:
        logger.warning("[Health] Whisper status check failed: %s", e)

    sign_status = inference_engine.get_status()
    logger.info("[Health] OK — lstm_loaded=%s whisper=%s",
                sign_status.get("lstm_weights_loaded"), whisper_ok)

    return {
        "status":            "healthy",
        "service":           "SilentBridge",
        "sign_engine":       sign_status,
        "whisper_available": whisper_ok,
    }


# ── Static Frontend ──────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    logger.info("[Main] Serving frontend from: %s", FRONTEND_DIR)
else:
    logger.warning("[Main] Frontend directory not found at: %s", FRONTEND_DIR)

    @app.get("/")
    async def no_frontend():
        return {
            "message": "SilentBridge API running. Frontend not found.",
            "docs":    "/docs",
        }
