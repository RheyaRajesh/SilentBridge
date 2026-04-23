"""
REST API endpoints for SilentBridge backend.

Provides:
    POST /api/predict      — Run sign language inference on a keypoint sequence
    GET  /api/vocabulary   — Get current gesture vocabulary
    GET  /api/model/status — Get model status (loaded, classes, etc.)
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

router = APIRouter()


# ── Request/Response schemas ──────────────────────────────────────────────

class PredictRequest(BaseModel):
    keypoint_sequence: list = Field(
        ...,
        description="List of 30 frames, each a list of 162 floats (MediaPipe keypoints)",
    )


class PredictResponse(BaseModel):
    label: str
    confidence: float
    all_scores: dict





# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Run inference on a 30-frame keypoint sequence."""
    from backend.pipelines.sign_inference import inference_engine

    result = inference_engine.predict(req.keypoint_sequence)
    if result is None:
        raise HTTPException(
            status_code=204,
            detail="No prediction (model not loaded or confidence below threshold)",
        )
    return result


@router.get("/vocabulary")
async def get_vocabulary():
    """Get the current gesture vocabulary."""
    from backend.pipelines.sign_inference import inference_engine
    return {
        "vocabulary": inference_engine.predefined_gestures,
        "num_classes": len(inference_engine.predefined_gestures),
    }


@router.get("/model/status")
async def get_model_status():
    """Get model status information."""
    from backend.pipelines.sign_inference import inference_engine
    return inference_engine.get_status()
