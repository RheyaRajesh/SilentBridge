"""
REST API endpoints for SilentBridge backend.

Provides:
    POST /api/predict      — Run sign language inference on a keypoint sequence
    POST /api/collect      — Store a labeled keypoint sequence for training
    POST /api/train        — Train the LSTM model on collected data
    GET  /api/vocabulary   — Get current gesture vocabulary
    GET  /api/model/status — Get model status (loaded, classes, etc.)
    GET  /api/collection/stats — Get training data collection statistics
    DELETE /api/collection — Delete collected training data
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


class CollectRequest(BaseModel):
    label: str = Field(..., description="Text label for the gesture (e.g. 'hello')")
    keypoint_sequence: list = Field(
        ...,
        description="List of 30 frames, each a list of 162 floats",
    )


class TrainRequest(BaseModel):
    epochs: int = Field(default=50, ge=1, le=500)
    batch_size: int = Field(default=16, ge=1, le=128)
    learning_rate: float = Field(default=0.001, gt=0, lt=1)


# ── Training state (simple in-memory tracker) ────────────────────────────

_training_state = {
    "is_training": False,
    "progress": None,
    "result": None,
}


def _training_progress_callback(epoch, loss, accuracy):
    _training_state["progress"] = {
        "epoch": epoch,
        "loss": round(loss, 4),
        "accuracy": round(accuracy, 4),
    }


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


@router.post("/collect")
async def collect_sample(req: CollectRequest):
    """Store a labeled keypoint sequence for training."""
    from backend.pipelines.training import collect_sample as _collect

    try:
        result = _collect(req.label, req.keypoint_sequence)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/train")
async def train_model(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger model training on collected data.
    Training runs in the background. Poll /api/train/status for progress.
    """
    from backend.pipelines.training import train_model as _train
    from backend.pipelines.sign_inference import inference_engine

    if _training_state["is_training"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    def _run_training():
        _training_state["is_training"] = True
        _training_state["progress"] = None
        _training_state["result"] = None

        try:
            result = _train(
                epochs=req.epochs,
                batch_size=req.batch_size,
                learning_rate=req.learning_rate,
                progress_callback=_training_progress_callback,
            )
            _training_state["result"] = result

            # Reload the newly trained model into the inference engine
            if result.get("status") == "complete":
                inference_engine.reload_model()

        except Exception as e:
            _training_state["result"] = {"status": "error", "message": str(e)}
        finally:
            _training_state["is_training"] = False

    background_tasks.add_task(_run_training)
    return {"status": "training_started", "epochs": req.epochs}


@router.get("/train/status")
async def get_training_status():
    """Get current training status and progress."""
    return {
        "is_training": _training_state["is_training"],
        "progress": _training_state["progress"],
        "result": _training_state["result"],
    }


@router.get("/vocabulary")
async def get_vocabulary():
    """Get the current gesture vocabulary."""
    from backend.pipelines.sign_inference import inference_engine

    return {
        "vocabulary": list(inference_engine.vocab.keys()) if inference_engine.vocab else [],
        "num_classes": len(inference_engine.vocab) if inference_engine.vocab else 0,
    }


@router.get("/model/status")
async def get_model_status():
    """Get model status information."""
    from backend.pipelines.sign_inference import inference_engine

    return inference_engine.get_status()


@router.get("/collection/stats")
async def get_collection_stats():
    """Get statistics about collected training data."""
    from backend.pipelines.training import get_collection_stats

    stats = get_collection_stats()
    total = sum(stats.values())
    return {
        "labels": stats,
        "total_samples": total,
        "num_labels": len(stats),
    }


@router.delete("/collection")
async def delete_collection(label: str = None):
    """Delete collected training data. Optionally for a specific label only."""
    from backend.pipelines.training import delete_collected_data

    result = delete_collected_data(label)
    return result
