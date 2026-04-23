"""
Sign language inference pipeline.

Loads a trained SignLanguageLSTM model and provides single-sequence inference.
If no trained model exists, inference returns None (filtered by confidence threshold).
"""

import os
import json
import numpy as np
import torch
from backend.models.sign_lstm import SignLanguageLSTM, load_model, create_model
from backend.pipelines.heuristic_engine import heuristic_engine

# Constants
SEQ_LENGTH = 30        # Number of frames per sequence
INPUT_DIM = 162        # MediaPipe keypoint feature vector size
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(DATA_DIR, "models", "sign_lstm.pt")
VOCAB_PATH = os.path.join(DATA_DIR, "models", "vocab.json")


class SignInferenceEngine:
    """Manages model loading and real-time sign language inference."""

    def __init__(self):
        self.model: SignLanguageLSTM | None = None
        self.vocab: dict = {}           # {label: index}
        self.index_to_label: dict = {}  # {index: label}
        self.device = "cpu"
        self.is_loaded = False

        # Attempt to load existing trained model
        self._try_load_model()

    def _try_load_model(self):
        """Try to load a previously trained model from disk."""
        if os.path.exists(MODEL_PATH):
            try:
                self.model, self.vocab = load_model(MODEL_PATH, self.device)
                self.index_to_label = {v: k for k, v in self.vocab.items()}
                self.is_loaded = True
                print(f"[SignInference] Model loaded with {len(self.vocab)} classes: {list(self.vocab.keys())}")
            except Exception as e:
                print(f"[SignInference] Failed to load model: {e}")
                self.is_loaded = False
        else:
            print("[SignInference] No trained model found. Use training mode to create one.")
            self.is_loaded = False

    def reload_model(self):
        """Reload model after training completes."""
        self._try_load_model()

    def predict(self, keypoint_sequence: list) -> dict | None:
        """
        Run inference on a keypoint sequence.

        Args:
            keypoint_sequence: List of 30 frames, each frame is a list of 162 floats.

        Returns:
            {
                "label": str,
                "confidence": float,
                "all_scores": {label: score, ...}
            }
            or None if model not loaded or confidence below threshold.
        """
        # 1. Try Heuristic Engine first (fast, pre-trained universal signs)
        try:
            h_label, h_conf = heuristic_engine.predict(keypoint_sequence)
            if h_label and h_conf >= self.threshold if hasattr(self, 'threshold') else 0.7:
                return {
                    "label": h_label,
                    "confidence": h_conf,
                    "all_scores": {h_label: h_conf}
                }
        except Exception as e:
            print(f"[SignInference] Heuristic error: {e}")

        # 2. Try Deep Learning model (if loaded)
        if not self.is_loaded or self.model is None:
            return None

        try:
            # Convert to tensor: (1, 30, 162)
            seq_array = np.array(keypoint_sequence, dtype=np.float32)
            if seq_array.shape != (SEQ_LENGTH, INPUT_DIM):
                print(f"[SignInference] Invalid shape: {seq_array.shape}, expected ({SEQ_LENGTH}, {INPUT_DIM})")
                return None

            input_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                logits = self.model(input_tensor)  # (1, num_classes)
                probs = torch.softmax(logits, dim=-1)  # (1, num_classes)

            probs_np = probs.squeeze(0).cpu().numpy()
            predicted_idx = int(np.argmax(probs_np))
            confidence = float(probs_np[predicted_idx])

            # Build all scores dict
            all_scores = {}
            for idx, score in enumerate(probs_np):
                label = self.index_to_label.get(idx, f"class_{idx}")
                all_scores[label] = round(float(score), 4)

            predicted_label = self.index_to_label.get(predicted_idx, f"class_{predicted_idx}")

            if confidence < CONFIDENCE_THRESHOLD:
                return None

            return {
                "label": predicted_label,
                "confidence": round(confidence, 4),
                "all_scores": all_scores,
            }

        except Exception as e:
            print(f"[SignInference] Inference error: {e}")
            return None

    def get_status(self) -> dict:
        """Return current model status."""
        return {
            "loaded": self.is_loaded,
            "num_classes": len(self.vocab) if self.vocab else 0,
            "vocabulary": list(self.vocab.keys()) if self.vocab else [],
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "model_path": MODEL_PATH if self.is_loaded else None,
        }


# Singleton instance
inference_engine = SignInferenceEngine()
