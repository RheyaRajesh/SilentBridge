import os
import logging
import numpy as np
from collections import deque, Counter

logger = logging.getLogger("sign_inference")

SEQ_LENGTH = 30
INPUT_DIM = 162

SMOOTHING_WINDOW = 4
MAJORITY_REQUIRED = 2
DUPLICATE_COOLDOWN = 4

from backend.pipelines.heuristic_engine import heuristic_engine


class SignInferenceEngine:

    PREDEFINED_GESTURES = [
        "HELLO", "PLEASE", "STOP", "REPEAT", "I / ME", "YOU", "EAT", "DRINK",
        "TOILET", "FRIEND", "SMALL", "READ", "WRITE", "HELP", "NEED", "WANT",
        "GO", "COME", "YES", "NO", "UNDERSTAND", "NOT UNDERSTAND", "CALL",
        "WAIT", "FINISH",
    ]

    def __init__(self):
        self.history = deque(maxlen=SMOOTHING_WINDOW)
        self.last_output = None
        self.dup_counter = 0

    def _normalize(self, seq):
        out = seq.copy()

        l_sh = seq[:, 0:3]
        r_sh = seq[:, 3:6]

        mid = (l_sh + r_sh) / 2.0
        dist = np.linalg.norm(l_sh - r_sh, axis=1)
        scale = np.mean(dist[dist > 1e-4]) if np.any(dist > 1e-4) else 1.0

        if scale < 1e-6:
            scale = 1.0

        for i in range(0, INPUT_DIM, 3):
            out[:, i] = (out[:, i] - mid[:, 0]) / scale
            out[:, i+1] = (out[:, i+1] - mid[:, 1]) / scale

        return out

    def predict(self, keypoint_sequence):

        seq = np.array(keypoint_sequence, dtype=np.float32)

        if seq.shape != (SEQ_LENGTH, INPUT_DIM):
            return None

        # hand presence check
        if not np.any(seq[:, 36:] != 0):
            self._reset()
            return None

        norm_seq = self._normalize(seq)

        label, conf = heuristic_engine.predict(norm_seq)

        if not label or conf < 0.6:
            self.history.append("NO_SIGN")
            return None

        self.history.append(label)

        return self._smooth_and_emit(label, conf)

    def _smooth_and_emit(self, label, conf):

        if len(self.history) < 3:
            return None

        votes = Counter(self.history)
        top, count = votes.most_common(1)[0]

        if count < MAJORITY_REQUIRED:
            return None

        if top == self.last_output:
            self.dup_counter += 1
            if self.dup_counter >= DUPLICATE_COOLDOWN:
                return None
        else:
            self.last_output = top
            self.dup_counter = 0

        return {
            "label": top,
            "confidence": round(conf, 4),
            "all_scores": {}
        }

    def _reset(self):
        self.history.clear()
        self.last_output = None
        self.dup_counter = 0


inference_engine = SignInferenceEngine()