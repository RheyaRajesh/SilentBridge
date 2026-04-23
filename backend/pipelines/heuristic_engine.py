import numpy as np
import logging
from collections import deque, Counter

logger = logging.getLogger("heuristic_engine")


class HeuristicSignEngine:

    LEFT_HAND_START  = 36
    RIGHT_HAND_START = 99
    POSE_START       = 0

    # Hand landmarks
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MID_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    MID_MCP = 9

    def __init__(self):
        self.history = deque(maxlen=8)

    def predict(self, sequence):
        seq = np.array(sequence, dtype=np.float32)

        if seq.shape[0] < 10:
            return None, 0.0

        last = seq[-1]

        # Hands
        lh = last[self.LEFT_HAND_START:self.LEFT_HAND_START+63].reshape(21,3)
        rh = last[self.RIGHT_HAND_START:self.RIGHT_HAND_START+63].reshape(21,3)

        # Pose (for face reference)
        pose = last[self.POSE_START:self.POSE_START+36].reshape(12,3)

        # choose dominant hand
        hand = rh if np.count_nonzero(rh) > np.count_nonzero(lh) else lh

        if np.count_nonzero(hand) < 10:
            return None, 0.0

        wrist = hand[self.WRIST]
        mid_mcp = hand[self.MID_MCP]

        scale = np.linalg.norm(mid_mcp - wrist)
        if scale < 1e-3:
            scale = 1.0

        def dist(a, b):
            return np.linalg.norm(a - b) / scale

        def direction(vec):
            x, y = vec[0], vec[1]
            if abs(x) > abs(y):
                return "RIGHT" if x > 0 else "LEFT"
            else:
                return "DOWN" if y > 0 else "UP"

        # fingertips
        t = hand[self.THUMB_TIP]
        i = hand[self.INDEX_TIP]
        m = hand[self.MID_TIP]
        r = hand[self.RING_TIP]
        p = hand[self.PINKY_TIP]

        # ───────────── GESTURE RULES ─────────────

        # 1. STOP → open palm
        spread = np.var([i[0], m[0], r[0], p[0]])
        if spread > 0.015:
            return self._stable("STOP", 0.95)

        # 2. VICTORY → index + middle extended
        if (dist(i, wrist) > 1.2 and
            dist(m, wrist) > 1.2 and
            dist(r, wrist) < 1.0 and
            dist(p, wrist) < 1.0):
            return self._stable("VICTORY", 0.95)

        # 3. THUMBS UP
        thumb_vec = t - wrist
        if dist(t, wrist) > 1.2 and direction(thumb_vec) == "UP":
            return self._stable("THUMBS_UP", 0.95)

        # 4–7. POINTING DIRECTIONS
        index_vec = i - wrist
        if dist(i, wrist) > 1.2 and all(dist(x, wrist) < 1.0 for x in [m, r, p]):
            dirn = direction(index_vec)

            if dirn == "UP":
                return self._stable("POINT_UP", 0.95)
            elif dirn == "DOWN":
                return self._stable("POINT_DOWN", 0.95)
            elif dirn == "LEFT":
                return self._stable("POINT_LEFT", 0.95)
            elif dirn == "RIGHT":
                return self._stable("POINT_RIGHT", 0.95)

        # 8. THANK YOU → hand near chin
        if len(pose) >= 1:
            chin = pose[0]  # approximate (mediapipe upper face ref)
            if dist(wrist, chin) < 1.2:
                return self._stable("THANK_YOU", 0.9)

        # 9. CRY → hand below eye
        if len(pose) >= 1:
            eye = pose[0]
            if wrist[1] > eye[1] + 0.1:
                return self._stable("CRY", 0.9)

        return None, 0.0

    # ───────────── STABILITY FILTER ─────────────

    def _stable(self, label, conf):
        self.history.append(label)

        if len(self.history) < 4:
            return None, 0.0

        most = Counter(self.history).most_common(1)[0]

        if most[1] >= 3:
            return most[0], conf

        return None, 0.0


heuristic_engine = HeuristicSignEngine()