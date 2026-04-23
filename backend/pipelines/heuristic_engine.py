import numpy as np

class HeuristicSignEngine:
    """
    Recognizes common sign language gestures using geometric heuristics on MediaPipe keypoints.
    This provides instant 'pre-trained' functionality without requiring user training.
    """

    def __init__(self, seq_length=30, confidence_threshold=0.7):
        self.seq_length = seq_length
        self.threshold = confidence_threshold
        # Keypoint layout constants
        self.POSE_START = 0
        self.LEFT_HAND_START = 36
        self.RIGHT_HAND_START = 99

    def predict(self, sequence):
        """
        Processes a sequence of 30 frames (each 162-dim).
        Returns (label, confidence) or (None, 0).
        """
        seq = np.array(sequence) # (30, 162)
        
        # 1. Analyze Motion
        # Extract right hand wrist (indices 99, 100, 101)
        rw_x = seq[:, self.RIGHT_HAND_START]
        rw_y = seq[:, self.RIGHT_HAND_START + 1]
        
        # Check for waving motion (high variance in X)
        x_variance = np.var(rw_x[rw_x != 0]) if np.any(rw_x != 0) else 0
        if x_variance > 0.005:
            return "HELLO", 0.95

        # 2. Analyze Hand Shapes (latest frame)
        last_frame = seq[-1]
        
        # Helper to check Right Hand
        label, conf = self._check_hand_shape(last_frame, self.RIGHT_HAND_START)
        if label:
            return label, conf
            
        # Helper to check Left Hand
        label, conf = self._check_hand_shape(last_frame, self.LEFT_HAND_START)
        if label:
            return label, conf

        return None, 0

    def _check_hand_shape(self, frame, offset):
        """Analyze a single hand shape from keypoints."""
        # Extract 21 landmarks (x, y, z)
        hand = frame[offset : offset + 63].reshape(21, 3)
        if np.all(hand == 0):
            return None, 0

        # Indices: 0: Wrist, 4: Thumb, 8: Index, 12: Middle, 16: Ring, 20: Pinky
        wrist = hand[0]
        thumb_tip = hand[4]
        index_tip = hand[8]
        middle_tip = hand[12]
        ring_tip = hand[16]
        pinky_tip = hand[20]

        # 1. THUMBS UP (GOOD / YES)
        # Thumb tip is higher (lower Y) than all other fingers and wrist
        fingers_y = [hand[8][1], hand[12][1], hand[16][1], hand[20][1]]
        if thumb_tip[1] < min(fingers_y) - 0.1 and thumb_tip[1] < wrist[1] - 0.1:
            # Check if other fingers are folded (close to wrist)
            if all(np.linalg.norm(hand[i] - wrist) < 0.2 for i in [8, 12, 16, 20]):
                return "GOOD", 0.9

        # 2. PEACE / VICTORY (V-Sign)
        # Index and Middle Extended, others folded
        if (index_tip[1] < wrist[1] - 0.15 and 
            middle_tip[1] < wrist[1] - 0.15 and 
            ring_tip[1] > middle_tip[1] + 0.1 and
            pinky_tip[1] > middle_tip[1] + 0.1):
            return "PEACE", 0.85

        # 3. I LOVE YOU (ILY)
        # Thumb, Index, Pinky Extended, Middle/Ring folded
        if (thumb_tip[0] > wrist[0] + 0.1 and
            index_tip[1] < wrist[1] - 0.15 and 
            pinky_tip[1] < wrist[1] - 0.15 and
            middle_tip[1] > wrist[1] - 0.1 and
            ring_tip[1] > wrist[1] - 0.1):
            return "LOVE", 0.95

        # 4. THANK YOU (Static snapshot of hand near chin/chest)
        # Hand flat, fingers extended together
        all_tips_extended = all(hand[i][1] < wrist[1] - 0.15 for i in [8, 12, 16, 20])
        tips_together = np.var([hand[i][0] for i in [8, 12, 16]]) < 0.002
        if all_tips_extended and tips_together:
            return "THANK_YOU", 0.8

        return None, 0

heuristic_engine = HeuristicSignEngine()
