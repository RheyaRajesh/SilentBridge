"""
Whisper-based speech-to-text pipeline using faster-whisper.

Uses the 'base' model by default (configurable via WHISPER_MODEL_SIZE env var).
Accepts raw PCM audio as numpy array (16kHz mono float32) and returns transcribed text.
"""

import os
import numpy as np

# Model configuration
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# Lazy-loaded model instance
_whisper_model = None


def _get_model():
    """Lazy-load Whisper model on first use to avoid slow startup."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            print(f"[Whisper] Loading model '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})...")
            _whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            print("[Whisper] Model loaded successfully.")
        except Exception as e:
            print(f"[Whisper] Failed to load model: {e}")
            raise
    return _whisper_model


def transcribe_audio(audio_data: np.ndarray, sample_rate: int = 16000, language: str = "en") -> str:
    """
    Transcribe audio using Whisper.

    Args:
        audio_data: numpy array of float32 PCM audio, mono, at `sample_rate` Hz.
        sample_rate: Sample rate of the audio (default 16000).
        language: Language code for transcription (default "en").

    Returns:
        Transcribed text string. Empty string if no speech detected.
    """
    model = _get_model()

    # Ensure correct dtype
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Normalize to [-1, 1] if needed
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        audio_data = audio_data / max_val

    try:
        segments, info = model.transcribe(
            audio_data,
            beam_size=5,
            language=language,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        transcription = " ".join(segment.text.strip() for segment in segments)
        return transcription.strip()

    except Exception as e:
        print(f"[Whisper] Transcription error: {e}")
        return ""


def transcribe_audio_bytes(audio_bytes: bytes, sample_rate: int = 16000, language: str = "en") -> str:
    """
    Transcribe raw PCM audio bytes (16-bit signed integers).

    Args:
        audio_bytes: Raw PCM audio bytes (16-bit little-endian mono).
        sample_rate: Sample rate (default 16000).
        language: Language code.

    Returns:
        Transcribed text string.
    """
    # Convert 16-bit PCM bytes to float32 numpy array
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    return transcribe_audio(audio_float32, sample_rate=sample_rate, language=language)


def is_model_available() -> bool:
    """Check if the Whisper model is available (without loading it)."""
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        return True
    except ImportError:
        return False
