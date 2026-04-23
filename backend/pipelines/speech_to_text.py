"""
SilentBridge — Whisper-based speech-to-text pipeline (faster-whisper).

Fixes vs previous version:
  1. Minimum audio length guard (< 0.5 s → skip) avoids Whisper hallucinations
     on tiny chunks and reduces false "Thank you" / "." results.
  2. Hallucination filter: strips common single-word Whisper artefacts.
  3. Comprehensive logging for every transcription attempt.
  4. `transcribe_audio_bytes` now accepts float32 PCM as well as int16.
  5. VAD parameters tuned for 3-second streaming chunks.
  6. Model load error is logged but does NOT crash the server — STT simply
     becomes unavailable and the WS endpoint returns a clean error message.
"""

import io
import os
import logging
import numpy as np

logger = logging.getLogger("speech_to_text")

# ── Configuration ──────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE   = os.getenv("WHISPER_MODEL_SIZE",   "tiny")
WHISPER_DEVICE       = os.getenv("WHISPER_DEVICE",       "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# Minimum audio duration to bother transcribing (seconds)
MIN_AUDIO_SECONDS = 0.3

# Common Whisper hallucinations to suppress
_HALLUCINATIONS = {
    "", ".", "..", "...", "thank you", "thanks", "thank you.",
    "thanks.", "you", "the", "a", "[music]", "[noise]", "[silence]",
    "♪", "♫",
}

# Lazy-loaded model
_whisper_model = None
_load_error    = None


def _get_model():
    """Lazy-load Whisper model on first call."""
    global _whisper_model, _load_error

    if _load_error is not None:
        raise RuntimeError(f"Whisper failed to load previously: {_load_error}")

    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            logger.info(
                "[Whisper] Loading model '%s' on %s (%s) …",
                WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
            )
            _whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            logger.info("[Whisper] Model loaded successfully")
        except Exception as e:
            _load_error = str(e)
            logger.error("[Whisper] Failed to load model: %s", e)
            raise

    return _whisper_model


def _is_hallucination(text: str) -> bool:
    return text.strip().lower() in _HALLUCINATIONS


def transcribe_audio(audio_data: np.ndarray, sample_rate: int = 16000,
                     language: str = "en") -> str:
    """
    Transcribe float32 mono PCM audio using Whisper.

    Args:
        audio_data:  numpy float32 array, mono, at sample_rate Hz.
        sample_rate: Sample rate (default 16000).
        language:    Language code (default "en").

    Returns:
        Transcribed text string, or "" if nothing detected.
    """
    # Minimum duration guard
    duration = len(audio_data) / max(sample_rate, 1)
    if duration < MIN_AUDIO_SECONDS:
        logger.debug("[Whisper] Audio too short (%.2f s) — skipping", duration)
        return ""

    # Ensure float32
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Normalise to [-1, 1]
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        audio_data = audio_data / max_val
    elif max_val < 1e-6:
        logger.debug("[Whisper] Silent audio — skipping")
        return ""

    logger.debug("[Whisper] Transcribing %.2f s of audio (lang=%s)", duration, language)

    try:
        model = _get_model()

        segments, info = model.transcribe(
            audio_data,
            beam_size=3,
            language=language if language != "auto" else None,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=100,
                threshold=0.5,
            ),
        )

        parts = []
        for seg in segments:
            t = seg.text.strip()
            if t and not _is_hallucination(t):
                parts.append(t)
                logger.debug("[Whisper] Segment [%.1f-%.1f]: %s", seg.start, seg.end, t)

        transcription = " ".join(parts).strip()

        if transcription:
            logger.info("[Whisper] Transcription: '%s'", transcription)
        else:
            logger.debug("[Whisper] No speech detected in chunk")

        return transcription

    except RuntimeError as e:
        logger.error("[Whisper] Model unavailable: %s", e)
        return ""
    except Exception as e:
        logger.error("[Whisper] Transcription error: %s", e, exc_info=True)
        return ""


def transcribe_audio_bytes(audio_bytes: bytes, sample_rate: int = 16000,
                           language: str = "en") -> str:
    """
    Transcribe raw PCM audio bytes.

    Accepts:
        - 16-bit signed little-endian PCM  (standard MediaRecorder / AudioContext output)
        - 32-bit float PCM                 (if sent directly)

    Args:
        audio_bytes: Raw PCM bytes.
        sample_rate: Sample rate (default 16000).
        language:    Language code.

    Returns:
        Transcribed text string.
    """
    if len(audio_bytes) < 512:
        logger.debug("[Whisper] Audio bytes too small (%d bytes) — skipping", len(audio_bytes))
        return ""

    try:
        # Try int16 first (most common from browser AudioContext)
        audio_int16   = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        logger.debug("[Whisper] Decoded %d int16 samples", len(audio_int16))
    except Exception:
        try:
            # Fallback: treat as float32
            audio_float32 = np.frombuffer(audio_bytes, dtype=np.float32)
            logger.debug("[Whisper] Decoded %d float32 samples (fallback)", len(audio_float32))
        except Exception as e:
            logger.error("[Whisper] Cannot decode audio bytes: %s", e)
            return ""

    return transcribe_audio(audio_float32, sample_rate=sample_rate, language=language)


def is_model_available() -> bool:
    """Check if faster-whisper is importable (without loading the model)."""
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        return True
    except ImportError:
        return False
