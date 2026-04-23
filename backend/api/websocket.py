"""
WebSocket endpoints for SilentBridge.

Provides:
    /ws/signal/{room_id}/{client_id}  — WebRTC signaling relay
    /ws/inference/{client_id}         — Real-time sign language inference
    /ws/speech/{client_id}            — Real-time Whisper speech-to-text

Fixes vs previous version:
    1. Inference WS: validates keypoint shape before sending to engine.
    2. Inference WS: structured error response instead of silent drop.
    3. Speech WS: catches base64 decode errors and returns error frame.
    4. Speech WS: logs every transcription attempt (success + failure).
    5. Signaling WS: logs room membership changes clearly.
    6. All endpoints: catch broad Exception with traceback logging.
    7. Inference WS: sends `no_sign` type when nothing detected (frontend
       can use this to reset interim display).
    8. Speech WS: sends `error` type on Whisper failure so client knows.
"""

import json
import asyncio
import base64
import logging
import traceback
import numpy as np
from collections import defaultdict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("websocket")

router = APIRouter()


# ── WebRTC Signaling ──────────────────────────────────────────────────────

# room_id → {client_id: WebSocket}
signaling_rooms: dict[str, dict[str, WebSocket]] = defaultdict(dict)


@router.websocket("/ws/signal/{room_id}/{client_id}")
async def websocket_signaling(websocket: WebSocket, room_id: str, client_id: str):
    """
    WebRTC signaling relay. Supports:
        offer | answer | ice_candidate | chat
    Also emits:
        peer_joined | peer_left
    """
    await websocket.accept()
    signaling_rooms[room_id][client_id] = websocket
    n = len(signaling_rooms[room_id])
    logger.info("[Signal] Client '%s' joined room '%s' (%d client(s))", client_id, room_id, n)

    # Notify existing peers
    for cid, ws in signaling_rooms[room_id].items():
        if cid != client_id:
            try:
                await ws.send_text(json.dumps({
                    "type":      "peer_joined",
                    "client_id": client_id,
                }))
                logger.info("[Signal] Notified '%s' that '%s' joined", cid, client_id)
            except Exception as e:
                logger.warning("[Signal] Could not notify '%s': %s", cid, e)

    try:
        while True:
            raw  = await websocket.receive_text()
            try:
                message  = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("[Signal] Received non-JSON from '%s'", client_id)
                continue

            msg_type = message.get("type", "unknown")
            logger.debug("[Signal] '%s' → type='%s'", client_id, msg_type)

            # Relay to all other clients in the room
            message["from"] = client_id
            dead_peers = []
            for cid, ws in signaling_rooms[room_id].items():
                if cid == client_id:
                    continue
                try:
                    await ws.send_text(json.dumps(message))
                except Exception as e:
                    logger.warning("[Signal] Relay to '%s' failed: %s", cid, e)
                    dead_peers.append(cid)

            for cid in dead_peers:
                signaling_rooms[room_id].pop(cid, None)

    except WebSocketDisconnect:
        logger.info("[Signal] Client '%s' disconnected from room '%s'", client_id, room_id)
    except Exception as e:
        logger.error("[Signal] Unhandled error for '%s': %s\n%s",
                     client_id, e, traceback.format_exc())
    finally:
        signaling_rooms[room_id].pop(client_id, None)
        # Notify remaining peers
        remaining = list(signaling_rooms[room_id].items())
        for cid, ws in remaining:
            try:
                await ws.send_text(json.dumps({
                    "type":      "peer_left",
                    "client_id": client_id,
                }))
            except Exception:
                pass
        if not signaling_rooms[room_id]:
            del signaling_rooms[room_id]
            logger.info("[Signal] Room '%s' closed (empty)", room_id)


@router.get("/rooms")
async def get_active_rooms():
    """List active signaling rooms and their occupants."""
    return {
        "rooms": {rid: list(clients.keys()) for rid, clients in signaling_rooms.items()}
    }


# ── Real-time Sign Language Inference ─────────────────────────────────────

SEQ_LENGTH = 30
INPUT_DIM  = 162


@router.websocket("/ws/inference/{client_id}")
async def websocket_inference(websocket: WebSocket, client_id: str):
    """
    Real-time sign language inference.

    Client → server:
        { "type": "keypoints", "data": [[162 floats] × 30], "timestamp": ms }
        { "type": "ping" }

    Server → client:
        { "type": "prediction",   "label": "HELLO", "confidence": 0.93, "timestamp": ms }
        { "type": "no_sign",      "reason": "below_threshold", "timestamp": ms }
        { "type": "error",        "message": "...", "timestamp": ms }
        { "type": "pong" }
    """
    await websocket.accept()
    logger.info("[Inference] Client '%s' connected", client_id)

    from backend.pipelines.sign_inference import inference_engine

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("[Inference] Non-JSON message from '%s'", client_id)
                continue

            msg_type  = message.get("type")
            timestamp = message.get("timestamp", 0)

            if msg_type == "keypoints":
                keypoint_data = message.get("data", [])

                # ── Shape validation ──────────────────────────────────────
                if (not isinstance(keypoint_data, list) or
                        len(keypoint_data) != SEQ_LENGTH):
                    logger.warning(
                        "[Inference] Invalid sequence length from '%s': %d (expected %d)",
                        client_id, len(keypoint_data) if isinstance(keypoint_data, list) else -1,
                        SEQ_LENGTH
                    )
                    await websocket.send_text(json.dumps({
                        "type":      "error",
                        "message":   f"Expected {SEQ_LENGTH} frames, got {len(keypoint_data)}",
                        "timestamp": timestamp,
                    }))
                    continue

                logger.debug(
                    "[Inference] Running inference for '%s' (seq=%d×%d)",
                    client_id, SEQ_LENGTH, INPUT_DIM
                )

                # Run inference off event-loop thread
                try:
                    result = await asyncio.to_thread(
                        inference_engine.predict, keypoint_data
                    )
                except Exception as e:
                    logger.error("[Inference] Engine error for '%s': %s", client_id, e,
                                 exc_info=True)
                    await websocket.send_text(json.dumps({
                        "type":      "error",
                        "message":   f"Inference error: {e}",
                        "timestamp": timestamp,
                    }))
                    continue

                if result:
                    logger.info(
                        "[Inference] → '%s' prediction: %s (%.2f)",
                        client_id, result["label"], result["confidence"]
                    )
                    await websocket.send_text(json.dumps({
                        "type":       "prediction",
                        "label":      result["label"],
                        "confidence": result["confidence"],
                        "timestamp":  timestamp,
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "type":      "no_sign",
                        "reason":    "below_threshold",
                        "timestamp": timestamp,
                    }))

            elif msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

            else:
                logger.debug("[Inference] Unknown message type '%s' from '%s'", msg_type, client_id)

    except WebSocketDisconnect:
        logger.info("[Inference] Client '%s' disconnected", client_id)
    except Exception as e:
        logger.error("[Inference] Unhandled error for '%s': %s\n%s",
                     client_id, e, traceback.format_exc())


# ── Real-time Speech-to-Text ─────────────────────────────────────────────

@router.websocket("/ws/speech/{client_id}")
async def websocket_speech(websocket: WebSocket, client_id: str):
    """
    Real-time speech-to-text via Whisper.

    Client → server:
        { "type": "audio", "data": "<base64 PCM>", "sample_rate": 16000,
          "language": "en", "timestamp": ms }
        { "type": "ping" }

    Server → client:
        { "type": "transcription", "text": "hello world", "timestamp": ms }
        { "type": "error",         "message": "...",       "timestamp": ms }
        { "type": "pong" }
    """
    await websocket.accept()
    logger.info("[Speech] Client '%s' connected", client_id)

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("[Speech] Non-JSON from '%s'", client_id)
                continue

            msg_type  = message.get("type")
            timestamp = message.get("timestamp", 0)

            if msg_type == "audio":
                audio_b64   = message.get("data", "")
                sample_rate = int(message.get("sample_rate", 16000))
                language    = message.get("language", "en")

                if not audio_b64:
                    logger.debug("[Speech] Empty audio payload from '%s'", client_id)
                    continue

                # ── Decode base64 ─────────────────────────────────────────
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                except Exception as e:
                    logger.error("[Speech] Base64 decode error for '%s': %s", client_id, e)
                    await websocket.send_text(json.dumps({
                        "type":      "error",
                        "message":   f"Audio decode error: {e}",
                        "timestamp": timestamp,
                    }))
                    continue

                logger.debug(
                    "[Speech] Received %d bytes of audio from '%s' (sr=%d, lang=%s)",
                    len(audio_bytes), client_id, sample_rate, language
                )

                # ── Transcribe in thread ──────────────────────────────────
                from backend.pipelines.speech_to_text import transcribe_audio_bytes
                try:
                    text = await asyncio.to_thread(
                        transcribe_audio_bytes, audio_bytes, sample_rate, language
                    )
                except Exception as e:
                    logger.error("[Speech] Transcription error for '%s': %s", client_id, e,
                                 exc_info=True)
                    await websocket.send_text(json.dumps({
                        "type":      "error",
                        "message":   f"Transcription error: {e}",
                        "timestamp": timestamp,
                    }))
                    continue

                if text:
                    logger.info("[Speech] Transcription for '%s': '%s'", client_id, text)
                    await websocket.send_text(json.dumps({
                        "type":      "transcription",
                        "text":      text,
                        "timestamp": timestamp,
                    }))
                else:
                    logger.debug("[Speech] No speech detected for '%s'", client_id)

            elif msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

            else:
                logger.debug("[Speech] Unknown type '%s' from '%s'", msg_type, client_id)

    except WebSocketDisconnect:
        logger.info("[Speech] Client '%s' disconnected", client_id)
    except Exception as e:
        logger.error("[Speech] Unhandled error for '%s': %s\n%s",
                     client_id, e, traceback.format_exc())
