"""
WebSocket endpoints for SilentBridge.

Provides:
    /ws/signal/{room_id}/{client_id}  — WebRTC signaling relay
    /ws/inference/{client_id}         — Real-time sign language inference
    /ws/speech/{client_id}            — Real-time Whisper speech-to-text
"""

import json
import asyncio
import base64
import numpy as np
from collections import defaultdict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


# ── WebRTC Signaling ──────────────────────────────────────────────────────

# room_id → {client_id: WebSocket}
signaling_rooms: dict[str, dict[str, WebSocket]] = defaultdict(dict)


@router.websocket("/ws/signal/{room_id}/{client_id}")
async def websocket_signaling(websocket: WebSocket, room_id: str, client_id: str):
    """
    WebRTC signaling relay.

    Messages are JSON with a "type" field:
        - "offer": SDP offer → relayed to other client
        - "answer": SDP answer → relayed to other client
        - "ice_candidate": ICE candidate → relayed to other client
        - "chat": text message → relayed to other client
    """
    await websocket.accept()
    signaling_rooms[room_id][client_id] = websocket
    print(f"[Signal] Client '{client_id}' joined room '{room_id}' "
          f"({len(signaling_rooms[room_id])} clients)")

    # Notify other clients that a new peer joined
    for cid, ws in signaling_rooms[room_id].items():
        if cid != client_id:
            try:
                await ws.send_text(json.dumps({
                    "type": "peer_joined",
                    "client_id": client_id,
                }))
            except Exception:
                pass

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            # Relay to all other clients in the same room
            for cid, ws in signaling_rooms[room_id].items():
                if cid != client_id:
                    try:
                        # Tag the sender
                        message["from"] = client_id
                        await ws.send_text(json.dumps(message))
                    except Exception:
                        pass

    except WebSocketDisconnect:
        print(f"[Signal] Client '{client_id}' left room '{room_id}'")
    except Exception as e:
        print(f"[Signal] Error for '{client_id}': {e}")
    finally:
        # Cleanup
        if room_id in signaling_rooms and client_id in signaling_rooms[room_id]:
            del signaling_rooms[room_id][client_id]
            # Notify remaining clients
            for cid, ws in signaling_rooms[room_id].items():
                try:
                    await ws.send_text(json.dumps({
                        "type": "peer_left",
                        "client_id": client_id,
                    }))
                except Exception:
                    pass
            if not signaling_rooms[room_id]:
                del signaling_rooms[room_id]


@router.get("/rooms")
async def get_active_rooms():
    """List active signaling rooms."""
    rooms = {}
    for room_id, clients in signaling_rooms.items():
        rooms[room_id] = list(clients.keys())
    return {"rooms": rooms}


# ── Real-time Sign Language Inference ─────────────────────────────────────

@router.websocket("/ws/inference/{client_id}")
async def websocket_inference(websocket: WebSocket, client_id: str):
    """
    Real-time sign language inference via WebSocket.

    Client sends JSON messages:
        {
            "type": "keypoints",
            "data": [[162 floats] × 30],  // 30-frame sequence
            "timestamp": 1234567890
        }

    Server responds:
        {
            "type": "prediction",
            "label": "hello",
            "confidence": 0.87,
            "timestamp": 1234567890
        }
        or
        {
            "type": "no_prediction",
            "reason": "below_threshold" | "model_not_loaded"
        }
    """
    await websocket.accept()
    print(f"[Inference] Client '{client_id}' connected")

    from backend.pipelines.sign_inference import inference_engine

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "keypoints":
                keypoint_data = message.get("data", [])
                timestamp = message.get("timestamp", 0)

                # Run inference (offload to thread to avoid blocking event loop)
                result = await asyncio.to_thread(
                    inference_engine.predict, keypoint_data
                )

                if result:
                    await websocket.send_text(json.dumps({
                        "type": "prediction",
                        "label": result["label"],
                        "confidence": result["confidence"],
                        "timestamp": timestamp,
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "no_prediction",
                        "reason": "below_threshold" if inference_engine.is_loaded else "model_not_loaded",
                        "timestamp": timestamp,
                    }))

            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        print(f"[Inference] Client '{client_id}' disconnected")
    except Exception as e:
        print(f"[Inference] Error for '{client_id}': {e}")


# ── Real-time Speech-to-Text ─────────────────────────────────────────────

@router.websocket("/ws/speech/{client_id}")
async def websocket_speech(websocket: WebSocket, client_id: str):
    """
    Real-time speech-to-text via WebSocket.

    Client sends JSON messages:
        {
            "type": "audio",
            "data": "<base64-encoded PCM audio>",
            "sample_rate": 16000,
            "language": "en",
            "timestamp": 1234567890
        }

    Server responds:
        {
            "type": "transcription",
            "text": "hello world",
            "timestamp": 1234567890
        }
    """
    await websocket.accept()
    print(f"[Speech] Client '{client_id}' connected")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "audio":
                audio_b64 = message.get("data", "")
                sample_rate = message.get("sample_rate", 16000)
                language = message.get("language", "en")
                timestamp = message.get("timestamp", 0)

                # Decode base64 audio
                audio_bytes = base64.b64decode(audio_b64)

                # Run Whisper transcription in thread
                from backend.pipelines.speech_to_text import transcribe_audio_bytes
                text = await asyncio.to_thread(
                    transcribe_audio_bytes, audio_bytes, sample_rate, language
                )

                if text:
                    await websocket.send_text(json.dumps({
                        "type": "transcription",
                        "text": text,
                        "timestamp": timestamp,
                    }))

            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        print(f"[Speech] Client '{client_id}' disconnected")
    except Exception as e:
        print(f"[Speech] Error for '{client_id}': {e}")
