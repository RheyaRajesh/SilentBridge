# SilentBridge

**Connecting voices beyond sound** — Real-time communication between deaf/mute sign language users and speaking users.

## What it does

SilentBridge is a web application that enables real-time video calls between:
- **Deaf/mute users** who communicate via sign language → recognized by AI → displayed as live text
- **Speaking users** who communicate via voice → transcribed by Whisper AI → displayed as live text

## Architecture

```
Browser (Frontend)                    FastAPI (Backend)
├── MediaPipe Holistic (WASM)         ├── POST /api/predict (LSTM inference)
│   → 162-dim keypoints/frame         ├── POST /api/collect (save training data)
├── WebRTC PeerConnection             ├── POST /api/train (train LSTM model)
│   → Peer-to-peer video/audio        ├── WS /ws/signal (WebRTC signaling)
├── Web Speech API (instant STT)      ├── WS /ws/inference (real-time keypoints)
└── Sliding window buffer             └── WS /ws/speech (Whisper ASR)
    → 30 frames × 162 dims
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, JavaScript (no frameworks) |
| Video Call | WebRTC (browser-native) with STUN servers |
| Sign Recognition | MediaPipe Holistic (browser) → PyTorch LSTM (backend) |
| Speech-to-Text | Web Speech API (browser) + faster-whisper (backend) |
| Backend | Python FastAPI |
| ML Model | Bidirectional LSTM with attention (PyTorch) |

## Quick Start

### Prerequisites
- Python 3.10+
- A modern browser (Chrome recommended for Web Speech API)
- Webcam & microphone

### Installation

```bash
cd "DL Package"
pip install -r backend/requirements.txt
```

### Run

```bash
python run.py
```

Opens at `http://localhost:8000`

### How to Use

1. **Choose Role**: Select "I use sign language" or "I use voice"
2. **Train Your Model** (for sign language users):
   - Go to **Train** tab
   - Enter a gesture label (e.g., "hello")
   - Start camera → Record the gesture → Repeat 5-10 times
   - Add more gesture labels (need at least 2)
   - Click **Train Model**
3. **Make a Call**: Click a contact to start a video call
4. **Live Subtitles**: Your signs or speech appear as live text overlays

## ML Model Details

### SignLanguageLSTM

| Parameter | Value |
|-----------|-------|
| Architecture | Bidirectional LSTM + Attention |
| Input | `(batch, 30, 162)` — 30 frames × 162 keypoints |
| Keypoints | 36 (upper body) + 63 (left hand) + 63 (right hand) |
| Hidden dim | 128 |
| LSTM layers | 2 |
| Output | `(batch, num_classes)` |
| Loss | CrossEntropyLoss |
| Optimizer | Adam (lr=0.001) |

### Pipeline

1. Webcam → MediaPipe Holistic (runs in browser WASM)
2. Extract 162-dim keypoint vector per frame
3. Sliding window: 30 frames, 15-frame stride
4. Send to backend via WebSocket
5. LSTM inference → predicted label + confidence
6. Display as live subtitle (if confidence > 0.6)

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Run inference on keypoint sequence |
| `/api/collect` | POST | Store labeled training sample |
| `/api/train` | POST | Train model (background task) |
| `/api/train/status` | GET | Training progress |
| `/api/vocabulary` | GET | Current gesture labels |
| `/api/model/status` | GET | Model loaded status |
| `/api/collection/stats` | GET | Training data stats |
| `/ws/signal/{room}/{client}` | WS | WebRTC signaling |
| `/ws/inference/{client}` | WS | Real-time keypoint inference |
| `/ws/speech/{client}` | WS | Real-time Whisper STT |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API docs |

## Project Structure

```
DL Package/
├── frontend/
│   ├── index.html          # All screens (splash, home, call, training, etc.)
│   ├── styles.css           # Design system
│   ├── app.js               # Core navigation & call logic
│   ├── webrtc.js            # WebRTC peer connection
│   ├── ml-pipeline.js       # MediaPipe + inference
│   └── speech.js            # Speech recognition & TTS
├── backend/
│   ├── main.py              # FastAPI app
│   ├── requirements.txt     # Python deps
│   ├── models/
│   │   └── sign_lstm.py     # PyTorch LSTM model
│   ├── pipelines/
│   │   ├── sign_inference.py
│   │   ├── speech_to_text.py
│   │   └── training.py
│   ├── api/
│   │   ├── routes.py        # REST endpoints
│   │   └── websocket.py     # WebSocket endpoints
│   └── data/                # Auto-created
│       ├── collected/       # Training data
│       └── models/          # Saved models
└── run.py                   # Launch everything
```

## Limitations

1. **Model ships untrained** — Use Training Mode to record your own gestures
2. **Whisper downloads ~140MB** on first run (the `base` model)
3. **WebRTC needs TURN server** for cross-network calls (STUN-only for local)
4. **No authentication** — prototype uses local state only
5. **Browser TTS quality** varies (Chrome best)

## Deployment (Render)

```bash
# Backend web service
Build: pip install -r backend/requirements.txt
Start: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Set environment variables:
- `WHISPER_MODEL_SIZE=tiny` (use `tiny` for faster inference on Render)
- `CONFIDENCE_THRESHOLD=0.6`
