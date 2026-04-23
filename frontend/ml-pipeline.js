/**
 * SilentBridge — MediaPipe + ML Inference Pipeline
 *
 * Handles:
 *   1. MediaPipe Holistic initialization (hand + pose keypoints in browser)
 *   2. Keypoint extraction (162-dim vector per frame)
 *   3. Sliding window buffer (30 frames, 15-frame stride)
 *   4. WebSocket connection to backend inference endpoint
 *   5. Sending keypoint sequences, receiving predictions
 *   6. Training mode: recording labeled gesture sequences
 */

// ── Constants ─────────────────────────────────────────────────────────

const SEQ_LENGTH = 30;           // Frames per inference window
const STRIDE = 15;               // Sliding window stride (50% overlap)
const INPUT_DIM = 162;           // Feature vector size
const FRAME_SKIP = 2;            // Process every Nth frame for performance

// Upper body pose landmark indices (MediaPipe Pose: 11-22)
const UPPER_BODY_INDICES = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22];

// ── State ─────────────────────────────────────────────────────────────

let holisticInstance = null;
let isSignRecognitionActive = false;
let inferenceWs = null;
let keypointBuffer = [];         // Rolling buffer of 162-dim vectors
let frameCount = 0;
let framesSinceLastSend = 0;
let lastPrediction = '';
let predictionCooldown = 0;

// Training mode state
let isRecording = false;
let recordingBuffer = [];
let recordingLabel = '';

// ── MediaPipe Initialization ──────────────────────────────────────────

async function initMediaPipe() {
    try {
        // Dynamically load MediaPipe Holistic from CDN
        if (!window.Holistic) {
            await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/holistic.js');
            await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@0.3.1675466862/camera_utils.js');
            await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.3.1675466124/drawing_utils.js');
        }

        holisticInstance = new window.Holistic({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${file}`;
            }
        });

        holisticInstance.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            enableSegmentation: false,
            smoothSegmentation: false,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
        });

        holisticInstance.onResults(onHolisticResults);

        console.log('[MediaPipe] Holistic model initialized');
        return true;
    } catch (err) {
        console.error('[MediaPipe] Failed to initialize:', err);
        // Fallback: try the simpler approach with GestureRecognizer
        return await initMediaPipeFallback();
    }
}

async function initMediaPipeFallback() {
    /**
     * Fallback: use @mediapipe/tasks-vision GestureRecognizer
     * This is simpler but only provides hand landmarks (not pose).
     * We zero-pad the pose portion of the 162-dim vector.
     */
    try {
        const { GestureRecognizer, HandLandmarker, FilesetResolver } = await import(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3'
        );

        const vision = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
        );

        window._handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numHands: 2,
        });

        console.log('[MediaPipe] HandLandmarker fallback initialized');
        return true;
    } catch (err) {
        console.error('[MediaPipe] Fallback also failed:', err);
        return false;
    }
}

function loadScript(src) {
    return new Promise((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) {
            resolve();
            return;
        }
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

// ── Keypoint Extraction ───────────────────────────────────────────────

function extractKeypoints(results) {
    /**
     * Extract 162-dim feature vector from MediaPipe Holistic results:
     *   - Upper body pose (landmarks 11-22): 12 × 3 = 36
     *   - Left hand (21 landmarks):          21 × 3 = 63
     *   - Right hand (21 landmarks):         21 × 3 = 63
     *   Total: 162
     */
    const keypoints = new Float32Array(INPUT_DIM);
    let offset = 0;

    // Upper body pose: landmarks 11-22 (12 landmarks × 3 coords)
    if (results.poseLandmarks) {
        for (const idx of UPPER_BODY_INDICES) {
            if (results.poseLandmarks[idx]) {
                keypoints[offset++] = results.poseLandmarks[idx].x;
                keypoints[offset++] = results.poseLandmarks[idx].y;
                keypoints[offset++] = results.poseLandmarks[idx].z;
            } else {
                offset += 3; // zeros
            }
        }
    } else {
        offset += 36; // zeros for pose
    }

    // Left hand: 21 landmarks × 3 coords
    if (results.leftHandLandmarks) {
        for (let i = 0; i < 21; i++) {
            if (results.leftHandLandmarks[i]) {
                keypoints[offset++] = results.leftHandLandmarks[i].x;
                keypoints[offset++] = results.leftHandLandmarks[i].y;
                keypoints[offset++] = results.leftHandLandmarks[i].z;
            } else {
                offset += 3;
            }
        }
    } else {
        offset += 63; // zeros
    }

    // Right hand: 21 landmarks × 3 coords
    if (results.rightHandLandmarks) {
        for (let i = 0; i < 21; i++) {
            if (results.rightHandLandmarks[i]) {
                keypoints[offset++] = results.rightHandLandmarks[i].x;
                keypoints[offset++] = results.rightHandLandmarks[i].y;
                keypoints[offset++] = results.rightHandLandmarks[i].z;
            } else {
                offset += 3;
            }
        }
    } else {
        offset += 63; // zeros
    }

    return Array.from(keypoints);
}

function extractKeypointsFallback(handResults) {
    /**
     * Fallback extraction from HandLandmarker (no pose).
     * Zeros for pose, fills left/right hand from detected hands.
     */
    const keypoints = new Float32Array(INPUT_DIM);
    // Pose portion is zeros (offset 0-35)
    let leftHandOffset = 36;
    let rightHandOffset = 36 + 63;

    if (handResults && handResults.landmarks) {
        for (let h = 0; h < handResults.landmarks.length && h < 2; h++) {
            const hand = handResults.landmarks[h];
            const handedness = handResults.handednesses?.[h]?.[0]?.categoryName || 'Left';
            const offset = handedness === 'Right' ? rightHandOffset : leftHandOffset;

            for (let i = 0; i < 21; i++) {
                if (hand[i]) {
                    keypoints[offset + i * 3] = hand[i].x;
                    keypoints[offset + i * 3 + 1] = hand[i].y;
                    keypoints[offset + i * 3 + 2] = hand[i].z;
                }
            }
        }
    }

    return Array.from(keypoints);
}

// ── MediaPipe Results Handler ─────────────────────────────────────────

function onHolisticResults(results) {
    if (!isSignRecognitionActive && !isRecording) return;

    const keypoints = extractKeypoints(results);
    processKeypoints(keypoints, results);
}

function processKeypoints(keypoints, results) {
    frameCount++;

    // Update UI status
    const hasHands = keypoints.slice(36).some(v => v !== 0);
    const signStatus = document.getElementById('sign-status');
    if (signStatus) {
        if (hasHands) {
            signStatus.textContent = 'Tracking hands · Extracting keypoints';
        } else {
            signStatus.textContent = 'MediaPipe active · No hands detected';
        }
    }

    // Training mode: accumulate into recording buffer
    if (isRecording && hasHands) {
        recordingBuffer.push(keypoints);
        updateRecordingProgress();
        return;
    }

    // Inference mode: add to sliding window buffer
    if (isSignRecognitionActive && hasHands) {
        keypointBuffer.push(keypoints);
        framesSinceLastSend++;

        // Send when we have enough frames and stride is met
        if (keypointBuffer.length >= SEQ_LENGTH && framesSinceLastSend >= STRIDE) {
            const sequence = keypointBuffer.slice(-SEQ_LENGTH);
            sendForInference(sequence);
            framesSinceLastSend = 0;

            // Keep last half for overlap
            if (keypointBuffer.length > SEQ_LENGTH * 2) {
                keypointBuffer = keypointBuffer.slice(-SEQ_LENGTH);
            }
        }
    }
}

// ── Video Frame Processing Loop ───────────────────────────────────────

let _animFrameId = null;
let _lastVideoTime = -1;

function processVideoFrame() {
    if (!isSignRecognitionActive && !isRecording) {
        _animFrameId = null;
        return;
    }

    // Determine the active screen to select the correct video element
    // This is critical because hidden video elements (display: none) stop ticking in many browsers.
    const isTrainingMode = document.getElementById('training')?.classList.contains('active');
    
    let video = isTrainingMode 
        ? document.getElementById('train-video') 
        : document.getElementById('self-video');

    // Fallback: if preferred is not ready, try the other one
    if (!video || video.readyState < 2) {
        video = isTrainingMode 
            ? document.getElementById('self-video') 
            : document.getElementById('train-video');
    }

    if (!video || !window.localStream || video.readyState < 2) {
        _animFrameId = requestAnimationFrame(processVideoFrame);
        return;
    }

    // Skip frames for performance
    if (video.currentTime === _lastVideoTime) {
        _animFrameId = requestAnimationFrame(processVideoFrame);
        return;
    }
    _lastVideoTime = video.currentTime;

    // Use Holistic if available, otherwise fallback
    if (holisticInstance) {
        holisticInstance.send({ image: video });
    } else if (window._handLandmarker) {
        const nowMs = performance.now();
        try {
            const results = window._handLandmarker.detectForVideo(video, nowMs);
            const keypoints = extractKeypointsFallback(results);
            processKeypoints(keypoints, results);
        } catch (e) {
            // Skip frame on error
        }
    }

    _animFrameId = requestAnimationFrame(processVideoFrame);
}

// ── Inference WebSocket ───────────────────────────────────────────────

function connectInferenceWs() {
    const wsUrl = `${WS_BASE}/ws/inference/${clientId}`;

    try {
        inferenceWs = new WebSocket(wsUrl);
    } catch (e) {
        console.warn('[Inference] WebSocket failed (backend may not be running)');
        return;
    }

    inferenceWs.onopen = () => {
        console.log('[Inference] WebSocket connected');
    };

    inferenceWs.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === 'prediction') {
            handlePrediction(msg.label, msg.confidence);
        }
    };

    inferenceWs.onerror = () => {
        console.warn('[Inference] WebSocket error (backend may not be running)');
    };

    inferenceWs.onclose = () => {
        console.log('[Inference] WebSocket closed');
        inferenceWs = null;
    };
}

function sendForInference(sequence) {
    // Try WebSocket first
    if (inferenceWs && inferenceWs.readyState === WebSocket.OPEN) {
        inferenceWs.send(JSON.stringify({
            type: 'keypoints',
            data: sequence,
            timestamp: Date.now(),
        }));
        return;
    }

    // Fallback to REST API
    fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ keypoint_sequence: sequence }),
    })
    .then(res => {
        if (res.status === 204) return null;
        return res.json();
    })
    .then(data => {
        if (data && data.label) {
            handlePrediction(data.label, data.confidence);
        }
    })
    .catch(() => {
        // Backend not available — silent fail
    });
}

function handlePrediction(label, confidence) {
    // Debounce: don't repeat the same prediction within cooldown
    if (label === lastPrediction && predictionCooldown > 0) {
        predictionCooldown--;
        return;
    }

    lastPrediction = label;
    predictionCooldown = 3; // Skip next 3 identical predictions

    console.log(`[Inference] Prediction: "${label}" (${(confidence * 100).toFixed(1)}%)`);

    // Display as live subtitle
    if (typeof onSignRecognized === 'function') {
        onSignRecognized(label, confidence);
    }

    // Send to peer via WebRTC data channel
    if (typeof sendTextToPeer === 'function') {
        sendTextToPeer(label, 'Sign → Text');
    }
}

// ── Start/Stop Sign Recognition ───────────────────────────────────────

async function startSignRecognition() {
    if (isSignRecognitionActive) return;

    console.log('[ML] Starting sign recognition pipeline...');

    // Initialize MediaPipe if not already done
    if (!holisticInstance && !window._handLandmarker) {
        const success = await initMediaPipe();
        if (!success) {
            console.error('[ML] MediaPipe initialization failed');
            return;
        }
    }

    isSignRecognitionActive = true;
    keypointBuffer = [];
    frameCount = 0;
    framesSinceLastSend = 0;

    // Connect inference WebSocket
    connectInferenceWs();

    // Start frame processing loop
    _animFrameId = requestAnimationFrame(processVideoFrame);

    const signStatus = document.getElementById('sign-status');
    if (signStatus) signStatus.textContent = 'MediaPipe active · Keypoints extracting';

    console.log('[ML] Sign recognition pipeline started');
}

function stopSignRecognition() {
    isSignRecognitionActive = false;

    if (_animFrameId) {
        cancelAnimationFrame(_animFrameId);
        _animFrameId = null;
    }

    if (inferenceWs) {
        inferenceWs.close();
        inferenceWs = null;
    }

    keypointBuffer = [];
    console.log('[ML] Sign recognition pipeline stopped');
}

// ── Exported ──────────────────────────────────────────────────────────

window.startSignRecognition = startSignRecognition;
window.stopSignRecognition = stopSignRecognition;
