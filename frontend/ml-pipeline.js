const SEQ_LENGTH = 30;
const INPUT_DIM = 162;

const LH_OFFSET = 36;
const RH_OFFSET = 99;

let holisticInstance = null;
let isActive = false;
let keypointBuffer = [];

let ws = null;

async function loadScript(src) {
    return new Promise((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) {
            resolve();
            return;
        }
        const s = document.createElement('script');
        s.src = src;
        s.onload = resolve;
        s.onerror = reject;
        document.head.appendChild(s);
    });
}

async function initMediaPipe() {
    if (!window.Holistic) {
        const BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/holistic';
        await loadScript(`${BASE}/holistic.js`);
    }

    holisticInstance = new window.Holistic({
        locateFile: (file) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });

    holisticInstance.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    holisticInstance.onResults(onHolisticResults);
}

function extractKeypoints(results) {
    const kp = new Float32Array(INPUT_DIM);

    function fill(lms, offset) {
        if (!lms) return;
        for (let i = 0; i < lms.length; i++) {
            kp[offset + i * 3] = lms[i].x;
            kp[offset + i * 3 + 1] = lms[i].y;
            kp[offset + i * 3 + 2] = lms[i].z;
        }
    }

    fill(results.poseLandmarks?.slice(11, 23), 0);
    fill(results.leftHandLandmarks, LH_OFFSET);
    fill(results.rightHandLandmarks, RH_OFFSET);

    return Array.from(kp);
}

/**
 * 🔴 FIXED CORE LOGIC
 * - NEVER drop frames
 * - ALWAYS build sequence
 * - ALWAYS send when ready
 */
function onHolisticResults(results) {
    if (!isActive) return;

    const kp = extractKeypoints(results);

    // ALWAYS push frame (even if weak / partial)
    keypointBuffer.push(kp);

    // Keep buffer fixed size
    if (keypointBuffer.length > SEQ_LENGTH) {
        keypointBuffer.shift();
    }

    // Send only when ready
    if (keypointBuffer.length === SEQ_LENGTH) {
        send(keypointBuffer);
    }
}

function send(seq) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    ws.send(JSON.stringify({
        type: "keypoints",
        data: seq,
        timestamp: Date.now()
    }));
}

function connectWS() {
    ws = new WebSocket(`${WS_BASE}/ws/inference/${clientId}`);

    ws.onopen = () => {
        console.log("[WS] Connected");
    };

    ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);

        if (msg.type === "prediction") {
            console.log("Detected:", msg.label, msg.confidence);
            const el = document.getElementById("sign-status");
            if (el) el.textContent = msg.label;
        }

        if (msg.type === "error") {
            console.error("[WS ERROR]", msg.message);
        }
    };

    ws.onclose = () => {
        console.log("[WS] Closed");
    };
}

async function startSignRecognition() {
    if (isActive) return;

    await initMediaPipe();
    connectWS();

    isActive = true;
    keypointBuffer = [];

    const video = document.getElementById("self-video");

    async function loop() {
        if (!isActive) return;

        if (video && video.readyState >= 2) {
            await holisticInstance.send({ image: video }).catch(() => {});
        }

        requestAnimationFrame(loop);
    }

    loop();
}

function stopSignRecognition() {
    isActive = false;

    if (ws) {
        ws.close();
        ws = null;
    }

    keypointBuffer = [];
}

window.startSignRecognition = startSignRecognition;
window.stopSignRecognition = stopSignRecognition;