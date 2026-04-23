/**
 * SilentBridge — Speech Recognition & TTS
 *
 * Handles:
 *   1. Web Speech API (SpeechRecognition) for instant browser-side STT
 *   2. Audio recording via MediaRecorder for Whisper backend (higher accuracy)
 *   3. WebSocket streaming of audio chunks to backend
 *   4. TTS via SpeechSynthesis API
 */

// ── State ─────────────────────────────────────────────────────────────

let speechRecognition = null;
let mediaRecorder = null;
let audioChunks = [];
let isSpeechActive = false;
let speechWs = null;

// ── Web Speech API (Browser-native, instant) ──────────────────────────

function initWebSpeechAPI() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn('[Speech] Web Speech API not supported in this browser');
        return false;
    }

    speechRecognition = new SpeechRecognition();
    speechRecognition.continuous = true;
    speechRecognition.interimResults = true;
    speechRecognition.lang = currentLang === 'ta' ? 'ta-IN' : 'en-US';

    let finalTranscript = '';
    let interimTimeout = null;

    speechRecognition.onresult = (event) => {
        let interim = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript.trim();

            if (event.results[i].isFinal) {
                finalTranscript = transcript;

                // Display final transcript
                if (typeof onSpeechTranscribed === 'function') {
                    onSpeechTranscribed(finalTranscript, 'browser');
                }

                // Send to peer
                if (typeof sendTextToPeer === 'function') {
                    sendTextToPeer(finalTranscript, 'Speech → Text');
                }

                finalTranscript = '';
            } else {
                interim += transcript;
            }
        }

        // Update sign status with interim results
        if (interim) {
            const signStatus = document.getElementById('sign-status');
            if (signStatus) {
                signStatus.textContent = 'Listening: "' + interim.substring(0, 40) + '..."';
            }
        }
    };

    speechRecognition.onerror = (event) => {
        if (event.error === 'no-speech') return; // Ignore no-speech errors
        console.warn('[Speech] Recognition error:', event.error);
    };

    speechRecognition.onend = () => {
        // Auto-restart if still active
        if (isSpeechActive) {
            try {
                speechRecognition.start();
            } catch (e) {
                // Already started
            }
        }
    };

    return true;
}

// ── Whisper Backend (Higher accuracy, async) ──────────────────────────

function connectSpeechWs() {
    const wsUrl = `${WS_BASE}/ws/speech/${clientId}`;

    try {
        speechWs = new WebSocket(wsUrl);
    } catch (e) {
        console.warn('[Speech] Whisper WebSocket failed');
        return;
    }

    speechWs.onopen = () => {
        console.log('[Speech] Whisper WebSocket connected');
    };

    speechWs.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'transcription' && msg.text) {
            // Whisper provides higher accuracy — display as authoritative
            if (typeof onSpeechTranscribed === 'function') {
                onSpeechTranscribed(msg.text, 'whisper');
            }
        }
    };

    speechWs.onerror = () => {
        console.warn('[Speech] Whisper WebSocket error');
    };

    speechWs.onclose = () => {
        speechWs = null;
    };
}

function startAudioRecording() {
    if (!window.localStream) return;

    const audioTracks = window.localStream.getAudioTracks();
    if (audioTracks.length === 0) return;

    const audioStream = new MediaStream(audioTracks);

    try {
        mediaRecorder = new MediaRecorder(audioStream, {
            mimeType: 'audio/webm;codecs=opus',
        });
    } catch (e) {
        // Fallback to default mime type
        mediaRecorder = new MediaRecorder(audioStream);
    }

    audioChunks = [];

    mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0) {
            // Convert to PCM and send to Whisper backend
            const arrayBuffer = await event.data.arrayBuffer();
            sendAudioToWhisper(arrayBuffer);
        }
    };

    // Collect audio in 3-second chunks
    mediaRecorder.start(3000);
    console.log('[Speech] Audio recording started (3s chunks)');
}

async function sendAudioToWhisper(arrayBuffer) {
    if (!speechWs || speechWs.readyState !== WebSocket.OPEN) return;

    try {
        // Convert ArrayBuffer to base64
        const uint8Array = new Uint8Array(arrayBuffer);
        let binary = '';
        for (let i = 0; i < uint8Array.length; i++) {
            binary += String.fromCharCode(uint8Array[i]);
        }
        const base64 = btoa(binary);

        speechWs.send(JSON.stringify({
            type: 'audio',
            data: base64,
            sample_rate: 16000,
            language: currentLang === 'ta' ? 'ta' : 'en',
            timestamp: Date.now(),
        }));
    } catch (e) {
        console.warn('[Speech] Failed to send audio to Whisper:', e);
    }
}

// ── Start/Stop Speech Recognition ─────────────────────────────────────

function startSpeechRecognition() {
    if (isSpeechActive) return;
    isSpeechActive = true;

    console.log('[Speech] Starting speech recognition...');

    // Start Web Speech API (instant)
    if (!speechRecognition) {
        initWebSpeechAPI();
    }

    if (speechRecognition) {
        try {
            speechRecognition.start();
            console.log('[Speech] Web Speech API started');
        } catch (e) {
            console.warn('[Speech] Web Speech API start failed:', e);
        }
    }

    // Connect Whisper backend (higher accuracy)
    connectSpeechWs();

    // Start audio recording for Whisper
    startAudioRecording();

    // Update UI
    const signStatus = document.getElementById('sign-status');
    if (signStatus) signStatus.textContent = 'Whisper active · Listening for speech';
}

function stopSpeechRecognition() {
    isSpeechActive = false;

    if (speechRecognition) {
        try {
            speechRecognition.stop();
        } catch (e) {
            // Already stopped
        }
    }

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder = null;
    }

    if (speechWs) {
        speechWs.close();
        speechWs = null;
    }

    console.log('[Speech] Speech recognition stopped');
}

// ── Exported ──────────────────────────────────────────────────────────

window.startSpeechRecognition = startSpeechRecognition;
window.stopSpeechRecognition = stopSpeechRecognition;
