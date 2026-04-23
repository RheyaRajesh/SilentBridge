/**
 * SilentBridge — Speech Recognition & TTS
 *
 * Key improvements in this version:
 *   1. Browser Web Speech API works standalone — no backend required for basic STT.
 *   2. MediaRecorder replaces deprecated ScriptProcessor for Whisper audio capture.
 *   3. Interim results shown as live subtitle overlay.
 *   4. Auto-restart on all non-fatal speech errors.
 *   5. Whisper WS reconnects with exponential back-off.
 *   6. Proper cleanup on stopSpeechRecognition.
 */

// ── State ─────────────────────────────────────────────────────────────────

let speechRecognition = null;
let mediaRecorder     = null;
let isSpeechActive    = false;
let speechWs          = null;

let _speechWsReconnectTimer = null;
let _speechWsReconnectDelay = 1500;
const SPEECH_WS_MAX_DELAY   = 16000;

// MediaRecorder chunks for Whisper
let _mediaRecorderChunks = [];

// ── Web Speech API (Browser-native, instant results) ──────────────────────

function initWebSpeechAPI() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn('[Speech] Web Speech API not supported in this browser');
        return false;
    }

    speechRecognition = new SpeechRecognition();
    speechRecognition.continuous     = true;
    speechRecognition.interimResults = true;
    speechRecognition.maxAlternatives = 1;
    speechRecognition.lang = (typeof currentLang !== 'undefined' && currentLang === 'ta')
        ? 'ta-IN'
        : 'en-US';

    speechRecognition.onresult = (event) => {
        let interimText = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript.trim();

            if (event.results[i].isFinal) {
                if (transcript) {
                    console.log('[Speech] Final transcript:', transcript);
                    if (typeof onSpeechTranscribed === 'function') {
                        onSpeechTranscribed(transcript, 'browser');
                    }
                }
            } else {
                interimText += transcript;
            }
        }

        // Show interim text in status banner and subtitle overlay
        if (interimText) {
            const signStatus = document.getElementById('sign-status');
            if (signStatus) {
                signStatus.textContent = '🎤 ' + interimText.substring(0, 60) + (interimText.length > 60 ? '…' : '');
            }
        }
    };

    speechRecognition.onerror = (event) => {
        const fatalErrors = ['not-allowed', 'service-not-allowed', 'audio-capture'];
        if (fatalErrors.includes(event.error)) {
            console.error('[Speech] Fatal Web Speech error:', event.error);
            isSpeechActive = false;
            const signStatus = document.getElementById('sign-status');
            if (signStatus) signStatus.textContent = 'Mic access denied — check browser permissions';
            return;
        }
        // Non-fatal errors (network, aborted, no-speech) — onend will restart
        console.debug('[Speech] Non-fatal error (will auto-restart):', event.error);
    };

    speechRecognition.onstart = () => {
        console.log('[Speech] Web Speech API started listening');
        const signStatus = document.getElementById('sign-status');
        if (signStatus) signStatus.textContent = 'AI Engine active · Listening for speech…';
    };

    speechRecognition.onend = () => {
        if (isSpeechActive) {
            console.debug('[Speech] Web Speech ended — restarting in 250ms…');
            setTimeout(() => {
                if (isSpeechActive && speechRecognition) {
                    try {
                        speechRecognition.start();
                    } catch (e) {
                        console.debug('[Speech] Web Speech restart skipped:', e.message);
                    }
                }
            }, 250);
        }
    };

    return true;
}

// ── Whisper Backend WebSocket ─────────────────────────────────────────────

function connectSpeechWs() {
    if (typeof WS_BASE === 'undefined' || typeof clientId === 'undefined') {
        console.warn('[Speech] WS_BASE or clientId not defined — Whisper WS skipped');
        return;
    }

    if (speechWs && (speechWs.readyState === WebSocket.OPEN ||
                      speechWs.readyState === WebSocket.CONNECTING)) {
        return;
    }

    const wsUrl = `${WS_BASE}/ws/speech/${clientId}`;
    console.log('[Speech] Connecting Whisper WS:', wsUrl);

    try {
        speechWs = new WebSocket(wsUrl);
    } catch (e) {
        console.warn('[Speech] Whisper WebSocket creation failed:', e);
        _scheduleSpeechWsReconnect();
        return;
    }

    speechWs.onopen = () => {
        console.log('[Speech] Whisper WebSocket connected');
        _speechWsReconnectDelay = 1500;
        clearTimeout(_speechWsReconnectTimer);
    };

    speechWs.onmessage = (event) => {
        let msg;
        try { msg = JSON.parse(event.data); }
        catch (e) { console.warn('[Speech] Non-JSON Whisper response:', event.data); return; }

        if (msg.type === 'transcription' && msg.text) {
            console.log('[Speech] Whisper transcription:', msg.text);
            if (typeof onSpeechTranscribed === 'function') {
                onSpeechTranscribed(msg.text, 'whisper');
            }
        } else if (msg.type === 'error') {
            console.error('[Speech] Backend Whisper error:', msg.message);
        } else if (msg.type === 'pong') {
            // heartbeat — ignore
        }
    };

    speechWs.onerror = (e) => {
        console.warn('[Speech] Whisper WebSocket error:', e);
    };

    speechWs.onclose = (ev) => {
        console.log('[Speech] Whisper WS closed (code=%d)', ev.code);
        speechWs = null;
        if (isSpeechActive) _scheduleSpeechWsReconnect();
    };
}

function _scheduleSpeechWsReconnect() {
    clearTimeout(_speechWsReconnectTimer);
    if (!isSpeechActive) return;
    console.log(`[Speech] Whisper WS reconnecting in ${_speechWsReconnectDelay}ms…`);
    _speechWsReconnectTimer = setTimeout(() => {
        connectSpeechWs();
        _speechWsReconnectDelay = Math.min(_speechWsReconnectDelay * 2, SPEECH_WS_MAX_DELAY);
    }, _speechWsReconnectDelay);
}

// ── MediaRecorder → Whisper ────────────────────────────────────────────────

function startMediaRecorderForWhisper() {
    if (!window.localStream) {
        console.warn('[Speech] No localStream for MediaRecorder');
        return;
    }

    const audioTracks = window.localStream.getAudioTracks();
    if (audioTracks.length === 0) {
        console.warn('[Speech] No audio tracks available');
        return;
    }

    try {
        const audioStream = new MediaStream(audioTracks);

        // Pick best supported MIME type
        const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', '']
            .find(m => !m || MediaRecorder.isTypeSupported(m)) || '';

        mediaRecorder = new MediaRecorder(audioStream, mimeType ? { mimeType } : {});
        _mediaRecorderChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) {
                _mediaRecorderChunks.push(e.data);
            }
        };

        mediaRecorder.onstop = async () => {
            if (_mediaRecorderChunks.length === 0) return;

            const blob = new Blob(_mediaRecorderChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
            _mediaRecorderChunks = [];

            if (blob.size < 1000) {
                console.debug('[Speech] Audio blob too small, skipping Whisper');
                return;
            }

            // Send to Whisper if WS is ready
            if (speechWs && speechWs.readyState === WebSocket.OPEN) {
                try {
                    const arrayBuffer = await blob.arrayBuffer();
                    const uint8 = new Uint8Array(arrayBuffer);
                    let binary = '';
                    const CHUNK = 8192;
                    for (let i = 0; i < uint8.length; i += CHUNK) {
                        binary += String.fromCharCode(...uint8.subarray(i, i + CHUNK));
                    }
                    const base64 = btoa(binary);
                    const lang = (typeof currentLang !== 'undefined' && currentLang === 'ta') ? 'ta' : 'en';

                    speechWs.send(JSON.stringify({
                        type:        'audio',
                        data:        base64,
                        sample_rate: 16000,
                        language:    lang,
                        timestamp:   Date.now(),
                    }));
                    console.debug('[Speech] Sent %d bytes of audio to Whisper', uint8.length);
                } catch (e) {
                    console.warn('[Speech] Failed to send audio to Whisper:', e);
                }
            }

            // Restart recording if still active
            if (isSpeechActive && mediaRecorder && mediaRecorder.state === 'inactive') {
                try {
                    mediaRecorder.start();
                    setTimeout(() => {
                        if (isSpeechActive && mediaRecorder && mediaRecorder.state === 'recording') {
                            mediaRecorder.stop();
                        }
                    }, 3000);
                } catch (e) {
                    console.debug('[Speech] MediaRecorder restart failed:', e);
                }
            }
        };

        // Start recording — stop every 3 seconds to send a chunk to Whisper
        mediaRecorder.start();
        setTimeout(() => {
            if (isSpeechActive && mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
        }, 3000);

        console.log('[Speech] MediaRecorder started (3s chunks for Whisper)');
    } catch (err) {
        console.error('[Speech] MediaRecorder init failed:', err);
    }
}

// ── Start / Stop ──────────────────────────────────────────────────────────

function startSpeechRecognition() {
    if (isSpeechActive) return;
    isSpeechActive = true;
    console.log('[Speech] Starting speech recognition…');

    // 1. Browser Web Speech API (instant, works on localhost/HTTPS)
    if (!speechRecognition) {
        initWebSpeechAPI();
    }
    if (speechRecognition) {
        try {
            speechRecognition.start();
        } catch (e) {
            console.warn('[Speech] Web Speech start failed:', e);
        }
    }

    // 2. Whisper backend via WebSocket + MediaRecorder
    _speechWsReconnectDelay = 1500;
    connectSpeechWs();
    startMediaRecorderForWhisper();
}

function stopSpeechRecognition() {
    isSpeechActive = false;
    clearTimeout(_speechWsReconnectTimer);
    console.log('[Speech] Stopping speech recognition…');

    if (speechRecognition) {
        try { speechRecognition.stop(); } catch (e) { /* already stopped */ }
        speechRecognition = null;
    }

    if (mediaRecorder) {
        try {
            if (mediaRecorder.state !== 'inactive') mediaRecorder.stop();
        } catch (e) { /* ignore */ }
        mediaRecorder = null;
    }

    _mediaRecorderChunks = [];

    if (speechWs) {
        speechWs.close();
        speechWs = null;
    }

    console.log('[Speech] Speech recognition stopped');
}

// ── Exports ───────────────────────────────────────────────────────────────

window.startSpeechRecognition = startSpeechRecognition;
window.stopSpeechRecognition  = stopSpeechRecognition;
