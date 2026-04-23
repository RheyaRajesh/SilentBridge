/**
 * SilentBridge — WebRTC Peer Connection & Signaling (Fixed)
 *
 * Fixes applied:
 *   1. ondatachannel now stores channel as global `dataChannel` — answerer can send
 *   2. `ignoreOffer` scoping bug fixed — was used outside its case block (ReferenceError)
 *   3. Data channel only created by offerer — removed from createPeerConnection()
 *   4. ICE candidates queued when remote description not yet set (InvalidStateError fix)
 *   5. Signaling WebSocket reconnect with back-off
 *   6. isPolite managed cleanly — guest flag set before call starts
 *   7. Connection state badge updates for all states
 *   8. Peer stream displayed reliably via ontrack
 */

// ── Configuration ──────────────────────────────────────────────────────────
const RTC_CONFIG = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
        { urls: 'stun:stun2.l.google.com:19302' },
        { urls: 'stun:stun3.l.google.com:19302' },
        { urls: 'stun:stun4.l.google.com:19302' },
    ]
};

// ── State ──────────────────────────────────────────────────────────────────
let peerConnection   = null;
let signalingSocket  = null;
let dataChannel      = null;   // Global — used by BOTH offerer and answerer
let currentRoomId    = null;

// clientId shared with ml-pipeline.js and speech.js
let clientId = 'client_' + Math.random().toString(36).slice(2, 11);

// Perfect-negotiation flags
let isPolite     = false;   // true = guest (answerer-preferred)
let makingOffer  = false;

// ICE candidate queue — holds candidates that arrive before remote SDP is set
let _iceCandidateQueue = [];

// Signaling reconnect
let _sigReconnectTimer = null;
let _sigReconnectDelay = 1500;

// Streams
window.localStream  = null;
window.remoteStream = null;

// ── Camera ─────────────────────────────────────────────────────────────────

async function startLiveCamera() {
    // Reset any previous error state
    const selfviewEl = document.querySelector('.video-selfview');
    if (selfviewEl) selfviewEl.classList.remove('cam-error');

    try {
        window.localStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 },
                     facingMode: 'user', frameRate: { ideal: 30 } },
            audio: true,
        });

        const selfVideo = document.getElementById('self-video');
        if (selfVideo) {
            selfVideo.srcObject = window.localStream;
            // Explicit play() — required by Chrome autoplay policy
            try {
                await selfVideo.play();
            } catch (playErr) {
                console.warn('[WebRTC] video.play() blocked:', playErr);
                // Non-fatal: srcObject is set, video will play on user gesture
            }
        }

        // (Removed background preview to prevent double-video)

        // Mark selfview as active with green border glow
        if (selfviewEl) {
            selfviewEl.classList.add('cam-active');
            const placeholder = document.getElementById('cam-placeholder');
            if (placeholder) placeholder.style.display = 'none';
            const liveBadge = document.getElementById('selfview-live-badge');
            if (liveBadge) liveBadge.style.display = 'flex';
        }

        const statusEl = document.getElementById('sign-status');
        if (statusEl) statusEl.textContent = 'Camera active · Initializing AI engine…';

        console.log('[WebRTC] Camera started, tracks:', window.localStream.getTracks().length);
    } catch (err) {
        console.error('[WebRTC] Camera access failed:', err);
        if (selfviewEl) selfviewEl.classList.add('cam-error');

        const statusEl = document.getElementById('sign-status');
        if (statusEl) {
            statusEl.textContent = err.name === 'NotAllowedError'
                ? '⚠ Camera blocked — click the 🔒 icon in the address bar to allow'
                : '⚠ Camera unavailable: ' + err.message;
        }
    }
}


function stopLiveCamera() {
    if (window.localStream) {
        window.localStream.getTracks().forEach(t => t.stop());
        window.localStream = null;
    }
    const selfVideo   = document.getElementById('self-video');
    const remoteVideo = document.getElementById('remote-video');
    if (selfVideo)   selfVideo.srcObject   = null;
    if (remoteVideo) remoteVideo.srcObject = null;
    console.log('[WebRTC] Camera stopped');
}

// ── Peer Connection ────────────────────────────────────────────────────────

function createPeerConnection() {
    if (peerConnection) return peerConnection;

    peerConnection = new RTCPeerConnection(RTC_CONFIG);

    // Add local tracks
    if (window.localStream) {
        window.localStream.getTracks().forEach(track =>
            peerConnection.addTrack(track, window.localStream)
        );
    }

    // ICE candidates
    peerConnection.onicecandidate = ({ candidate }) => {
        if (candidate && signalingSocket?.readyState === WebSocket.OPEN) {
            signalingSocket.send(JSON.stringify({
                type:      'ice_candidate',
                candidate: {
                    candidate:     candidate.candidate,
                    sdpMid:        candidate.sdpMid,
                    sdpMLineIndex: candidate.sdpMLineIndex,
                },
            }));
        }
    };

    // Remote tracks
    peerConnection.ontrack = ({ track, streams }) => {
        console.log('[WebRTC] Remote track:', track.kind);
        if (!window.remoteStream) window.remoteStream = new MediaStream();
        window.remoteStream.addTrack(track);
        const rv = document.getElementById('remote-video');
        if (rv) {
            rv.srcObject = window.remoteStream;
            rv.style.filter = '';   // Clear blurred self-preview filter
            rv.muted = false;
            rv.play().catch(e => console.warn('[WebRTC] remote video play():', e));
        }
    };

    // Connection state
    peerConnection.onconnectionstatechange = () => {
        const state = peerConnection.connectionState;
        console.log('[WebRTC] State:', state);
        _updateBadge(state);

        if (state === 'failed') {
            console.warn('[WebRTC] Connection failed — restarting ICE');
            peerConnection.restartIce();
        }
    };

    // ── FIX: ondatachannel stores to global `dataChannel` ─────────────────
    // This was the bug: the old code used a local variable `incomingChannel`
    // so the answerer could never call sendTextToPeer.
    peerConnection.ondatachannel = (event) => {
        console.log('[WebRTC] Data channel received from offerer');
        dataChannel = event.channel;   // <── FIXED: global, not local
        _setupDataChannel(dataChannel);
    };

    // NOTE: data channel is NOT created here — only the offerer creates it
    // (moved to createAndSendOffer). This prevents duplicate channel negotiation.

    return peerConnection;
}

function _setupDataChannel(ch) {
    ch.onopen    = () => console.log('[WebRTC] Data channel open');
    ch.onclose   = () => console.log('[WebRTC] Data channel closed');
    ch.onmessage = (ev) => {
        try {
            const msg = JSON.parse(ev.data);
            if (msg.type === 'text') {
                addBubble('transcript-panel', 'them', msg.label || 'Peer', msg.text);
            }
        } catch {
            addBubble('transcript-panel', 'them', 'Peer', ev.data);
        }
    };
}

function _updateBadge(state) {
    const badge = document.getElementById('webrtc-badge');
    if (!badge) return;
    const map = {
        'connected':    ['WebRTC ●', '#1D9E75'],
        'connecting':   ['Connecting…', '#f0a500'],
        'disconnected': ['Reconnecting…', '#f0a500'],
        'failed':       ['Connection failed', '#e24b4a'],
        'closed':       ['Disconnected', '#e24b4a'],
    };
    const [text, color] = map[state] || ['WebRTC ●', '#9b9baa'];
    badge.textContent  = text;
    badge.style.color  = color;
}

// ── Signaling ──────────────────────────────────────────────────────────────

async function startWebRTCCall(roomId) {
    currentRoomId = roomId;
    _iceCandidateQueue = [];
    connectSignaling(roomId);
}

function stopWebRTCCall() {
    clearTimeout(_sigReconnectTimer);

    if (peerConnection) { peerConnection.close(); peerConnection = null; }
    if (dataChannel)    { dataChannel.close();    dataChannel    = null; }
    if (signalingSocket){ signalingSocket.close(); signalingSocket = null; }

    window.remoteStream = null;
    currentRoomId       = null;
    _iceCandidateQueue  = [];
    console.log('[WebRTC] Call ended');
}

function connectSignaling(roomId) {
    const url = `${WS_BASE}/ws/signal/${roomId}/${clientId}`;
    console.log('[Signal] Connecting:', url);

    try {
        signalingSocket = new WebSocket(url);
    } catch (e) {
        console.warn('[Signal] WS failed:', e);
        scheduleSignalingReconnect(roomId);
        return;
    }

    signalingSocket.onopen = () => {
        console.log('[Signal] Connected');
        _sigReconnectDelay = 1500;
        clearTimeout(_sigReconnectTimer);
        // Create peer connection now (but NOT offer — wait for peer_joined)
        createPeerConnection();
    };

    signalingSocket.onmessage = async (ev) => {
        try { await handleSignalingMessage(JSON.parse(ev.data)); }
        catch (e) { console.error('[Signal] Message error:', e); }
    };

    signalingSocket.onerror = (e) => {
        console.warn('[Signal] WS error:', e);
    };

    signalingSocket.onclose = () => {
        console.log('[Signal] WS closed');
        signalingSocket = null;
        if (currentRoomId) scheduleSignalingReconnect(currentRoomId);
    };
}

function scheduleSignalingReconnect(roomId) {
    clearTimeout(_sigReconnectTimer);
    _sigReconnectTimer = setTimeout(() => {
        if (currentRoomId) {
            console.log(`[Signal] Reconnecting (${_sigReconnectDelay}ms)…`);
            connectSignaling(roomId);
            _sigReconnectDelay = Math.min(_sigReconnectDelay * 2, 16000);
        }
    }, _sigReconnectDelay);
}

// ── Signaling Message Handler ──────────────────────────────────────────────

async function handleSignalingMessage(msg) {
    if (!peerConnection) createPeerConnection();

    switch (msg.type) {

        case 'peer_joined':
            // We are the host (already in room) → become offerer
            // isPolite stays false for the host; guest already has isPolite=true
            console.log('[Signal] Peer joined — sending offer');
            await createAndSendOffer();
            break;

        case 'offer': {
            // ── FIX: `ignoreOffer` properly scoped inside this case block ─
            const offerCollision = makingOffer ||
                                   peerConnection.signalingState !== 'stable';
            const ignoreOffer    = !isPolite && offerCollision;

            if (ignoreOffer) {
                console.log('[Signal] Ignoring glare offer (impolite peer)');
                return;
            }

            console.log('[Signal] Received offer — creating answer');
            await peerConnection.setRemoteDescription(
                new RTCSessionDescription({ type: 'offer', sdp: msg.sdp })
            );

            // Flush queued ICE candidates now that remote SDP is set
            await _flushIceCandidateQueue();

            const answer = await peerConnection.createAnswer();
            await peerConnection.setLocalDescription(answer);
            signalingSocket.send(JSON.stringify({
                type: 'answer',
                sdp:  peerConnection.localDescription.sdp,
            }));
            break;
        }

        case 'answer':
            console.log('[Signal] Received answer');
            if (peerConnection.signalingState === 'have-local-offer') {
                await peerConnection.setRemoteDescription(
                    new RTCSessionDescription({ type: 'answer', sdp: msg.sdp })
                );
                await _flushIceCandidateQueue();
            }
            break;

        case 'ice_candidate':
            // ── FIX: queue candidates if remote description not yet set ───
            if (peerConnection.remoteDescription && peerConnection.remoteDescription.type) {
                try {
                    await peerConnection.addIceCandidate(
                        new RTCIceCandidate(msg.candidate)
                    );
                } catch (e) {
                    console.warn('[Signal] ICE candidate error:', e);
                }
            } else {
                console.log('[Signal] Queuing ICE candidate (no remote SDP yet)');
                _iceCandidateQueue.push(msg.candidate);
            }
            break;

        case 'chat':
            if (msg.text) {
                addBubble('transcript-panel', 'them', msg.label || 'Peer', msg.text);
            }
            break;

        case 'peer_left':
            console.log('[Signal] Peer left room');
            _updateBadge('disconnected');
            break;
    }
}

async function _flushIceCandidateQueue() {
    while (_iceCandidateQueue.length) {
        const candidate = _iceCandidateQueue.shift();
        try {
            await peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
        } catch (e) {
            console.warn('[Signal] Queued ICE candidate error:', e);
        }
    }
}

// ── Offer / Answer ─────────────────────────────────────────────────────────

async function createAndSendOffer() {
    if (!peerConnection || !signalingSocket) return;

    // ── FIX: data channel only created by offerer ─────────────────────────
    if (!dataChannel || dataChannel.readyState === 'closed') {
        dataChannel = peerConnection.createDataChannel('silentbridge-text', { ordered: true });
        _setupDataChannel(dataChannel);
    }

    try {
        makingOffer = true;
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        signalingSocket.send(JSON.stringify({
            type: 'offer',
            sdp:  peerConnection.localDescription.sdp,
        }));
        console.log('[Signal] Offer sent');
    } catch (e) {
        console.error('[WebRTC] Offer error:', e);
    } finally {
        makingOffer = false;
    }
}

// ── Text Relay ─────────────────────────────────────────────────────────────

function sendTextToPeer(text, label) {
    const payload = JSON.stringify({ type: 'text', text, label: label || 'Peer' });

    // Data channel (fastest, P2P)
    if (dataChannel && dataChannel.readyState === 'open') {
        dataChannel.send(payload);
    } else if (signalingSocket && signalingSocket.readyState === WebSocket.OPEN) {
        // Signaling fallback (goes via server — always works)
        signalingSocket.send(JSON.stringify({ type: 'chat', text, label: label || 'Peer' }));
    }
}

// ── Exports ────────────────────────────────────────────────────────────────
window.startLiveCamera  = startLiveCamera;
window.stopLiveCamera   = stopLiveCamera;
window.startWebRTCCall  = startWebRTCCall;
window.stopWebRTCCall   = stopWebRTCCall;
window.sendTextToPeer   = sendTextToPeer;
