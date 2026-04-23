/**
 * SilentBridge — WebRTC Peer Connection & Signaling
 *
 * Implements real WebRTC peer-to-peer video/audio with:
 *   - RTCPeerConnection with STUN servers
 *   - WebSocket-based signaling relay
 *   - SDP offer/answer exchange
 *   - ICE candidate trickle
 *   - Local camera/mic stream management
 *   - Data channel for text relay between peers
 */

// ── Configuration ─────────────────────────────────────────────────────

const RTC_CONFIG = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
        { urls: 'stun:stun2.l.google.com:19302' },
    ]
};

// ── State ─────────────────────────────────────────────────────────────

let peerConnection = null;
let signalingSocket = null;
let dataChannel = null;
let currentRoomId = null;
let clientId = 'client_' + Math.random().toString(36).substr(2, 9);

// Polite Peer variables to prevent race conditions
let isPolite = false; 
let makingOffer = false;
let isSettingRemoteAnswerPending = false;

// window.localStream is shared with other modules (ml-pipeline, speech)
window.localStream = null;
window.remoteStream = null;

// ── Camera Management ─────────────────────────────────────────────────

async function startLiveCamera() {
    try {
        // Request camera and microphone
        window.localStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user',
                frameRate: { ideal: 30 },
            },
            audio: true,
        });

        // Attach to self-view video element
        const selfVideo = document.getElementById('self-video');
        if (selfVideo) {
            selfVideo.srcObject = window.localStream;
        }

        // Let the remote-video remain empty (or showing poster) until peer joins

        console.log('[WebRTC] Camera started successfully');

        // Update sign status
        const signStatus = document.getElementById('sign-status');
        if (signStatus) signStatus.textContent = 'Camera active · Waiting for MediaPipe';

    } catch (err) {
        console.error('[WebRTC] Camera access failed:', err);
        const signStatus = document.getElementById('sign-status');
        if (signStatus) signStatus.textContent = 'Camera access denied';
    }
}

function stopLiveCamera() {
    if (window.localStream) {
        window.localStream.getTracks().forEach(track => track.stop());
        window.localStream = null;
    }

    const selfVideo = document.getElementById('self-video');
    if (selfVideo) selfVideo.srcObject = null;

    const remoteVideo = document.getElementById('remote-video');
    if (remoteVideo) remoteVideo.srcObject = null;

    console.log('[WebRTC] Camera stopped');
}

// ── WebRTC Peer Connection ────────────────────────────────────────────

async function startWebRTCCall(roomId) {
    currentRoomId = roomId;

    // Connect signaling WebSocket
    connectSignaling(roomId);
}

function stopWebRTCCall() {
    // Close peer connection
    if (peerConnection) {
        peerConnection.close();
        peerConnection = null;
    }

    // Close data channel
    if (dataChannel) {
        dataChannel.close();
        dataChannel = null;
    }

    // Close signaling socket
    if (signalingSocket) {
        signalingSocket.close();
        signalingSocket = null;
    }

    window.remoteStream = null;
    currentRoomId = null;

    console.log('[WebRTC] Call ended, connections closed');
}

function createPeerConnection() {
    peerConnection = new RTCPeerConnection(RTC_CONFIG);

    // Add local tracks to the connection
    if (window.localStream) {
        window.localStream.getTracks().forEach(track => {
            peerConnection.addTrack(track, window.localStream);
        });
    }

    // Handle ICE candidates
    peerConnection.onicecandidate = (event) => {
        if (event.candidate && signalingSocket && signalingSocket.readyState === WebSocket.OPEN) {
            signalingSocket.send(JSON.stringify({
                type: 'ice_candidate',
                candidate: {
                    candidate: event.candidate.candidate,
                    sdpMid: event.candidate.sdpMid,
                    sdpMLineIndex: event.candidate.sdpMLineIndex,
                },
            }));
        }
    };

    // Handle remote tracks
    peerConnection.ontrack = (event) => {
        console.log('[WebRTC] Remote track received:', event.track.kind);

        if (!window.remoteStream) {
            window.remoteStream = new MediaStream();
        }
        window.remoteStream.addTrack(event.track);

        const remoteVideo = document.getElementById('remote-video');
        if (remoteVideo) {
            remoteVideo.srcObject = window.remoteStream;
        }
    };

    // Handle connection state changes
    peerConnection.onconnectionstatechange = () => {
        const state = peerConnection.connectionState;
        console.log('[WebRTC] Connection state:', state);

        const badge = document.getElementById('webrtc-badge');
        if (badge) {
            switch (state) {
                case 'connected':
                    badge.textContent = 'WebRTC ●';
                    badge.style.color = '#1D9E75';
                    break;
                case 'connecting':
                    badge.textContent = 'Connecting...';
                    badge.style.color = '#9b9baa';
                    break;
                case 'disconnected':
                case 'failed':
                    badge.textContent = 'Disconnected';
                    badge.style.color = '#e24b4a';
                    break;
            }
        }
    };

    // Create data channel for text relay
    dataChannel = peerConnection.createDataChannel('silentbridge-text', {
        ordered: true,
    });

    dataChannel.onopen = () => {
        console.log('[WebRTC] Data channel open');
    };

    dataChannel.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'text') {
                // Remote peer sent a recognized text
                addBubble('transcript-panel', 'them', msg.label || 'Peer', msg.text);
            }
        } catch (e) {
            // Plain text message
            addBubble('transcript-panel', 'them', 'Peer', event.data);
        }
    };

    // Handle incoming data channels from remote peer
    peerConnection.ondatachannel = (event) => {
        const incomingChannel = event.channel;
        incomingChannel.onmessage = (evt) => {
            try {
                const msg = JSON.parse(evt.data);
                if (msg.type === 'text') {
                    addBubble('transcript-panel', 'them', msg.label || 'Peer', msg.text);
                }
            } catch (e) {
                addBubble('transcript-panel', 'them', 'Peer', evt.data);
            }
        };
    };

    return peerConnection;
}

// ── Signaling WebSocket ───────────────────────────────────────────────

function connectSignaling(roomId) {
    const wsUrl = `${WS_BASE}/ws/signal/${roomId}/${clientId}`;
    console.log('[Signal] Connecting to:', wsUrl);

    try {
        signalingSocket = new WebSocket(wsUrl);
    } catch (e) {
        console.warn('[Signal] WebSocket connection failed (backend may not be running):', e);
        return;
    }

    signalingSocket.onopen = () => {
        console.log('[Signal] Connected to signaling server');
        // We only create the offer if we are the "Master" (the first one)
        // or just wait for peer_joined logic to handle it reliably.
        createPeerConnection();
    };

    signalingSocket.onmessage = async (event) => {
        const message = JSON.parse(event.data);
        await handleSignalingMessage(message);
    };

    signalingSocket.onerror = (err) => {
        console.warn('[Signal] WebSocket error (backend may not be running):', err);
    };

    signalingSocket.onclose = () => {
        console.log('[Signal] WebSocket closed');
    };
}

async function handleSignalingMessage(message) {
    if (!peerConnection) createPeerConnection();

    try {
        switch (message.type) {
            case 'peer_joined':
                console.log('[Signal] Peer joined, we will offer');
                isPolite = false; // The person already in the room becomes Master
                await createAndSendOffer();
                break;

            case 'offer':
                const offerCollision = (makingOffer || peerConnection.signalingState !== "stable");
                const ignoreOffer = !isPolite && offerCollision;
                if (ignoreOffer) {
                    console.log('[Signal] Ignoring collision offer');
                    return;
                }

                console.log('[Signal] Received offer, sending answer');
                await peerConnection.setRemoteDescription(new RTCSessionDescription({ type: 'offer', sdp: message.sdp }));
                const answer = await peerConnection.createAnswer();
                await peerConnection.setLocalDescription(answer);
                signalingSocket.send(JSON.stringify({
                    type: 'answer',
                    sdp: peerConnection.localDescription.sdp
                }));
                break;

            case 'answer':
                console.log('[Signal] Received answer');
                await peerConnection.setRemoteDescription(new RTCSessionDescription({ type: 'answer', sdp: message.sdp }));
                break;

            case 'ice_candidate':
                console.log('[Signal] Received ICE candidate');
                try {
                    await peerConnection.addIceCandidate(new RTCIceCandidate(message.candidate));
                } catch (e) {
                    if (!ignoreOffer) console.warn('[Signal] ICE failure:', e);
                }
                break;

            case 'chat':
                if (message.text) {
                    addBubble('transcript-panel', 'them', message.label || 'Peer', message.text);
                }
                break;
        }
    } catch (err) {
        console.error('[Signal] Message error:', err);
    }
}

async function createAndSendOffer() {
    if (!peerConnection) return;
    try {
        makingOffer = true;
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        signalingSocket.send(JSON.stringify({
            type: 'offer',
            sdp: peerConnection.localDescription.sdp
        }));
    } catch (e) {
        console.warn('[WebRTC] Offer error:', e);
    } finally {
        makingOffer = false;
    }
}

// ── Send text to peer via data channel ────────────────────────────────

function sendTextToPeer(text, label) {
    if (dataChannel && dataChannel.readyState === 'open') {
        dataChannel.send(JSON.stringify({
            type: 'text',
            text: text,
            label: label || 'Peer',
        }));
    }

    // Also send via signaling as fallback
    if (signalingSocket && signalingSocket.readyState === WebSocket.OPEN) {
        signalingSocket.send(JSON.stringify({
            type: 'chat',
            text: text,
            label: label || 'Peer',
        }));
    }
}

// ── Exported to window ────────────────────────────────────────────────

window.startLiveCamera = startLiveCamera;
window.stopLiveCamera = stopLiveCamera;
window.startWebRTCCall = startWebRTCCall;
window.stopWebRTCCall = stopWebRTCCall;
window.sendTextToPeer = sendTextToPeer;
