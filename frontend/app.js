/**
 * SilentBridge — Core Application Logic
 *
 * Handles screen navigation, role/language selection, call management,
 * transcript rendering, settings toggles, and incoming call overlay.
 * This preserves and extends all UI logic from the original prototype.
 */

// ── Global State ──────────────────────────────────────────────────────

const APP_SCREENS = ['splash', 'onboard', 'home', 'call', 'transcript-screen', 'settings', 'training'];
let prevScreen = 'home';
let callTimer = null;
let callSeconds = 0;
let callPeer = 'Rheya';
let currentRole = 'deaf';      // 'deaf' or 'speak'
let currentLang = 'en';        // 'en' or 'ta'
let ttsOn = true;
let micOn = true;
let camOn = true;
let targetRoomId = null;

// Live transcript buffer (for current call session)
let liveTranscript = [];

// ── Backend Configuration ─────────────────────────────────────────────
// Point to the Render backend even if the frontend is hosted on Vercel
const BACKEND_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? window.location.origin 
    : 'https://silentbridge-app.onrender.com';

const API_BASE = `${BACKEND_URL}/api`;
const WS_BASE = `${BACKEND_URL.replace(/^http/, 'ws')}`;

// ── Screen Navigation ─────────────────────────────────────────────────

function go(screenId) {
    if (screenId !== 'call') {
        prevScreen = document.querySelector('.screen.active')?.id || 'home';
    }

    APP_SCREENS.forEach(s => {
        const el = document.getElementById(s);
        if (el) el.classList.remove('active');
    });

    const target = document.getElementById(screenId);
    if (target) target.classList.add('active');

    // Screen-specific hooks
    if (screenId === 'call') {
        startCallSession();
    }
    if (screenId === 'transcript-screen') {
        populateTranscript();
    }
    if (screenId === 'training') {
        if (typeof initTrainingMode === 'function') initTrainingMode();
    }

    // Update bottom nav active state
    updateBottomNav(screenId);
}

function goBack() {
    go(prevScreen === 'call' ? 'call' : 'home');
}

function updateBottomNav(activeScreen) {
    const homeScreens = ['home'];
    const historyScreens = ['transcript-screen'];
    const settingsScreens = ['settings'];
    if (homeScreens.includes(activeScreen)) activeNav = 'nav-home';
    else if (historyScreens.includes(activeScreen)) activeNav = 'nav-history';
    else if (settingsScreens.includes(activeScreen)) activeNav = 'nav-settings';

    if (activeNav) {
        const el = document.getElementById(activeNav);
        if (el) el.classList.add('active');
    }
}

// ── Role & Language Selection ─────────────────────────────────────────

function selectRole(role) {
    currentRole = role;
    document.getElementById('role-deaf')?.classList.toggle('sel', role === 'deaf');
    document.getElementById('role-speak')?.classList.toggle('sel', role === 'speak');

    const modeEl = document.getElementById('mode-display');
    if (modeEl) {
        modeEl.textContent = (role === 'deaf' ? 'Sign language user' : 'Speaking user') + ' · ' + (currentLang === 'en' ? 'English' : 'Tamil');
    }
}

function selectLang(lang) {
    currentLang = lang;
    document.getElementById('lang-en')?.classList.toggle('sel', lang === 'en');
    document.getElementById('lang-ta')?.classList.toggle('sel', lang === 'ta');
}

// ── Call Management ───────────────────────────────────────────────────

function startCall(e, name, peerRole) {
    if (e) e.stopPropagation();
    targetRoomId = null;
    callPeer = name;
    document.getElementById('call-peer-name').textContent = name;
    liveTranscript = [];
    go('call');
}

async function startCallSession() {
    callSeconds = 0;
    if (callTimer) clearInterval(callTimer);

    callTimer = setInterval(() => {
        callSeconds++;
        const m = Math.floor(callSeconds / 60);
        const s = callSeconds % 60;
        const el = document.getElementById('call-timer');
        if (el) el.textContent = m + ':' + String(s).padStart(2, '0');
    }, 1000);

    // Start camera for self-view AND wait for it to be fully resolved
    if (typeof startLiveCamera === 'function') {
        await startLiveCamera();
    }

    // Start WebRTC connection only after camera handles track logic
    if (typeof startWebRTCCall === 'function') {
        const roomId = targetRoomId || ('room_' + Date.now());
        targetRoomId = roomId;
        
        // Detect if we are the person joining (Guest) vs Host
        const params = new URLSearchParams(window.location.search);
        if (params.has('room') && typeof isPolite !== 'undefined') {
            isPolite = true; // Guests are polite
        }
        
        startWebRTCCall(roomId);
    }

    // Start ML pipelines based on role
    if (currentRole === 'deaf') {
        if (typeof startSignRecognition === 'function') startSignRecognition();
    } else {
        if (typeof startSpeechRecognition === 'function') startSpeechRecognition();
    }
}

function endCall() {
    if (callTimer) clearInterval(callTimer);
    callTimer = null;

    // Stop ML pipelines
    if (typeof stopSignRecognition === 'function') stopSignRecognition();
    if (typeof stopSpeechRecognition === 'function') stopSpeechRecognition();

    // Stop camera
    if (typeof stopLiveCamera === 'function') stopLiveCamera();

    // Stop WebRTC
    if (typeof stopWebRTCCall === 'function') stopWebRTCCall();

    go('home');
}

// ── Incoming Call Overlay ─────────────────────────────────────────────

function showIncoming() {
    document.getElementById('incoming')?.classList.add('show');
}

function declineCall() {
    document.getElementById('incoming')?.classList.remove('show');
}

function acceptCall() {
    document.getElementById('incoming')?.classList.remove('show');
    const name = document.getElementById('incoming-name')?.textContent || 'Unknown';
    startCall(null, name, 'speak');
}

// ── Transcript Panel (Live Subtitles) ─────────────────────────────────

function addBubble(panelId, who, label, text) {
    const panel = document.getElementById(panelId);
    if (!panel) return;

    // Limit visible bubbles
    while (panel.children.length > 8) panel.removeChild(panel.firstChild);

    const wrap = document.createElement('div');
    wrap.className = 'bubble-wrap';
    wrap.style.alignItems = who === 'me' ? 'flex-end' : 'flex-start';

    const lbl = document.createElement('div');
    lbl.className = 'bubble-label';
    lbl.textContent = label;

    const bub = document.createElement('div');
    bub.className = 'bubble ' + who;
    bub.textContent = text;

    wrap.appendChild(lbl);
    wrap.appendChild(bub);
    panel.appendChild(wrap);
    panel.scrollTop = panel.scrollHeight;

    // Store in live transcript
    liveTranscript.push({ from: who, label: label, text: text, time: new Date().toISOString() });
}

/**
 * Called by ML pipeline when sign language is recognized.
 */
function onSignRecognized(text, confidence) {
    addBubble('transcript-panel', 'me', `You (sign → text) · ${Math.round(confidence * 100)}%`, text);

    // Send to peer via WebRTC data channel
    if (typeof sendTextToPeer === 'function') {
        sendTextToPeer(text, 'Sign → Text');
    }

    // TTS: speak the recognized text aloud
    if (ttsOn && 'speechSynthesis' in window) {
        const msg = new SpeechSynthesisUtterance(text);
        msg.rate = 1.0;
        msg.pitch = 1.0;
        window.speechSynthesis.speak(msg);
    }
}

/**
 * Called by speech pipeline when speech is transcribed.
 */
function onSpeechTranscribed(text, source) {
    const isMe = (source === 'browser' || source === 'whisper');
    const who = isMe ? 'me' : 'them';
    const label = isMe ? `You (${source} → text)` : `${callPeer} (speech → text)`;
    
    addBubble('transcript-panel', who, label, text);
    
    // If it's my own speech, broadcast it to the peer
    if (isMe && typeof sendTextToPeer === 'function') {
        sendTextToPeer(text, source === 'whisper' ? 'Whisper → Text' : 'Speech → Text');
    }
}

// ── Transcript History ────────────────────────────────────────────────

function populateTranscript() {
    if (liveTranscript.length === 0) return;

    const body = document.getElementById('ts-body');
    if (!body) return;

    // Remove previous live bubbles
    body.querySelectorAll('.live-bubble').forEach(e => e.remove());

    // Insert call header
    const sep = document.createElement('div');
    sep.className = 'ts-meta live-bubble';
    sep.textContent = `${callPeer} · Current call`;
    body.insertBefore(sep, body.firstChild);

    // Insert transcript bubbles in order
    liveTranscript.forEach(item => {
        const b = document.createElement('div');
        b.className = 'ts-bubble live-bubble ' + (item.from === 'me' ? 'me' : 'them');

        const lbl = document.createElement('div');
        lbl.className = 'ts-speaker';
        lbl.style.color = item.from === 'me' ? '#378ADD' : '#1D9E75';
        lbl.textContent = item.label || (item.from === 'me' ? 'You (sign → text)' : `${callPeer} (speech → text)`);
        b.appendChild(lbl);

        b.appendChild(document.createTextNode(item.text));
        body.insertBefore(b, sep.nextSibling);
    });
}

// ── Call Controls ─────────────────────────────────────────────────────

function toggleMic() {
    micOn = !micOn;
    const btn = document.getElementById('btn-mic');
    if (btn) btn.style.background = micOn ? 'rgba(255,255,255,.1)' : 'rgba(226,75,74,.3)';

    // Mute/unmute local audio track
    if (window.localStream) {
        window.localStream.getAudioTracks().forEach(t => { t.enabled = micOn; });
    }
}

function toggleCam() {
    camOn = !camOn;
    const btn = document.getElementById('btn-cam');
    if (btn) btn.style.background = camOn ? 'rgba(255,255,255,.1)' : 'rgba(226,75,74,.3)';

    // Enable/disable local video track
    if (window.localStream) {
        window.localStream.getVideoTracks().forEach(t => { t.enabled = camOn; });
    }
}

function toggleTTS() {
    ttsOn = !ttsOn;
    const label = document.getElementById('tts-label');
    if (label) label.textContent = ttsOn ? 'TTS on' : 'TTS off';
}

function toggleBtn(el) {
    el.classList.toggle('on');
}

function toggleLangSetting() {
    const el = document.getElementById('lang-display');
    if (el) el.textContent = el.textContent === 'English' ? 'Tamil' : 'English';
}

// ── Clock ─────────────────────────────────────────────────────────────

function updateClock() {
    const now = new Date();
    const h = now.getHours();
    const m = String(now.getMinutes()).padStart(2, '0');
    const el = document.getElementById('clock');
    if (el) el.textContent = `${h}:${m}`;
}

setInterval(updateClock, 10000);
updateClock();

// ── Greeting ──────────────────────────────────────────────────────────

function updateGreeting() {
    const h = new Date().getHours();
    let greeting = 'Good evening';
    if (h < 12) greeting = 'Good morning';
    else if (h < 17) greeting = 'Good afternoon';

    const el = document.getElementById('greeting-label');
    if (el) el.textContent = greeting;
}

updateGreeting();

// ── Initialize ────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    console.log('[SilentBridge] Application initialized');
    selectRole('deaf');
    
    const params = new URLSearchParams(window.location.search);
    if (params.has('room')) {
        setTimeout(() => {
            alert("Join link detected! Please enter your name to join the call.");
            const emailInput = document.getElementById('login-email');
            if (emailInput) emailInput.placeholder = "Enter your display name";
            const pwdInput = document.getElementById('login-password');
            if (pwdInput) pwdInput.style.display = "none";
        }, 200);
    }
});

// ── Meeting Link Flow ─────────────────────────────────────────────────
function continueFromOnboard() {
    if (targetRoomId) {
        callPeer = "Guest";
        document.getElementById('call-peer-name').textContent = callPeer;
        liveTranscript = [];
        go('call');
    } else {
        go('home');
    }
}

function startNewMeeting() {
    targetRoomId = 'room_' + Math.random().toString(36).substr(2, 9);
    callPeer = 'Guest';
    document.getElementById('call-peer-name').textContent = callPeer;
    liveTranscript = [];
    go('call');
    
    setTimeout(() => { copyMeetingLink(); }, 500);
}

function copyMeetingLink() {
    if (!targetRoomId) return;
    const url = window.location.origin + window.location.pathname + '?room=' + targetRoomId;
    
    if (navigator.clipboard) {
        navigator.clipboard.writeText(url).then(() => {
            alert('Meeting link copied! Share this link directly via WhatsApp: \n\n' + url);
        }).catch(err => {
            alert('Share this link: \n' + url);
        });
    } else {
        alert('Share this link: \n' + url);
    }
}

// ── Login Flow ────────────────────────────────────────────────────────
const handleLogin = () => {
    try {
        console.log('[Login] Handling sign-in request...');
        const emailInput = document.getElementById('login-email');
        const email = emailInput ? emailInput.value : "Guest";
        
        const params = new URLSearchParams(window.location.search);
        
        const usernameEl = document.getElementById('user-name-display');
        if (usernameEl) {
            usernameEl.textContent = email.split('@')[0] || "Guest";
        }
        
        if (params.has('room')) {
             targetRoomId = params.get('room');
             console.log('[Login] Room detected, routing to onboard lobby:', targetRoomId);
             
             // Update UI to make it feel like a Join Lobby
             const onboardTitle = document.querySelector('.onboard-title');
             if (onboardTitle) onboardTitle.textContent = "Join Meeting Lobby";
             
             const onboardBtn = document.querySelector('.onboard-body .next-btn');
             if (onboardBtn) onboardBtn.textContent = "Join Call Now →";
             
             go('onboard'); // Required to configure WebRTC preferences first
        } else {
             console.log('[Login] No room detected, routing to home');
             go('home');
        }
    } catch (err) {
        console.error('[Login] Critical failure:', err);
        alert("Login system error. Please refresh. Technical details: " + err.message);
    }
};

// Map to window for reliable HTML onclick access
window.handleLogin = handleLogin;

// ── Export Tools ──────────────────────────────────────────────────────
function shareTranscript() {
    let transcriptText = "SilentBridge Call Transcript:\n\n";
    
    const bubbles = document.querySelectorAll('#ts-body .ts-bubble, #ts-body .ts-meta');
    bubbles.forEach(el => {
        transcriptText += el.innerText.replace(/\n+/g, ': ') + "\n";
    });
    
    if (navigator.share) {
        navigator.share({
            title: 'SilentBridge Call Transcript',
            text: transcriptText,
        }).catch(err => console.error("Share failed:", err));
    } else if (navigator.clipboard) {
        navigator.clipboard.writeText(transcriptText).then(() => {
            alert('Transcript copied to clipboard!');
        }).catch(err => {
            alert('Failed to copy transcript');
        });
    } else {
        alert("Sharing not supported on this browser.");
    }
}

function exportPDF() {
    const element = document.getElementById('ts-body');
    if (!element) return;
    
    const opt = {
        margin:       10,
        filename:     'SilentBridge_Transcript.pdf',
        image:        { type: 'jpeg', quality: 0.98 },
        html2canvas:  { scale: 2 },
        jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' }
    };
    
    if (typeof html2pdf !== 'undefined') {
        html2pdf().set(opt).from(element).save();
    } else {
        alert("PDF export library not loaded. Please try again later.");
    }
}
