/**
 * SilentBridge — Core Application Logic (Fixed)
 *
 * Fixes applied:
 *   1. Room IDs use crypto.randomUUID() — collision-safe
 *   2. isPolite set reliably via setGuestMode() before call starts
 *   3. handleLogin no longer tries to reference isPolite before webrtc.js sets it up
 *   4. NO_SIGN is never rendered as a transcript bubble
 *   5. updateBottomNav fixed — activeNav declared at proper scope
 *   6. endCall properly clears targetRoomId so next call gets a fresh room
 */

// ── Global State ────────────────────────────────────────────────────────────
const APP_SCREENS = ['splash', 'onboard', 'home', 'call', 'transcript-screen', 'settings', 'training'];
let prevScreen   = 'home';
let callTimer    = null;
let callSeconds  = 0;
let callPeer     = 'Guest';
let currentRole  = 'deaf';   // 'deaf' | 'speak'
let currentLang  = 'en';     // 'en'   | 'ta'
let ttsOn        = true;
let micOn        = true;
let camOn        = true;
let targetRoomId = null;

// Live transcript buffer for the current call session
let liveTranscript = [];

// ── Backend Configuration ───────────────────────────────────────────────────
// Automatically uses the current origin — works on localhost AND on Render
// (no need to hard-code a specific Render subdomain).
const BACKEND_URL = window.location.origin;
const API_BASE    = `${BACKEND_URL}/api`;
// ws:// on localhost, wss:// on HTTPS (Render)
const WS_BASE     = `${BACKEND_URL.replace(/^http/, 'ws')}`;

// ── Screen Navigation ───────────────────────────────────────────────────────

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

    if (screenId === 'call')              startCallSession();
    if (screenId === 'transcript-screen') populateTranscript();
    if (screenId === 'training' && typeof initTrainingMode === 'function') initTrainingMode();

    updateBottomNav(screenId);
}

function goBack() {
    go(prevScreen === 'call' ? 'call' : 'home');
}

function updateBottomNav(activeScreen) {
    // Remove all active states first
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));

    let navId = null;
    if (activeScreen === 'home')              navId = 'nav-home';
    else if (activeScreen === 'transcript-screen') navId = 'nav-history';
    else if (activeScreen === 'settings')     navId = 'nav-settings';

    if (navId) document.getElementById(navId)?.classList.add('active');
}

// ── Role & Language ─────────────────────────────────────────────────────────

function selectRole(role) {
    currentRole = role;
    document.getElementById('role-deaf')?.classList.toggle('sel', role === 'deaf');
    document.getElementById('role-speak')?.classList.toggle('sel', role === 'speak');

    const modeEl = document.getElementById('mode-display');
    if (modeEl) {
        modeEl.textContent = (role === 'deaf' ? 'Sign language user' : 'Speaking user')
            + ' · ' + (currentLang === 'en' ? 'English' : 'Tamil');
    }

    const settingsRole = document.getElementById('settings-role-display');
    if (settingsRole) {
        settingsRole.textContent = role === 'deaf' ? 'Sign language user' : 'Speaking user';
    }
}

function selectLang(lang) {
    currentLang = lang;
    document.getElementById('lang-en')?.classList.toggle('sel', lang === 'en');
    document.getElementById('lang-ta')?.classList.toggle('sel', lang === 'ta');
}

// ── UUID helper ─────────────────────────────────────────────────────────────
function generateRoomId() {
    // Use crypto.randomUUID() where available (all modern browsers)
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
        return crypto.randomUUID();
    }
    // Polyfill for older browsers
    return 'xxxx-xxxx-4xxx-yxxx'.replace(/[xy]/g, c => {
        const r = Math.random() * 16 | 0;
        return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
}

// ── Call Management ─────────────────────────────────────────────────────────

function startCall(e, name, peerRole) {
    if (e) e.stopPropagation();
    targetRoomId = null;   // Fresh room for direct calls
    callPeer     = name;
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
        if (el) el.textContent = `${m}:${String(s).padStart(2, '0')}`;
    }, 1000);

    // Start camera first
    if (typeof startLiveCamera === 'function') await startLiveCamera();

    // Determine room + politeness BEFORE starting WebRTC
    const params = new URLSearchParams(window.location.search);
    if (!targetRoomId) {
        // Fresh call — generate UUID room
        targetRoomId = generateRoomId();
    }

    // Fix: set isPolite cleanly here where webrtc.js is guaranteed loaded
    if (params.has('room') && typeof isPolite !== 'undefined') {
        // Guest joining via link → polite peer (answerer)
        isPolite = true;
    } else if (typeof isPolite !== 'undefined') {
        // Host creating meeting → impolite peer (offerer)
        isPolite = false;
    }

    // Start WebRTC
    if (typeof startWebRTCCall === 'function') {
        startWebRTCCall(targetRoomId);
    }

    // Start the relevant ML pipeline based on user role
    if (currentRole === 'deaf') {
        if (typeof startSignRecognition === 'function') startSignRecognition();
    } else {
        if (typeof startSpeechRecognition === 'function') startSpeechRecognition();
    }
}

function endCall() {
    if (callTimer) { clearInterval(callTimer); callTimer = null; }

    if (typeof stopSignRecognition  === 'function') stopSignRecognition();
    if (typeof stopSpeechRecognition === 'function') stopSpeechRecognition();
    if (typeof stopLiveCamera        === 'function') stopLiveCamera();
    if (typeof stopWebRTCCall        === 'function') stopWebRTCCall();

    targetRoomId = null;   // Clear so next call gets fresh room
    go('home');
}

// ── Incoming Call Overlay ───────────────────────────────────────────────────

function showIncoming() { document.getElementById('incoming')?.classList.add('show'); }
function declineCall()  { document.getElementById('incoming')?.classList.remove('show'); }

function acceptCall() {
    document.getElementById('incoming')?.classList.remove('show');
    const name = document.getElementById('incoming-name')?.textContent || 'Unknown';
    startCall(null, name, 'speak');
}

// ── Transcript Bubbles ──────────────────────────────────────────────────────

function addBubble(panelId, who, label, text) {
    // Fix: never render NO_SIGN or UNKNOWN
    if (!text || text === 'NO_SIGN' || text === 'UNKNOWN') return;

    const panel = document.getElementById(panelId);
    if (!panel) return;

    while (panel.children.length > 8) panel.removeChild(panel.firstChild);

    const wrap = document.createElement('div');
    wrap.className = 'bubble-wrap';
    wrap.style.alignItems = who === 'me' ? 'flex-end' : 'flex-start';

    const lbl = document.createElement('div');
    lbl.className   = 'bubble-label';
    lbl.textContent = label;

    const bub = document.createElement('div');
    bub.className   = 'bubble ' + who;
    bub.textContent = text;

    wrap.appendChild(lbl);
    wrap.appendChild(bub);
    panel.appendChild(wrap);
    panel.scrollTop = panel.scrollHeight;

    liveTranscript.push({ from: who, label, text, time: new Date().toISOString() });
}

// Called by ML pipeline when sign is recognized
function onSignRecognized(text, confidence) {
    if (!text || text === 'NO_SIGN') return;
    addBubble('transcript-panel', 'me', `You (sign → text) · ${Math.round(confidence * 100)}%`, text);

    // Flash the status banner with the recognized sign
    const statusEl = document.getElementById('sign-status');
    if (statusEl) {
        statusEl.textContent = `✋ ${text}  (${Math.round(confidence * 100)}%)`;
        statusEl.style.color = '#5fffb8';
        // Reset after 2.5s
        clearTimeout(statusEl._resetTimer);
        statusEl._resetTimer = setTimeout(() => {
            statusEl.textContent = 'AI Engine active · Detecting gestures…';
            statusEl.style.color = '';
        }, 2500);
    }

    // Show prominent subtitle overlay on the video for the peer to see
    showSubtitleOverlay(text, 'sign');

    if (typeof sendTextToPeer === 'function') sendTextToPeer(text, 'Sign → Text');

    if (ttsOn && 'speechSynthesis' in window) {
        const utt = new SpeechSynthesisUtterance(text);
        utt.rate = 1.0; utt.pitch = 1.0;
        window.speechSynthesis.speak(utt);
    }
}

// Called by speech pipeline when audio is transcribed
function onSpeechTranscribed(text, source) {
    if (!text || text.trim() === '') return;
    const isMe  = (source === 'browser' || source === 'whisper');
    const who   = isMe ? 'me' : 'them';
    const label = isMe
        ? `You (speech → text)`
        : `${callPeer} (speech → text)`;

    addBubble('transcript-panel', who, label, text);

    // Show prominent subtitle overlay on the video
    showSubtitleOverlay(text, 'speech');

    if (isMe && typeof sendTextToPeer === 'function') {
        sendTextToPeer(text, 'Speech → Text');
    }
}

// ── Subtitle Overlay (large text on video) ──────────────────────────────────
let _subtitleTimer = null;

function showSubtitleOverlay(text, type) {
    let el = document.getElementById('subtitle-overlay');
    if (!el) {
        el = document.createElement('div');
        el.id = 'subtitle-overlay';
        el.className = 'subtitle-overlay';
        const callScreen = document.getElementById('call');
        if (callScreen) callScreen.appendChild(el);
    }
    el.textContent = text;
    el.className = 'subtitle-overlay ' + (type === 'sign' ? 'subtitle-sign' : 'subtitle-speech');
    el.style.display = 'block';
    el.style.opacity = '1';

    clearTimeout(_subtitleTimer);
    _subtitleTimer = setTimeout(() => {
        el.style.opacity = '0';
        setTimeout(() => { el.style.display = 'none'; }, 400);
    }, 4000);
}

// ── Transcript History ──────────────────────────────────────────────────────

function populateTranscript() {
    if (liveTranscript.length === 0) return;
    const body = document.getElementById('ts-body');
    if (!body) return;

    body.querySelectorAll('.live-bubble').forEach(e => e.remove());

    const sep = document.createElement('div');
    sep.className   = 'ts-meta live-bubble';
    sep.textContent = `${callPeer} · Current call`;
    body.insertBefore(sep, body.firstChild);

    liveTranscript.forEach(item => {
        const b   = document.createElement('div');
        b.className = 'ts-bubble live-bubble ' + (item.from === 'me' ? 'me' : 'them');

        const lbl = document.createElement('div');
        lbl.className   = 'ts-speaker';
        lbl.style.color = item.from === 'me' ? '#378ADD' : '#1D9E75';
        lbl.textContent = item.label || (item.from === 'me' ? 'You (sign → text)' : `${callPeer} (speech → text)`);
        b.appendChild(lbl);
        b.appendChild(document.createTextNode(item.text));
        body.insertBefore(b, sep.nextSibling);
    });
}

// ── Call Controls ───────────────────────────────────────────────────────────

function toggleMic() {
    micOn = !micOn;
    const btn = document.getElementById('btn-mic');
    if (btn) btn.style.background = micOn ? 'rgba(255,255,255,.1)' : 'rgba(226,75,74,.3)';
    window.localStream?.getAudioTracks().forEach(t => { t.enabled = micOn; });
}

function toggleCam() {
    camOn = !camOn;
    const btn = document.getElementById('btn-cam');
    if (btn) btn.style.background = camOn ? 'rgba(255,255,255,.1)' : 'rgba(226,75,74,.3)';
    window.localStream?.getVideoTracks().forEach(t => { t.enabled = camOn; });
}

function toggleTTS() {
    ttsOn = !ttsOn;
    const label = document.getElementById('tts-label');
    if (label) label.textContent = ttsOn ? 'TTS on' : 'TTS off';
}

function toggleBtn(el) { el.classList.toggle('on'); }

function toggleLangSetting() {
    const el = document.getElementById('lang-display');
    if (el) el.textContent = el.textContent === 'English' ? 'Tamil' : 'English';
}

// ── Clock & Greeting ────────────────────────────────────────────────────────

function updateClock() {
    const now = new Date();
    const el  = document.getElementById('clock');
    if (el) el.textContent = `${now.getHours()}:${String(now.getMinutes()).padStart(2, '0')}`;
}
setInterval(updateClock, 10000);
updateClock();

function updateGreeting() {
    const h  = new Date().getHours();
    const el = document.getElementById('greeting-label');
    if (!el) return;
    if (h < 12)      el.textContent = 'Good morning';
    else if (h < 17) el.textContent = 'Good afternoon';
    else             el.textContent = 'Good evening';
}
updateGreeting();

// ── Login Flow ──────────────────────────────────────────────────────────────

const handleLogin = () => {
    try {
        const emailInput = document.getElementById('login-email');
        const email      = emailInput?.value?.trim() || 'Guest';

        // Set display name
        const nameEl = document.getElementById('user-name-display');
        if (nameEl) nameEl.textContent = email.split('@')[0] || 'Guest';

        const params = new URLSearchParams(window.location.search);

        if (params.has('room')) {
            // Guest joining via shared link
            targetRoomId = params.get('room');
            callPeer     = 'Host';
            console.log('[Login] Joining room:', targetRoomId);

            // Customise onboard screen for join flow
            const title = document.querySelector('.onboard-title');
            const btn   = document.querySelector('.onboard-body .next-btn');
            if (title) title.textContent = 'Join Meeting';
            if (btn)   btn.textContent   = 'Join Call →';

            go('onboard');
        } else {
            console.log('[Login] No room param — going home');
            go('home');
        }
    } catch (err) {
        console.error('[Login] Error:', err);
        go('home');
    }
};
window.handleLogin = handleLogin;

// ── Meeting Link ────────────────────────────────────────────────────────────

function continueFromOnboard() {
    if (targetRoomId) {
        document.getElementById('call-peer-name').textContent = callPeer;
        liveTranscript = [];
        go('call');
    } else {
        go('home');
    }
}

function startNewMeeting() {
    targetRoomId = generateRoomId();
    callPeer     = 'Guest';
    document.getElementById('call-peer-name').textContent = callPeer;
    liveTranscript = [];
    go('call');
    // NOTE: WhatsApp share opens only when the Share button is tapped
    // (not auto-opened here — that was breaking camera permission flow)
    setTimeout(() => showToast('Tap ‘Share’ to invite someone via WhatsApp 💬'), 1200);
}

function copyMeetingLink() {
    if (!targetRoomId) return;
    const url = `${window.location.origin}${window.location.pathname}?room=${targetRoomId}`;

    const message = `Join my SilentBridge call! 🤝\n\nClick this link to join:\n${url}\n\n_SilentBridge — Real-time sign language & speech communication_`;

    // Copy to clipboard first
    if (navigator.clipboard) {
        navigator.clipboard.writeText(url).catch(() => {});
    }

    // Open WhatsApp share dialog (works on mobile and desktop)
    const waUrl = `https://wa.me/?text=${encodeURIComponent(message)}`;
    window.open(waUrl, '_blank', 'noopener,noreferrer');

    showToast('Opening WhatsApp to share link 💬');
}

function shareMeetingLinkFallback() {
    if (!targetRoomId) return;
    const url = `${window.location.origin}${window.location.pathname}?room=${targetRoomId}`;
    if (navigator.share) {
        navigator.share({ title: 'Join SilentBridge Call', url })
            .catch(e => console.log('[Share] cancelled or failed:', e));
    } else {
        navigator.clipboard?.writeText(url).then(() => showToast('Link copied: ' + url));
    }
}

// ── Toast Notification ──────────────────────────────────────────────────────

function showToast(message) {
    let toast = document.getElementById('sb-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id        = 'sb-toast';
        toast.className = 'sb-toast';
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.classList.add('show');
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => toast.classList.remove('show'), 3500);
}

// ── Initialize ──────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    console.log('[SilentBridge] Application initialized');
    selectRole('deaf');
    updateGreeting();

    // Auto-detect room join from URL
    const params = new URLSearchParams(window.location.search);
    if (params.has('room')) {
        // Pre-fill the join UI
        const emailInput = document.getElementById('login-email');
        const pwdInput   = document.getElementById('login-password');
        if (emailInput) emailInput.placeholder = 'Your display name';
        if (pwdInput)   pwdInput.style.display  = 'none';

        const loginBtn = document.getElementById('btn-login');
        if (loginBtn) loginBtn.textContent = 'Join Meeting →';
    }
});

// ── Export & Share ──────────────────────────────────────────────────────────

function shareTranscript() {
    let text = 'SilentBridge Call Transcript\n\n';
    document.querySelectorAll('#ts-body .ts-bubble, #ts-body .ts-meta').forEach(el => {
        text += el.innerText.replace(/\n+/g, ': ') + '\n';
    });

    if (navigator.share) {
        navigator.share({ title: 'SilentBridge Transcript', text }).catch(console.error);
    } else if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => showToast('Transcript copied!'));
    }
}

function exportPDF() {
    const el = document.getElementById('ts-body');
    if (!el) return;
    if (typeof html2pdf !== 'undefined') {
        html2pdf().set({
            margin: 10, filename: 'SilentBridge_Transcript.pdf',
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: { scale: 2 },
            jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
        }).from(el).save();
    } else {
        showToast('PDF library not loaded yet. Try again.');
    }
}
