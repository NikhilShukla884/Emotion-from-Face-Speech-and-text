const API = "http://127.0.0.1:5000/api/emotion";

// ── Emotion colors ──────────────────────────────────────────────────────────
const EMOTION_COLORS = {
  happy:     "#22d3a5", Happy:     "#22d3a5", hap: "#22d3a5",
  sad:       "#60a5fa", Sad:       "#60a5fa",
  angry:     "#ef4444", Angry:     "#ef4444", ang: "#ef4444",
  fearful:   "#a78bfa", Fearful:   "#a78bfa", fear: "#a78bfa", Fear: "#a78bfa",
  surprised: "#f59e0b", Surprised: "#f59e0b", surprise: "#f59e0b", Surprise: "#f59e0b",
  disgust:   "#f97316", Disgust:   "#f97316",
  neutral:   "#8892b0", Neutral:   "#8892b0", neu: "#8892b0",
  calm:      "#34d399", Calm:      "#34d399",
};

const EMOTION_ICONS = {
  happy:"😄",     Happy:"😄",     hap:"😄",
  sad:"😢",       Sad:"😢",
  angry:"😠",     Angry:"😠",     ang:"😠",
  fearful:"😨",   Fearful:"😨",   fear:"😨",  Fear:"😨",
  surprised:"😲", Surprised:"😲", surprise:"😲", Surprise:"😲",
  disgust:"🤢",   Disgust:"🤢",
  neutral:"😐",   Neutral:"😐",   neu:"😐",
  calm:"😌",      Calm:"😌",
};

// ── Tab switching ────────────────────────────────────────────────────────────
document.querySelectorAll(".nav-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// ── Loading helpers ──────────────────────────────────────────────────────────
function showLoading(msg = "Analyzing...") {
  document.getElementById("loadingText").textContent = msg;
  document.getElementById("loadingOverlay").classList.add("active");
}
function hideLoading() {
  document.getElementById("loadingOverlay").classList.remove("active");
}

// ── Result renderers ─────────────────────────────────────────────────────────
function emotionBars(scores) {
  if (!scores || !Object.keys(scores).length) return "";
  const sorted = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  return sorted.map(([label, val]) => {
    const pct = Math.round((val > 1 ? val : val * 100));
    const color = EMOTION_COLORS[label.toLowerCase()] || "#6c63ff";
    return `
      <div class="emotion-bar-row">
        <span class="emotion-bar-label">${label}</span>
        <div class="emotion-bar-track">
          <div class="emotion-bar-fill" style="width:${pct}%;background:${color}"></div>
        </div>
        <span class="emotion-bar-pct">${pct}%</span>
      </div>`;
  }).join("");
}

function emotionBadge(emotion) {
  const color = EMOTION_COLORS[emotion?.toLowerCase()] || "#6c63ff";
  const icon  = EMOTION_ICONS[emotion?.toLowerCase()] || "🎭";
  return `<div class="emotion-badge" style="background:${color}22;color:${color};border:1px solid ${color}44">
    ${icon} ${emotion}
  </div>`;
}

function modelBlock(title, data, dotColor) {
  if (data?.error) {
    return `<div class="model-block">
      <div class="model-label"><span class="dot" style="background:${dotColor||'#ef4444'}"></span>${title}</div>
      <div class="error-msg"><i class="fa-solid fa-triangle-exclamation"></i>${data.error}</div>
    </div>`;
  }
  return `<div class="model-block">
    <div class="model-label"><span class="dot" style="background:${dotColor||'#6c63ff'}"></span>${title}</div>
    ${emotionBadge(data.emotion)}
    ${emotionBars(data.scores)}
  </div>`;
}

// ── FACE ─────────────────────────────────────────────────────────────────────
let stream = null;
let uploadedImageData = null;
const video   = document.getElementById("video");
const canvas  = document.getElementById("canvas");
const overlay = document.getElementById("camera-overlay");

// Source toggle
document.getElementById("toggleCamera").addEventListener("click", () => {
  document.getElementById("cameraView").style.display = "block";
  document.getElementById("uploadView").style.display = "none";
  document.getElementById("toggleCamera").classList.add("active");
  document.getElementById("toggleUpload").classList.remove("active");
});

document.getElementById("toggleUpload").addEventListener("click", () => {
  document.getElementById("cameraView").style.display = "none";
  document.getElementById("uploadView").style.display = "block";
  document.getElementById("toggleUpload").classList.add("active");
  document.getElementById("toggleCamera").classList.remove("active");
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    video.srcObject = null;
    overlay.classList.remove("hidden");
    document.getElementById("startCamera").disabled = false;
    document.getElementById("captureBtn").disabled = true;
    document.getElementById("stopCamera").disabled = true;
  }
});

// Image upload
document.getElementById("imageUpload").addEventListener("change", e => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    uploadedImageData = ev.target.result;
    const previewImg = document.getElementById("uploadPreviewImg");
    previewImg.src = uploadedImageData;
    previewImg.classList.add("visible");
    document.getElementById("imageUploadArea").classList.add("has-image");
    document.getElementById("analyzeUploadBtn").disabled = false;
  };
  reader.readAsDataURL(file);
});

document.getElementById("analyzeUploadBtn").addEventListener("click", async () => {
  if (!uploadedImageData) return;
  showPreview(uploadedImageData);
  await analyzeImage(uploadedImageData);
});

// Camera
document.getElementById("startCamera").addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    overlay.classList.add("hidden");
    document.getElementById("stopCamera").disabled = false;
    document.getElementById("startCamera").disabled = true;
    // Only enable capture once video dimensions are known
    video.onloadedmetadata = () => {
      document.getElementById("captureBtn").disabled = false;
    };
  } catch {
    alert("Camera access denied or unavailable.");
  }
});

document.getElementById("stopCamera").addEventListener("click", () => {
  stream?.getTracks().forEach(t => t.stop());
  video.srcObject = null;
  overlay.classList.remove("hidden");
  document.getElementById("captureBtn").disabled = true;
  document.getElementById("stopCamera").disabled = true;
  document.getElementById("startCamera").disabled = false;
});

document.getElementById("captureBtn").addEventListener("click", async () => {
  if (!video.videoWidth || !video.videoHeight) {
    alert("Camera not ready yet. Please wait a moment.");
    return;
  }
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  const imageData = canvas.toDataURL("image/jpeg", 0.9);
  showPreview(imageData);
  await analyzeImage(imageData);
});

function showPreview(src) {
  const preview = document.getElementById("capturedPreview");
  document.getElementById("previewImg").src = src;
  preview.style.display = "block";
}

async function analyzeImage(imageData) {
  showLoading("Analyzing facial emotions...");
  try {
    const res  = await fetch(`${API}/face`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageData }),
    });
    const data = await res.json();
    renderFaceResults(data);
  } catch {
    renderError("face-results", "Failed to connect to backend.");
  } finally {
    hideLoading();
  }
}

function renderFaceResults(data) {
  const panel = document.getElementById("face-results");
  panel.innerHTML = modelBlock("Facial Emotion", data.primary_model, "#6c63ff");
}

// ── TEXT ──────────────────────────────────────────────────────────────────────
const textInput = document.getElementById("textInput");
textInput.addEventListener("input", () => {
  document.getElementById("charCount").textContent = `${textInput.value.length} characters`;
});

document.getElementById("analyzeText").addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (!text) return;

  showLoading("Analyzing text emotions...");
  try {
    const res  = await fetch(`${API}/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    renderTextResults(data);
  } catch {
    renderError("text-results", "Failed to connect to backend.");
  } finally {
    hideLoading();
  }
});

function renderTextResults(data) {
  const panel = document.getElementById("text-results");
  const chipClass = data.sentiment === "Positive" ? "chip-positive"
                  : data.sentiment === "Negative" ? "chip-negative" : "chip-neutral";

  const polarity = data.polarity ?? 0;
  const pct      = Math.round(((polarity + 1) / 2) * 100);
  const fillColor = polarity > 0.1 ? "#22d3a5" : polarity < -0.1 ? "#ef4444" : "#8892b0";
  const fillLeft  = polarity >= 0 ? "50%" : `${pct}%`;
  const fillWidth = `${Math.abs(polarity) * 50}%`;

  panel.innerHTML = `
    <div class="model-block">
      <div class="model-label"><span class="dot"></span>Dominant Emotion</div>
      ${emotionBadge(data.emotion)}
      ${emotionBars(data.emotion_scores)}
    </div>
    <hr class="divider"/>
    <div class="model-block">
      <div class="model-label"><span class="dot" style="background:#f59e0b"></span>Sentiment</div>
      <div class="sentiment-row">
        <span class="sentiment-chip ${chipClass}">${data.sentiment}</span>
        <span class="sentiment-chip chip-neutral">Subjectivity: ${Math.round((data.subjectivity??0)*100)}%</span>
      </div>
      <div class="polarity-bar-wrapper">
        <div class="polarity-label">Polarity: ${polarity > 0 ? "+" : ""}${polarity}</div>
        <div class="polarity-track">
          <div class="polarity-fill" style="left:${fillLeft};width:${fillWidth};background:${fillColor}"></div>
        </div>
      </div>
    </div>
  `;
}

// ── SPEECH ────────────────────────────────────────────────────────────────────
let mediaRecorder = null, audioChunks = [], audioBlob = null;
let timerInterval = null, timerSeconds = 0;
let audioCtx = null, analyser = null, animFrame = null;

const recordBtn    = document.getElementById("recordBtn");
const analyzeAudio = document.getElementById("analyzeAudio");
const waveCanvas   = document.getElementById("waveform");
const waveCtx      = waveCanvas.getContext("2d");
const wfPlaceholder = document.getElementById("waveform-placeholder");

/**
 * Converts an AudioBuffer to a proper WAV Blob.
 * This guarantees the backend always receives a real WAV file.
 */
function audioBufferToWavBlob(buffer) {
  const numChannels = buffer.numberOfChannels;
  const sampleRate  = buffer.sampleRate;
  const samples     = buffer.getChannelData(0); // mono
  const dataLen     = samples.length * 2;       // 16-bit
  const wavBuffer   = new ArrayBuffer(44 + dataLen);
  const view        = new DataView(wavBuffer);

  function writeStr(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeStr(0, "RIFF");
  view.setUint32(4,  36 + dataLen, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);          // PCM chunk size
  view.setUint16(20, 1,  true);          // PCM format
  view.setUint16(22, 1,  true);          // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2,  true);          // block align
  view.setUint16(34, 16, true);          // bits per sample
  writeStr(36, "data");
  view.setUint32(40, dataLen, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }
  return new Blob([wavBuffer], { type: "audio/wav" });
}

/**
 * Decodes a recorded webm/ogg blob into a proper WAV blob using AudioContext.
 */
async function toWavBlob(rawBlob) {
  const arrayBuffer = await rawBlob.arrayBuffer();
  const decodeCtx   = new AudioContext({ sampleRate: 16000 });
  const audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer);
  await decodeCtx.close();
  return audioBufferToWavBlob(audioBuffer);
}

recordBtn.addEventListener("click", async () => {
  if (mediaRecorder?.state === "recording") {
    mediaRecorder.stop();
    return;
  }
  try {
    const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(micStream);

    audioCtx  = new AudioContext();
    analyser  = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    audioCtx.createMediaStreamSource(micStream).connect(analyser);
    wfPlaceholder.classList.add("hidden");
    drawWaveform();

    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      stopTimer();
      cancelAnimationFrame(animFrame);
      micStream.getTracks().forEach(t => t.stop());
      audioCtx?.close();
      recordBtn.innerHTML = '<i class="fa-solid fa-microphone"></i> Start Recording';
      recordBtn.classList.remove("recording");

      // Convert raw browser recording to proper WAV
      const rawBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
      try {
        audioBlob = await toWavBlob(rawBlob);
      } catch {
        audioBlob = rawBlob; // fallback
      }
      analyzeAudio.disabled = false;
    };

    mediaRecorder.start();
    startTimer();
    recordBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Recording';
    recordBtn.classList.add("recording");
  } catch {
    alert("Microphone access denied or unavailable.");
  }
});

function drawWaveform() {
  animFrame = requestAnimationFrame(drawWaveform);
  const buf = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(buf);
  const W = waveCanvas.width  = waveCanvas.offsetWidth;
  const H = waveCanvas.height = waveCanvas.offsetHeight;
  waveCtx.clearRect(0, 0, W, H);
  waveCtx.strokeStyle = "#6c63ff";
  waveCtx.lineWidth   = 2;
  waveCtx.beginPath();
  const slice = W / buf.length;
  let x = 0;
  buf.forEach((v, i) => {
    const y = (v / 128) * (H / 2);
    i === 0 ? waveCtx.moveTo(x, y) : waveCtx.lineTo(x, y);
    x += slice;
  });
  waveCtx.stroke();
}

function startTimer() {
  timerSeconds = 0;
  document.getElementById("recordTimer").style.display = "flex";
  timerInterval = setInterval(() => {
    timerSeconds++;
    const m = Math.floor(timerSeconds / 60);
    const s = String(timerSeconds % 60).padStart(2, "0");
    document.getElementById("timerDisplay").textContent = `${m}:${s}`;
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
  document.getElementById("recordTimer").style.display = "none";
}

document.getElementById("audioUpload").addEventListener("change", e => {
  const file = e.target.files[0];
  if (!file) return;
  // Uploaded file is already a valid WAV — use directly, no conversion needed
  audioBlob = new Blob([file], { type: "audio/wav" });
  analyzeAudio.disabled = false;
  wfPlaceholder.classList.add("hidden");
  wfPlaceholder.innerHTML = `<i class="fa-solid fa-file-audio"></i><span>${file.name}</span>`;
});

analyzeAudio.addEventListener("click", async () => {
  if (!audioBlob) return;
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.wav");

  showLoading("Analyzing speech emotions...");
  try {
    const res  = await fetch(`${API}/speech`, { method: "POST", body: formData });
    const data = await res.json();
    renderSpeechResults(data);
  } catch {
    renderError("speech-results", "Failed to connect to backend.");
  } finally {
    hideLoading();
  }
});

function renderSpeechResults(data) {
  const panel  = document.getElementById("speech-results");
  const result = data.primary_model;

  if (result?.error) {
    renderError("speech-results", result.error);
    return;
  }

  const color   = EMOTION_COLORS[result.emotion?.toLowerCase()] || "#6c63ff";
  const icon    = EMOTION_ICONS[result.emotion?.toLowerCase()]  || "🎤";
  const sorted  = Object.entries(result.scores).sort((a, b) => b[1] - a[1]);

  panel.innerHTML = `
    <div class="model-block">
      <div class="model-label"><span class="dot" style="background:#6c63ff"></span>Detected Emotion</div>
      <div class="emotion-badge" style="background:${color}22;color:${color};border:1px solid ${color}44;margin-bottom:16px">
        ${icon} ${result.emotion}
      </div>
      <div class="model-label" style="margin-top:8px"><span class="dot" style="background:#a78bfa"></span>All Emotions</div>
      ${sorted.map(([label, val]) => {
        const pct   = Math.round(val * 100);
        const c     = EMOTION_COLORS[label.toLowerCase()] || "#6c63ff";
        const ic    = EMOTION_ICONS[label.toLowerCase()]  || "🎤";
        return `
          <div class="emotion-bar-row">
            <span class="emotion-bar-label">${ic} ${label}</span>
            <div class="emotion-bar-track">
              <div class="emotion-bar-fill" style="width:${pct}%;background:${c}"></div>
            </div>
            <span class="emotion-bar-pct">${pct}%</span>
          </div>`;
      }).join("")}
    </div>
  `;
}

// ── Shared error renderer ────────────────────────────────────────────────────
function renderError(panelId, msg) {
  document.getElementById(panelId).innerHTML = `
    <div class="results-placeholder">
      <div class="error-msg"><i class="fa-solid fa-triangle-exclamation"></i>${msg}</div>
    </div>`;
}
