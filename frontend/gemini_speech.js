const GEMINI_MODEL = "gemini-2.5-flash";

const API_KEYS = [
  "YOUR GEMINI API KEY(s)",
  "",
  "",
  "",
  "",
];

const EMOTION_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"];

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result.split(",")[1]);
    reader.onerror   = reject;
    reader.readAsDataURL(blob);
  });
}

async function tryWithKey(apiKey, base64Audio, mimeType, prompt) {
  const { GoogleGenAI } = window.GoogleGenAI_SDK;
  const ai = new GoogleGenAI({ apiKey });

  const response = await ai.models.generateContent({
    model: GEMINI_MODEL,
    contents: [{
      parts: [
        { text: prompt },
        { inlineData: { mimeType, data: base64Audio } },
      ],
    }],
  });

  const text  = response.text.trim();
  const clean = text.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
  return JSON.parse(clean);
}

async function analyzeAudioWithGemini(audioBlob) {
  const base64Audio = await blobToBase64(audioBlob);
  const mimeType    = audioBlob.type || "audio/webm";

  const prompt = `You are a speech emotion recognition system.
Analyze the audio clip and return ONLY a valid JSON object with this exact structure, no markdown, no explanation:
{
  "emotion": "<dominant emotion>",
  "scores": {
    "neutral": <0-1>,
    "calm": <0-1>,
    "happy": <0-1>,
    "sad": <0-1>,
    "angry": <0-1>,
    "fearful": <0-1>,
    "disgust": <0-1>,
    "surprised": <0-1>
  }
}
The scores must sum to 1.0. The dominant emotion must match the highest score.`;

  const activeKeys = API_KEYS.filter(k => k.trim() !== "");
  let lastError;

  for (const key of activeKeys) {
    try {
      const parsed = await tryWithKey(key, base64Audio, mimeType, prompt);

      const scores = {};
      let total = 0;
      for (const label of EMOTION_LABELS) {
        scores[label] = Math.max(0, parseFloat(parsed.scores?.[label] ?? 0));
        total += scores[label];
      }
      if (total > 0) {
        for (const label of EMOTION_LABELS) {
          scores[label] = parseFloat((scores[label] / total).toFixed(4));
        }
      }

      const dominant = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];
      return { emotion: dominant, scores };

    } catch (e) {
      lastError = e;
      console.warn(`Key ending ...${key.slice(-6)} failed: ${e.message}. Trying next key...`);
    }
  }

  throw new Error(lastError?.message || "All API keys failed.");
}

function generateModelVariant(base, noise = 0.12) {
  const scores = {};
  let total = 0;
  for (const [label, val] of Object.entries(base.scores)) {
    scores[label] = Math.max(0, val + (Math.random() * 2 - 1) * noise);
    total += scores[label];
  }
  for (const label of EMOTION_LABELS) {
    scores[label] = parseFloat((scores[label] / total).toFixed(4));
  }
  const dominant = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];
  return { emotion: dominant, scores };
}

window.GeminiSpeech = { analyzeAudioWithGemini, generateModelVariant };
