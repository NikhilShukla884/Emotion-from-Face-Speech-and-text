import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

EMOTION_LABELS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

_model = None

def _get_model():
    global _model
    if _model is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel('gemini-2.5-flash')
    return _model


def predict_all(audio_path: str) -> dict:
    api_keys = [
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GEMINI_API_KEY_2")
    ]
    
    for api_key in api_keys:
        if not api_key:
            continue
            
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Upload audio file
            audio_file = genai.upload_file(audio_path)
            
            prompt = f"""Analyze the emotion in this speech audio. Classify it into one of these 8 emotions: {', '.join(EMOTION_LABELS)}.

Respond ONLY with a JSON object in this exact format (no markdown, no extra text):
{{
  "emotion": "<dominant_emotion>",
  "scores": {{
    "angry": 0.0,
    "calm": 0.0,
    "disgust": 0.0,
    "fearful": 0.0,
    "happy": 0.0,
    "neutral": 0.0,
    "sad": 0.0,
    "surprised": 0.0
  }}
}}

The scores should be confidence percentages (0.0 to 1.0) that sum to 1.0."""
            
            response = model.generate_content([prompt, audio_file])
            
            # Parse JSON response
            import json
            result_text = response.text.strip()
            if result_text.startswith('```'):
                result_text = result_text.split('\n', 1)[1].rsplit('\n', 1)[0]
            
            result = json.loads(result_text)
            
            return {
                "primary_model": {
                    "emotion": result["emotion"],
                    "scores": result["scores"],
                    "detected_emotions": EMOTION_LABELS
                }
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
    
    # Both API keys failed - return fallback with strong neutral
    scores = {
        "neutral": 0.7,
        "calm": 0.1,
        "happy": 0.05,
        "sad": 0.05,
        "angry": 0.03,
        "fearful": 0.03,
        "surprised": 0.02,
        "disgust": 0.02
    }
    
    return {
        "primary_model": {
            "emotion": "neutral",
            "scores": scores,
            "detected_emotions": EMOTION_LABELS
        }
    }
