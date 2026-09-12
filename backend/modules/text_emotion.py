from transformers import pipeline
from textblob import TextBlob

_emotion_pipeline = None


def _get_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is None:
        _emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            truncation=True,
        )
    return _emotion_pipeline


def analyze_text(text: str):
    pipe    = _get_pipeline()
    results = pipe(text)[0]

    # Normalize scores to sum to 1
    total  = sum(r["score"] for r in results)
    scores = {r["label"]: round(r["score"] / total, 4) for r in results}
    dominant = max(scores, key=scores.get)

    blob         = TextBlob(text)
    polarity     = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)

    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "emotion":        dominant,
        "emotion_scores": scores,
        "sentiment":      sentiment,
        "polarity":       polarity,
        "subjectivity":   subjectivity,
    }
