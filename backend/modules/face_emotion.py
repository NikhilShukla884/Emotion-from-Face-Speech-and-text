import cv2
import numpy as np
import os

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def predict_all(img_array):
    try:
        from deepface import DeepFace
        result   = DeepFace.analyze(img_array, actions=["emotion"], enforce_detection=False)
        emotions = result[0]["emotion"]
        dominant = result[0]["dominant_emotion"]
        scores   = {k: round(v / 100, 4) for k, v in emotions.items()}
        return {"primary_model": {"emotion": dominant, "scores": scores}}
    except Exception as e:
        return {"primary_model": {"error": str(e)}}
