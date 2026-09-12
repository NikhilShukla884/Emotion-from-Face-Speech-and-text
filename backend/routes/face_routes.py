from flask import Blueprint, request, jsonify
import numpy as np
import cv2
import base64
from modules.face_emotion import predict_all

face_bp = Blueprint("face", __name__)


@face_bp.route("/face", methods=["POST"])
def face_emotion():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400
    try:
        img_data = base64.b64decode(data["image"].split(",")[-1])
        np_arr   = np.frombuffer(img_data, np.uint8)
        img      = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return jsonify(predict_all(img))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
