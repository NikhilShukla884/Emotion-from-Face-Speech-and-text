from flask import Blueprint, request, jsonify
from modules.text_emotion import analyze_text

text_bp = Blueprint("text", __name__)


@text_bp.route("/text", methods=["POST"])
def text_emotion():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = analyze_text(data["text"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
