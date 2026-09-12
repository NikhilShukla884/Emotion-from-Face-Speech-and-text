from flask import Blueprint, request, jsonify
import os
import tempfile
from modules.speech_emotion import predict_all

speech_bp = Blueprint("speech", __name__)


@speech_bp.route("/speech", methods=["POST"])
def speech_emotion():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    tmp_path   = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        result = predict_all(tmp_path)
        return jsonify(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.unlink(tmp_path)
            except: pass
