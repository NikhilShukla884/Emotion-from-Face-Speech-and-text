from flask import Flask
from flask_cors import CORS
from routes.face_routes import face_bp
from routes.text_routes import text_bp
from routes.speech_routes import speech_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(face_bp, url_prefix="/api/emotion")
app.register_blueprint(text_bp, url_prefix="/api/emotion")
app.register_blueprint(speech_bp, url_prefix="/api/emotion")

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
