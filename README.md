# EmotionAI — Multi-Modal Emotion Detection

## Components
| Module | Models |
|--------|--------|
| Facial Emotion | DeepFace (CNN) vs Random Forest vs SVM |
| Text Emotion | DistilRoBERTa (pretrained, no training needed) + TextBlob sentiment |
| Speech Emotion | Your primary model vs Random Forest vs SVM |

## Quick Start
1. Double-click `run.bat` — it will create the venv, install all deps, and start Flask.
2. Open `frontend/index.html` in your browser.

> All packages install into `venv/` only — nothing touches your global Python.

## Training Your Models

### Face (RF + SVM)
Edit `training/train_face.py` — load your dataset into `X` (flattened 48×48 grayscale) and `y` (labels), then run:
```
venv\Scripts\python training\train_face.py
```
Saves `backend/models/face_rf.pkl` and `backend/models/face_svm.pkl`.

### Speech
Paste your training code into `training/train_speech.py`.
Save your models as:
- `backend/models/speech_primary.pkl`
- `backend/models/speech_rf.pkl`
- `backend/models/speech_svm.pkl`

Run:
```
venv\Scripts\python training\train_speech.py
```

## Project Structure
```
ML_AAT/
├── venv/                    # Isolated Python environment
├── backend/
│   ├── app.py               # Flask entry point
│   ├── requirements.txt
│   ├── models/              # Trained .pkl files go here
│   ├── modules/             # Core ML inference logic
│   └── routes/              # Flask API routes
├── training/                # Paste your training code here
├── frontend/                # HTML/CSS/JS UI
└── run.bat                  # One-click setup & launch
```
