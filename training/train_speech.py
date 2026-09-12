"""
Speech Emotion Training — Primary Model + RF + SVM
Paste your training code here.
All three models must be saved to backend/models/ as:
  - speech_primary.pkl
  - speech_rf.pkl
  - speech_svm.pkl

Feature extraction used at inference (speech_emotion.py):
  MFCC (40) + Chroma (12) + Mel (128) + Spectral Contrast (7) + Tonnetz (6)
  = 193 features per sample
"""
import pickle

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── PASTE YOUR TRAINING CODE BELOW ──────────────────────────────────────────
"""
Speech Emotion Detection on RAVDESS Dataset
============================================
Pipeline: Load → Preprocess → Extract Features → Train → Evaluate → Visualize

RAVDESS Emotion Labels:
  01=neutral, 02=calm, 03=happy, 04=sad,
  05=angry, 06=fearful, 07=disgust, 08=surprised

Run in Google Colab with:
  !pip install librosa scikit-learn tensorflow matplotlib seaborn
"""

# ─────────────────────────────────────────────
# 0. INSTALL & IMPORTS
# ─────────────────────────────────────────────
import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────────────────────
# 1. MOUNT GOOGLE DRIVE  (Colab only)
# ─────────────────────────────────────────────
def mount_drive():
    """Mount Google Drive in Colab and return the RAVDESS root path."""
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("✅ Drive mounted.")
    except ImportError:
        print("ℹ️  Not running in Colab – skipping drive mount.")

    # ── EDIT THIS PATH to match your Drive folder ──
    ravdess_path = "/content/drive/MyDrive/RAVDESS"
    return ravdess_path


# ─────────────────────────────────────────────
# 2. LOAD DATASET PATHS & LABELS
# ─────────────────────────────────────────────
EMOTION_MAP = {
    "01": "neutral",  "02": "calm",     "03": "happy",    "04": "sad",
    "05": "angry",    "06": "fearful",  "07": "disgust",  "08": "surprised"
}

def parse_ravdess_filename(filepath: str) -> dict | None:
    """
    RAVDESS naming: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    e.g. 03-01-06-01-02-01-12.wav  →  emotion code = field[2]
    """
    fname = os.path.basename(filepath)
    parts = fname.replace(".wav", "").split("-")
    if len(parts) != 7:
        return None
    emotion_code = parts[2]
    return {
        "filepath":  filepath,
        "emotion":   EMOTION_MAP.get(emotion_code, "unknown"),
        "actor":     int(parts[6]),
        "intensity": parts[3],   # 01=normal, 02=strong
        "modality":  parts[0],   # 03=audio-only
    }

def load_dataset(ravdess_root: str) -> pd.DataFrame:
    """Recursively find all .wav files and build a metadata DataFrame."""
    wav_files = glob.glob(os.path.join(ravdess_root, "**", "*.wav"), recursive=True)
    print(f"Found {len(wav_files)} .wav files under: {ravdess_root}")
    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found. Check your path:\n  {ravdess_root}\n"
            "Common structure: RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
        )

    records = [parse_ravdess_filename(f) for f in wav_files]
    df = pd.DataFrame([r for r in records if r is not None])

    # Keep audio-only modality (03); skip video-only (01) and video+audio (02)
    df = df[df["modality"] == "03"].reset_index(drop=True)
    print(f"Audio-only files kept: {len(df)}")
    print(df["emotion"].value_counts().to_string())
    return df


# ─────────────────────────────────────────────
# 3. FEATURE EXTRACTION
# ─────────────────────────────────────────────
SR        = 22050   # target sample rate
DURATION  = 3.0     # seconds to keep / pad each clip
N_MFCC    = 40
N_MELS    = 128
HOP_LEN   = 512

def extract_features(filepath: str,
                     sr: int      = SR,
                     duration: float = DURATION) -> np.ndarray | None:
    """
    Returns a 1-D feature vector:
      [MFCC mean×40, MFCC std×40, chroma mean×12, mel mean×128,
       ZCR mean, RMS mean]
    = 222 features total
    """
    try:
        y, _ = librosa.load(filepath, sr=sr, duration=duration, mono=True)
        # pad / trim to fixed length
        target_len = int(sr * duration)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LEN)
        chroma      = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LEN)
        mel         = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LEN)
        zcr         = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LEN)
        rms         = librosa.feature.rms(y=y, hop_length=HOP_LEN)

        feats = np.concatenate([
            np.mean(mfcc,   axis=1), np.std(mfcc,   axis=1),  # 80
            np.mean(chroma, axis=1),                            # 12
            np.mean(mel,    axis=1),                            # 128
            [np.mean(zcr)],                                     # 1
            [np.mean(rms)],                                     # 1
        ])
        return feats
    except Exception as e:
        print(f"  ⚠️  Skipping {filepath}: {e}")
        return None


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract features for all files; return (X, y_strings)."""
    from tqdm import tqdm
    print("\nExtracting features …")
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), ncols=80):
        feats = extract_features(row["filepath"])
        if feats is not None:
            X.append(feats)
            y.append(row["emotion"])
    return np.array(X, dtype=np.float32), np.array(y)


# ─────────────────────────────────────────────
# 4. DATA SPLITTING & SCALING
# ─────────────────────────────────────────────
def prepare_data(X: np.ndarray, y: np.ndarray):
    """Encode labels, scale features, split into train/val/test."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    y_cat = to_categorical(y_enc, n_classes)

    # stratified split: 70 / 15 / 15
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y_cat, test_size=0.30, stratify=y_enc, random_state=42)
    y_enc_tmp = np.argmax(y_tmp, axis=1)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_enc_tmp, random_state=42)

    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    print(f"\nSplit → Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_te)}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te, le, scaler, n_classes


# ─────────────────────────────────────────────
# 5. MODEL DEFINITION
# ─────────────────────────────────────────────
def build_model(input_dim: int, n_classes: int) -> tf.keras.Model:
    """
    Dense feed-forward network with BatchNorm + Dropout.
    Simple yet effective for hand-crafted audio features.
    """
    inp = layers.Input(shape=(input_dim,), name="features")

    x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64)(x)
    x = layers.Activation("relu")(x)

    out = layers.Dense(n_classes, activation="softmax", name="emotion")(x)

    model = models.Model(inp, out, name="SER_MLP")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()
    return model


# ─────────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────────
def train_model(model, X_tr, y_tr, X_val, y_val,
                epochs: int = 100, batch_size: int = 64):
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=7, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint("best_ser_model.h5", monitor="val_accuracy",
                                  save_best_only=True, verbose=0),
    ]
    print("\nTraining …")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────
# 7. EVALUATION HELPERS
# ─────────────────────────────────────────────
def evaluate_model(model, X_te, y_te, le):
    """Print test accuracy and full classification report."""
    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"\n{'='*45}")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"{'='*45}\n")

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    y_true = np.argmax(y_te, axis=1)
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    return y_true, y_pred


# ─────────────────────────────────────────────
# 8. VISUALISATION
# ─────────────────────────────────────────────
PALETTE = sns.color_palette("Set2", 8)

def plot_class_distribution(df: pd.DataFrame, save: bool = True):
    fig, ax = plt.subplots(figsize=(10, 4))
    counts = df["emotion"].value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color=PALETTE[:len(counts)],
                  edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title("RAVDESS – Emotion Class Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig("01_class_distribution.png", dpi=150)
    plt.show()


def plot_sample_waveform_and_spectrogram(df: pd.DataFrame, save: bool = True):
    """Show waveform + mel-spectrogram for one sample per emotion."""
    emotions = sorted(df["emotion"].unique())
    n = len(emotions)
    fig, axes = plt.subplots(n, 2, figsize=(13, n * 2.2))
    fig.suptitle("Sample Waveforms & Mel-Spectrograms per Emotion",
                 fontsize=13, fontweight="bold", y=1.01)

    for i, emotion in enumerate(emotions):
        row = df[df["emotion"] == emotion].sample(1, random_state=42).iloc[0]
        y, sr = librosa.load(row["filepath"], duration=3.0, sr=SR, mono=True)

        # waveform
        axes[i, 0].plot(np.linspace(0, 3, len(y)), y,
                        color=PALETTE[i % len(PALETTE)], linewidth=0.6)
        axes[i, 0].set_title(f"{emotion} – waveform", fontsize=9)
        axes[i, 0].set_yticks([])

        # mel-spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel",
                                 ax=axes[i, 1], cmap="magma")
        axes[i, 1].set_title(f"{emotion} – mel-spectrogram", fontsize=9)
        axes[i, 1].set_ylabel("")

    plt.tight_layout()
    if save:
        plt.savefig("02_waveforms_spectrograms.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(history, save: bool = True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ep = range(1, len(history.history["loss"]) + 1)

    ax1.plot(ep, history.history["loss"],     label="Train Loss",  color="#2196F3")
    ax1.plot(ep, history.history["val_loss"], label="Val Loss",    color="#F44336", linestyle="--")
    ax1.set_title("Loss Curve", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(ep, history.history["accuracy"],     label="Train Acc",  color="#4CAF50")
    ax2.plot(ep, history.history["val_accuracy"], label="Val Acc",    color="#FF9800", linestyle="--")
    ax2.set_title("Accuracy Curve", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig("03_training_history.png", dpi=150)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, le, save: bool = True):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("Confusion Matrix – Test Set", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    if save:
        plt.savefig("04_confusion_matrix.png", dpi=150)
    plt.show()


def plot_per_class_accuracy(y_true, y_pred, le, save: bool = True):
    cm = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1) * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(le.classes_, per_class, color=PALETTE[:len(le.classes_)],
                  edgecolor="white")
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in per_class], padding=3, fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_title("Per-Class Accuracy on Test Set", fontsize=13, fontweight="bold")
    ax.set_xlabel("Emotion"); ax.set_ylabel("Accuracy (%)")
    ax.axhline(np.mean(per_class), color="gray", linestyle="--",
               label=f"Mean = {np.mean(per_class):.1f}%")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig("05_per_class_accuracy.png", dpi=150)
    plt.show()


def plot_feature_importance(model, le, save: bool = True):
    """Visualise mean absolute first-layer weights per feature group."""
    w = np.abs(model.layers[1].get_weights()[0])   # (input_dim, 512)
    mean_w = w.mean(axis=1)                         # (input_dim,)

    labels = (
        [f"MFCC_mean_{i}" for i in range(N_MFCC)] +
        [f"MFCC_std_{i}"  for i in range(N_MFCC)] +
        [f"Chroma_{i}"    for i in range(12)]     +
        [f"Mel_{i}"       for i in range(N_MELS)] +
        ["ZCR", "RMS"]
    )

    # Group-level summary
    groups = {"MFCC Mean": 40, "MFCC Std": 40, "Chroma": 12,
              "Mel": 128, "ZCR+RMS": 2}
    group_means, start = [], 0
    for name, size in groups.items():
        group_means.append((name, mean_w[start:start+size].mean()))
        start += size

    names, vals = zip(*group_means)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, vals, color=PALETTE[:len(names)], edgecolor="white")
    ax.set_title("Mean |Weight| by Feature Group (Layer 1)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean |Weight|")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig("06_feature_importance.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# 9. INFERENCE UTILITY
# ─────────────────────────────────────────────
def predict_emotion(filepath: str, model, scaler, le) -> str:
    """Predict the emotion of a single .wav file."""
    feats = extract_features(filepath)
    if feats is None:
        return "Error loading file"
    feats_scaled = scaler.transform(feats.reshape(1, -1))
    probs = model.predict(feats_scaled, verbose=0)[0]
    idx   = np.argmax(probs)
    emotion = le.classes_[idx]
    print(f"  → Predicted: {emotion}  (confidence {probs[idx]*100:.1f}%)")
    for e, p in zip(le.classes_, probs):
        print(f"    {e:<12} {p*100:5.1f}%")
    return emotion


# ─────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────
def main():
    # ── Step 1: Mount Drive & load dataset ──────────────────────
    ravdess_root = mount_drive()
    df = load_dataset(ravdess_root)

    # ── Step 2: EDA plots ────────────────────────────────────────
    plot_class_distribution(df)
    plot_sample_waveform_and_spectrogram(df)

    # ── Step 3: Feature extraction ───────────────────────────────
    X, y = build_feature_matrix(df)
    print(f"\nFeature matrix shape: {X.shape}")

    # ── Step 4: Prepare data ─────────────────────────────────────
    X_tr, X_val, X_te, y_tr, y_val, y_te, le, scaler, n_classes = prepare_data(X, y)

    # ── Step 5: Build & train ────────────────────────────────────
    model = build_model(X_tr.shape[1], n_classes)
    history = train_model(model, X_tr, y_tr, X_val, y_val, epochs=100)
    plot_training_history(history)

    # ── Step 6: Evaluate ─────────────────────────────────────────
    y_true, y_pred = evaluate_model(model, X_te, y_te, le)
    plot_confusion_matrix(y_true, y_pred, le)
    plot_per_class_accuracy(y_true, y_pred, le)
    plot_feature_importance(model, le)

    # ── Step 7: Save artefacts ───────────────────────────────────
    model.save("ser_model_final.h5")
    np.save("label_classes.npy", le.classes_)
    import joblib; joblib.dump(scaler, "scaler.pkl")
    print("\n✅ Saved: ser_model_final.h5 | label_classes.npy | scaler.pkl")

    # ── Step 8: Sample inference ─────────────────────────────────
    sample = df.sample(1, random_state=7).iloc[0]
    print(f"\n🎤 Predicting on: {os.path.basename(sample['filepath'])}")
    print(f"   Ground truth: {sample['emotion']}")
    predict_emotion(sample["filepath"], model, scaler, le)

    return model, le, scaler, history


if __name__ == "__main__":
    model, le, scaler, history = main()

    # ── Train RF & SVM on the same features for comparison ───────────────────


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# Re-build the full feature matrix and prepare data (reuses functions above)
ravdess_root = mount_drive()
df_full = load_dataset(ravdess_root)
X_all, y_all = build_feature_matrix(df_full)

le2 = LabelEncoder()
y_enc = le2.fit_transform(y_all)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_all, y_enc, test_size=0.20, stratify=y_enc, random_state=42)
X_tr2 = scaler.transform(X_tr2)
X_te2 = scaler.transform(X_te2)

print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_tr2, y_tr2)
print(classification_report(y_te2, rf.predict(X_te2), target_names=le2.classes_))

print("Training SVM...")
svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(X_tr2, y_tr2)
print(classification_report(y_te2, svm.predict(X_te2), target_names=le2.classes_))

# ── Save all artefacts to backend/models/ ────────────────────────────────
joblib.dump(rf,  os.path.join(MODELS_DIR, "speech_rf.pkl"))
joblib.dump(svm, os.path.join(MODELS_DIR, "speech_svm.pkl"))

# Copy Keras model + scaler + label classes to backend/models/
import shutil
shutil.copy("ser_model_final.h5",  os.path.join(MODELS_DIR, "ser_model_final.h5"))
shutil.copy("scaler.pkl",          os.path.join(MODELS_DIR, "scaler.pkl"))
shutil.copy("label_classes.npy",   os.path.join(MODELS_DIR, "label_classes.npy"))

print("\n✅ All models saved to backend/models/")

