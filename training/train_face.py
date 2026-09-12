"""
Face Emotion Training — RF & SVM
Paste your dataset loading and feature extraction code here.
This script trains Random Forest and SVM on 48x48 grayscale face images
and saves models to backend/models/
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


try:
    data = pd.read_csv('fer2013.csv')  # Try current directory
except FileNotFoundError:
    data = pd.read_csv('/content/fer2013.csv')  # Try Colab default upload location

print(f"Loaded {len(data)} images")
print(data.head())


def process_pixels(pixel_string):
    """Convert a pixel string into a 48x48 numpy array of grayscale values"""
    pixels = np.array([int(p) for p in pixel_string.split()], dtype='float32')
    return pixels.reshape(48, 48)  # Reshape to 48x48 image

# Apply the processing to all rows
X = np.array([process_pixels(p) for p in data['pixels']])
y = data['emotion'].values

# Normalize pixel values (0-255 becomes 0-1) – this helps the model train faster
X = X / 255.0

# Add channel dimension (48,48) becomes (48,48,1) for grayscale
X = X.reshape(-1, 48, 48, 1)

print(f"Images shape: {X.shape}")
print(f"Labels shape: {y.shape}")

emotion_labels = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
    4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

# Convert numeric labels to categorical format for training
y_one_hot = tf.keras.utils.to_categorical(y, num_classes=7)

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} images")
print(f"Testing set: {X_test.shape[0]} images")

y = data_cleaned['emotion'].values
y_one_hot = tf.keras.utils.to_categorical(y, num_classes=7)

# Re-split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} images")
print(f"Testing set: {X_test.shape[0]} images")

model = keras.Sequential([
    # First convolutional block – learns basic features (edges, corners)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Second convolutional block – learns more complex patterns
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Third convolutional block – learns high-level features
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Dense layers – decision-making layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')  # 7 emotion classes
])

# Compile the model – define how training will work
model.compile(
    optimizer='adam',  # Algorithm that updates weights during training
    loss='categorical_crossentropy',  # Measures prediction error
    metrics=['accuracy']
)

model.summary()

callbacks = [
    # Early stopping – stops training when no improvement
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10, restore_best_weights=True
    ),
    # Reduce learning rate when progress stalls
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
    )
]

print("Starting training (this will take 15-30 minutes)...")

history = model.fit(
    X_train, y_train,
    batch_size=64,  # Number of images processed at once
    epochs=65,           # Total epochs
    # Maximum training rounds
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


assert X is not None and y is not None, "Load your data into X and y first."

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print(classification_report(y_test, rf.predict(X_test)))
with open(os.path.join(MODELS_DIR, "face_rf.pkl"), "wb") as f:
    pickle.dump(rf, f)

print("Training SVM...")
svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(X_train, y_train)
print(classification_report(y_test, svm.predict(X_test)))
with open(os.path.join(MODELS_DIR, "face_svm.pkl"), "wb") as f:
    pickle.dump(svm, f)

print("Models saved to backend/models/")
