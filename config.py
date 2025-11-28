# config.py - Configuration file for the ASL Interpreter

import os

# MediaPipe Settings
MEDIAPIPE_MODEL_COMPLEXITY = 0  # 0 (fast), 1 (balanced), 2 (accurate)
STATIC_IMAGE_MODE = False
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5

# Video Settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
CAMERA_INDEX = 0  # 0 for default webcam

# Data Collection Settings
SAMPLES_PER_GESTURE = 30
SEQUENCE_LENGTH = 30  # Number of frames to capture per gesture
GESTURE_TIME = 1.5  # Seconds to hold gesture

# Model Training Settings
LSTM_UNITS = 128
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# Prediction Settings
CONFIDENCE_THRESHOLD = 0.7  # Min confidence to display prediction
SMOOTHING_WINDOW = 3  # Frames for smoothing predictions

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAINING_DATA_DIR = os.path.join(DATA_DIR, "training_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "asl_model.h5")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# Create directories if they don't exist
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ASL Gestures to Recognize
ASL_GESTURES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'HELLO', 'THANK_YOU', 'PLEASE', 'YES', 'NO'
]
