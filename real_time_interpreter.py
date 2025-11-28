# real_time_interpreter.py - Real-time ASL recognition

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from collections import deque
from config import *

class ASLInterpreter:
    def __init__(self):
        # Load model and preprocessing objects
        print("\n📦 Loading model...")
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
        
        self.model = load_model(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        
        print("   ✓ Model loaded")
        print(f"   ✓ Gestures recognized: {len(self.label_encoder.classes_)}")
        print(f"     {', '.join(self.label_encoder.classes_)}")
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=STATIC_IMAGE_MODE,
            max_num_hands=2,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # State variables
        self.recording = False
        self.frame_buffer = []
        self.frame_count = 0
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)
        self.last_prediction = None
        self.last_confidence = 0.0
    
    def extract_landmarks(self, frame):
        """Extract hand landmarks from frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
        
        # Pad or truncate to consistent size (126 values)
        landmarks = np.array(landmarks).flatten()
        if len(landmarks) < 126:
            landmarks = np.pad(landmarks, (0, 126 - len(landmarks)))
        else:
            landmarks = landmarks[:126]
        
        return landmarks, results
    
    def predict_gesture(self, sequence):
        """Predict gesture from sequence of landmarks"""
        # Reshape for scaler
        seq_reshaped = sequence.reshape(-1, sequence.shape[-1])
        seq_reshaped = self.scaler.transform(seq_reshaped)
        seq_reshaped = seq_reshaped.reshape(sequence.shape)
        
        # Predict
        predictions = self.model.predict(np.array([seq_reshaped]), verbose=0)
        confidence = np.max(predictions[0])
        pred_idx = np.argmax(predictions[0])
        gesture = self.label_encoder.classes_[pred_idx]
        
        return gesture, confidence
    
    def smooth_prediction(self, gesture, confidence):
        """Smooth predictions over multiple frames"""
        if confidence > CONFIDENCE_THRESHOLD:
            self.prediction_history.append(gesture)
            
            # Return most common prediction
            if len(self.prediction_history) > 0:
                from collections import Counter
                most_common = Counter(self.prediction_history).most_common(1)[0][0]
                return most_common, confidence
        
        return self.last_prediction, self.last_confidence
    
    def draw_ui(self, frame, hand_detected):
        """Draw UI elements on frame"""
        height, width = frame.shape[:2]
        
        # Top bar
        cv2.rectangle(frame, (0, 0), (width, 80), (30, 30, 30), -1)
        cv2.putText(frame, "ASL Real-Time Interpreter", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if self.recording:
            cv2.putText(frame, f"REC: {self.frame_count}/{SEQUENCE_LENGTH}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press SPACE to record | Q to quit",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Hand detection status
        if hand_detected:
            cv2.circle(frame, (width - 30, 30), 10, (0, 255, 0), -1)
            cv2.putText(frame, "Hand", (width - 70, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.circle(frame, (width - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "No Hand", (width - 80, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Bottom prediction display
        if self.last_prediction:
            cv2.rectangle(frame, (0, height - 80), (width, height), (30, 30, 30), -1)
            cv2.putText(frame, "PREDICTION", (10, height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(frame, f"{self.last_prediction}",
                       (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {self.last_confidence:.2f}",
                       (width - 250, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)
        
        return frame
    
    def run(self):
        """Main inference loop"""
        print("\n" + "="*50)
        print("   ASL REAL-TIME INTERPRETER")
        print("="*50)
        print("Controls:")
        print("   SPACE - Start/Stop recording gesture")
        print("   Q     - Quit")
        print("="*50 + "\n")
        
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        if not cap.isOpened():
            raise RuntimeError("Cannot access camera. Check camera index in config.py")
        
        print("✓ Camera activated. Press SPACE to start recording...\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            landmarks, results = self.extract_landmarks(frame)
            hand_detected = results.multi_hand_landmarks is not None
            
            # Handle recording
            if self.recording:
                self.frame_buffer.append(landmarks)
                self.frame_count += 1
                
                if self.frame_count >= SEQUENCE_LENGTH:
                    # Sequence complete, make prediction
                    sequence = np.array(self.frame_buffer)
                    gesture, confidence = self.predict_gesture(sequence)
                    self.last_prediction, self.last_confidence = self.smooth_prediction(gesture, confidence)
                    
                    print(f"🎯 Prediction: {self.last_prediction} (Confidence: {self.last_confidence:.2f})")
                    
                    # Reset recording
                    self.recording = False
                    self.frame_buffer = []
                    self.frame_count = 0
            
            # Draw landmarks
            if hand_detected:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Draw UI
            frame = self.draw_ui(frame, hand_detected)
            
            cv2.imshow('ASL Interpreter', frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if not self.recording:
                    self.recording = True
                    self.frame_buffer = []
                    self.frame_count = 0
                    print("\n🔴 Recording started...")
                else:
                    self.recording = False
                    print("⏹ Recording stopped")
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Interpreter closed")

if __name__ == "__main__":
    interpreter = ASLInterpreter()
    interpreter.run()
