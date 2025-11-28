# data_collector.py - Collect training data for ASL gestures

import cv2
import mediapipe as mp
import numpy as np
import os
from config import *

class ASLDataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=STATIC_IMAGE_MODE,
            max_num_hands=2,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_landmarks(self, frame, hands):
        """Extract hand landmarks from frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
        
        # Pad or truncate to consistent size (21 landmarks × 3 coordinates × 2 hands = 126)
        landmarks = np.array(landmarks).flatten()
        if len(landmarks) < 126:
            landmarks = np.pad(landmarks, (0, 126 - len(landmarks)))
        else:
            landmarks = landmarks[:126]
        
        return landmarks
    
    def collect_gesture(self, gesture_name):
        """Collect samples for a specific gesture"""
        gesture_dir = os.path.join(TRAINING_DATA_DIR, gesture_name)
        os.makedirs(gesture_dir, exist_ok=True)
        
        existing_samples = len(os.listdir(gesture_dir))
        samples_to_collect = SAMPLES_PER_GESTURE - existing_samples
        
        if samples_to_collect <= 0:
            print(f"✓ Already have {SAMPLES_PER_GESTURE} samples for '{gesture_name}'")
            return
        
        print(f"\n📸 Collecting {samples_to_collect} samples for '{gesture_name}'")
        print(f"   Existing: {existing_samples}/{SAMPLES_PER_GESTURE}")
        
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        sample_count = 0
        frame_count = 0
        collecting = False
        frames_buffer = []
        
        while sample_count < samples_to_collect:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            # Display info
            info_text = f"Gesture: {gesture_name} | Sample: {sample_count + existing_samples + 1}/{SAMPLES_PER_GESTURE}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if collecting:
                frame_count += 1
                if frame_count <= SEQUENCE_LENGTH:
                    landmarks = self.extract_landmarks(frame, self.hands)
                    frames_buffer.append(landmarks)
                    cv2.putText(frame, f"REC: {frame_count}/{SEQUENCE_LENGTH}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # Save collected sequence
                    sample_file = os.path.join(gesture_dir, f"sample_{sample_count + existing_samples}.npy")
                    np.save(sample_file, np.array(frames_buffer))
                    sample_count += 1
                    collecting = False
                    frame_count = 0
                    frames_buffer = []
                    print(f"   ✓ Saved sample {sample_count + existing_samples}")
            else:
                cv2.putText(frame, "Press SPACE to start", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, "✓ Hand Detected", (10, FRAME_HEIGHT - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "✗ No Hand Detected", (10, FRAME_HEIGHT - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            cv2.imshow('ASL Data Collector', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                collecting = True
                frames_buffer = []
                frame_count = 0
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"✓ Completed data collection for '{gesture_name}'")
    
    def run(self):
        """Main collection loop"""
        print("\n" + "="*50)
        print("   ASL GESTURE DATA COLLECTOR")
        print("="*50)
        print(f"Total gestures to collect: {len(ASL_GESTURES)}")
        print(f"Samples per gesture: {SAMPLES_PER_GESTURE}")
        print(f"Total data points: ~{len(ASL_GESTURES) * SAMPLES_PER_GESTURE}")
        print("="*50)
        
        for gesture in ASL_GESTURES:
            self.collect_gesture(gesture)
        
        print("\n✓ Data collection complete!")
        print(f"Saved to: {TRAINING_DATA_DIR}")

if __name__ == "__main__":
    collector = ASLDataCollector()
    collector.run()
