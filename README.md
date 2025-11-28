# 🤟 Real-Time ASL Interpreter - Complete Project

A full-stack machine learning application for real-time American Sign Language recognition using MediaPipe hand detection and LSTM deep learning.

---

## 📋 Quick Start (5 Minutes to Running)

### Step 1: Setup Environment
```bash
# Create and activate virtual environment
python -m venv asl_env

# Windows:
asl_env\Scripts\activate
# macOS/Linux:
source asl_env/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install opencv-python mediapipe numpy tensorflow scikit-learn joblib
```

### Step 3: Run in Order
```bash
# Phase 1: Collect training data (5-10 min)
python data_collector.py

# Phase 2: Train the model (2-5 min)
python train_model.py

# Phase 3: Real-time recognition (anytime after training)
python real_time_interpreter.py
```

---

## 🎯 How to Use

### Data Collection (`data_collector.py`)
```
1. Script will ask for each gesture (A-Z, HELLO, etc.)
2. Look at camera, perform the gesture
3. Press SPACE to START recording
4. Hold gesture for 1-2 seconds
5. Release SPACE automatically saves
6. Repeat until 30 samples collected per gesture
7. Press Q to quit
```

**💡 Tips for Good Data:**
- Good lighting on your hands
- Keep hands clearly visible
- Perform gesture consistently
- Vary angles slightly (move hand left/right)
- Use good posture and clear hand shapes

### Model Training (`train_model.py`)
```
1. Automatically loads all collected data
2. Normalizes and encodes labels
3. Trains LSTM model for ~1-5 minutes
4. Displays accuracy and loss metrics
5. Saves model to models/asl_model.h5
```

**Expected Accuracy:** 85-95% (depending on data quality)

### Real-Time Recognition (`real_time_interpreter.py`)
```
1. Opens webcam feed
2. Shows real-time hand landmarks
3. SPACE to START recording gesture
4. SPACE again to STOP and get prediction
5. Shows gesture name and confidence score
6. Q to quit
```

**📊 Display Shows:**
- Hand detection status (green = detected)
- Recording progress (when recording)
- Current prediction and confidence
- Landmarks and hand skeleton

---

## 📁 Project Structure

```
asl_interpreter/
├── config.py                    # Configuration settings
├── data_collector.py            # Data collection script
├── train_model.py              # Model training script
├── real_time_interpreter.py    # Real-time inference
├── data/
│   └── training_data/          # Raw gesture data (auto-created)
│       ├── A/
│       ├── B/
│       ├── HELLO/
│       └── ...
├── models/                      # Trained models (auto-created)
│   ├── asl_model.h5            # Trained LSTM model
│   ├── scaler.pkl              # Data scaler
│   └── label_encoder.pkl       # Label encoder
└── README.md                    # This file
```

---

## 🔧 Configuration

Edit `config.py` to customize:

**Performance:**
```python
FRAME_WIDTH = 640        # Lower to 320 for faster inference
MEDIAPIPE_MODEL_COMPLEXITY = 0  # 0=fast, 1=balanced, 2=accurate
SEQUENCE_LENGTH = 30     # Frames per gesture (lower = faster)
```

**Model:**
```python
LSTM_UNITS = 128         # Increase for more complex patterns
EPOCHS = 100             # Training iterations
BATCH_SIZE = 32          # Samples per training step
```

**Data Collection:**
```python
SAMPLES_PER_GESTURE = 30  # Increase for better accuracy
```

---

## 🧠 How It Works

### Architecture

```
Webcam
   ↓
[OpenCV] Capture Frames
   ↓
[MediaPipe] Extract Hand Landmarks (21 keypoints × 2 hands = 126 values)
   ↓
[Sequence Buffer] Collect 30 frames
   ↓
[Data Normalization] StandardScaler
   ↓
[LSTM Model] Bidirectional LSTM layers (128 units)
   ↓
[Dense Layers] Feature extraction and classification
   ↓
[Softmax] Probability distribution across 31 gestures
   ↓
[Smoothing] Majority voting over last 3 predictions
   ↓
Display Result
```

### Key Technologies

| Component | Technology | Role |
|-----------|-----------|------|
| Hand Detection | MediaPipe | Extract 21 keypoints per hand in real-time |
| Temporal Modeling | LSTM | Learn sequences of hand movements |
| Data Processing | NumPy, Scikit-learn | Normalize landmarks and encode labels |
| Model Training | TensorFlow/Keras | Train and save neural network |
| Video Processing | OpenCV | Capture and display frames |

---

## 📊 Expected Performance

| Metric | Expected | Notes |
|--------|----------|-------|
| **Data Collection Time** | 10-20 min | 30 samples × 31 gestures |
| **Training Time** | 2-5 min | On CPU; faster on GPU |
| **Inference Speed** | 30 FPS | Real-time on most laptops |
| **Accuracy** | 85-95% | With 30+ samples per gesture |
| **Model Size** | ~2-3 MB | Lightweight, runs on CPU |

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'mediapipe'"
```bash
pip install mediapipe
```

### "Cannot access camera"
- Check camera permissions
- Try: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`
- If False, camera not accessible

### "No training data found"
- Run `data_collector.py` first
- Make sure to collect for at least a few gestures

### Model training is very slow
- Reduce `FRAME_WIDTH` to 320 in config.py
- Reduce `SEQUENCE_LENGTH` to 20
- Reduce `EPOCHS` to 50

### Low accuracy
- Collect more samples (50+ per gesture)
- Ensure good lighting
- Keep hands clearly visible
- Maintain consistent gesture speed

---

## 💡 Next Steps & Extensions

### Beginner Improvements
1. **Add more gestures** - Extend to full sentences
2. **Fine-tune confidence threshold** - In `config.py`
3. **Record results** - Save predictions to file

### Intermediate Features
4. **Spell-out mode** - Interpret individual signs letter by letter
5. **Gesture dictionary** - Map signs to common words/phrases
6. **Real-time transcription** - Build sentences from continuous stream

### Advanced Enhancements
7. **Deploy to web** - Streamlit or Flask API
8. **Optimize model** - Quantization for mobile devices
9. **Multi-hand tracking** - Improve accuracy for two-handed signs
10. **Add facial expressions** - MediaPipe Facemesh integration

---

## 📚 Learning Resources

- **MediaPipe Documentation**: https://mediapipe.dev/
- **LSTM Tutorial**: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **ASL Resources**: https://www.handspeak.com/
- **TensorFlow Guide**: https://www.tensorflow.org/guide

---

## ⚠️ Important Notes

1. **Privacy**: This app processes video locally—nothing is uploaded
2. **Accuracy**: Varies by hand size, lighting, and background
3. **Generalization**: Model works best on hands similar to training data
4. **Real ASL**: This is gesture recognition, not a full ASL interpreter
5. **Accessibility**: Use this as a learning tool, not a replacement for professional ASL services

---

## 📝 Project Stats

- **Total Gestures**: 31 (26 letters + 5 common words)
- **Training Data**: ~930 sequences (30 per gesture)
- **Model Parameters**: ~100K
- **Inference Latency**: <50ms
- **Dependencies**: 6 major packages

---

## 🎓 Educational Value for Portfolio

This project demonstrates:
- ✅ Deep Learning (LSTM architecture)
- ✅ Real-time Computer Vision (MediaPipe)
- ✅ Data Collection & Preprocessing
- ✅ Model Training & Evaluation
- ✅ TensorFlow/Keras expertise
- ✅ OpenCV for video processing
- ✅ End-to-end ML pipeline
- ✅ Performance optimization

---

## 🤝 Contributing

Suggestions for improvement? Ideas to extend?
- Document your approach
- Share accuracy metrics
- Contribute gesture data
- Optimize performance

---

## 📄 License

Free to use for educational purposes.

---

**Made with ❤️ for accessibility and learning** 🤟
