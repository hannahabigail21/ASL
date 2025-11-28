# train_model.py - Train LSTM model for ASL recognition

import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from config import *

class ASLModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.X_data = []
        self.y_data = []
        
    def load_training_data(self):
        """Load all collected training data"""
        print("\n📂 Loading training data...")
        
        for gesture in ASL_GESTURES:
            gesture_dir = os.path.join(TRAINING_DATA_DIR, gesture)
            if not os.path.exists(gesture_dir):
                print(f"   ⚠ No data found for '{gesture}'")
                continue
            
            sample_files = [f for f in os.listdir(gesture_dir) if f.endswith('.npy')]
            print(f"   ✓ {gesture}: {len(sample_files)} samples")
            
            for sample_file in sample_files:
                file_path = os.path.join(gesture_dir, sample_file)
                sequence = np.load(file_path)
                self.X_data.append(sequence)
                self.y_data.append(gesture)
        
        self.X_data = np.array(self.X_data)
        self.y_data = np.array(self.y_data)
        
        print(f"\n✓ Total samples loaded: {len(self.X_data)}")
        print(f"   Data shape: {self.X_data.shape}")
        
        if len(self.X_data) == 0:
            raise ValueError("No training data found! Run data_collector.py first.")
        
        return self.X_data, self.y_data
    
    def preprocess_data(self, X, y):
        """Normalize data and encode labels"""
        print("\n🔧 Preprocessing data...")
        
        # Reshape for scaler (flatten sequences)
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_reshaped = self.scaler.fit_transform(X_reshaped)
        X = X_reshaped.reshape(X.shape)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"   ✓ Data normalized")
        print(f"   ✓ Labels encoded: {len(self.label_encoder.classes_)} unique gestures")
        
        return X, y_encoded
    
    def build_model(self, input_shape, num_classes):
        """Build LSTM model"""
        print("\n🏗 Building LSTM model...")
        
        model = Sequential([
            LSTM(LSTM_UNITS, activation='relu', return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(LSTM_UNITS, activation='relu', return_sequences=False),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"   ✓ Model architecture created")
        model.summary()
        
        return model
    
    def train_model(self, model, X, y):
        """Train the model"""
        print("\n🚀 Training model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=VALIDATION_SPLIT, random_state=42
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=1
        )
        
        # Evaluate
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"\n✓ Training complete!")
        print(f"   Validation Accuracy: {val_accuracy:.4f}")
        print(f"   Validation Loss: {val_loss:.4f}")
        
        return model, history
    
    def save_model(self, model):
        """Save model and preprocessing objects"""
        print("\n💾 Saving model...")
        
        # Save model
        model.save(MODEL_PATH)
        print(f"   ✓ Model saved to: {MODEL_PATH}")
        
        # Save scaler
        joblib.dump(self.scaler, SCALER_PATH)
        print(f"   ✓ Scaler saved to: {SCALER_PATH}")
        
        # Save label encoder
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
        print(f"   ✓ Label encoder saved to: {LABEL_ENCODER_PATH}")
        
        print(f"\n✓ All files saved to: {MODELS_DIR}")
    
    def run(self):
        """Main training pipeline"""
        print("\n" + "="*50)
        print("   ASL MODEL TRAINER")
        print("="*50)
        
        # Load data
        X, y = self.load_training_data()
        
        # Preprocess
        X, y = self.preprocess_data(X, y)
        
        # Build model
        model = self.build_model((X.shape[1], X.shape[2]), len(self.label_encoder.classes_))
        
        # Train model
        model, history = self.train_model(model, X, y)
        
        # Save model
        self.save_model(model)
        
        print("\n✓ Training pipeline complete!")

if __name__ == "__main__":
    trainer = ASLModelTrainer()
    trainer.run()
