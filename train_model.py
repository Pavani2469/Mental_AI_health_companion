import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import os

def load_data(data_dir):
    """Loads preprocessed training and testing data."""
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    return X_train, X_test, y_train, y_test

def build_model(input_shape, num_classes):
    """Builds a simple Deep Neural Network for classification."""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')  # Multi-class classification
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_and_save_model(data_dir, model_output):
    """Trains the model and saves it."""
    X_train, X_test, y_train, y_test = load_data(data_dir)
    
    num_classes = len(np.unique(y_train))  # Number of emotion categories
    model = build_model(X_train.shape[1], num_classes)

    # Train model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

    # Save the trained model
    model.save(model_output)
    print(f"Model saved at {model_output}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_and_save_model("processed_data/english", "models/emotion_model_english.h5")
    train_and_save_model("processed_data/kannada", "models/emotion_model_kannada.h5")
