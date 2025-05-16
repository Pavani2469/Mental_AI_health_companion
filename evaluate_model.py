import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def load_data(data_dir):
    """Loads preprocessed testing data."""
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_test, y_test

def evaluate_model(model_path, data_dir, label_encoder_path):
    """Evaluates a trained model and prints classification results."""
    # Load trained model
    model = load_model(model_path)

    # Load test data
    X_test, y_test = load_data(data_dir)

    # Predict labels
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Load label encoder to get emotion names
    label_encoder = joblib.load(label_encoder_path)
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Print evaluation metrics
    print(f"Evaluation for {model_path}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels))

if __name__ == "__main__":
    evaluate_model("models/emotion_model_english.h5", "processed_data/english", "processed_data/english/label_encoder.pkl")
    evaluate_model("models/emotion_model_kannada.h5", "processed_data/kannada", "processed_data/kannada/label_encoder.pkl")
