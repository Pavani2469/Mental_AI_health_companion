import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def load_data(data_dir):
    """Loads test data from another language dataset."""
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_test, y_test


def map_labels(y_pred_labels, mapping):
    """Maps predicted labels to match the target dataset's encoding."""
    return [mapping[label] if label in mapping else -1 for label in y_pred_labels]



def cross_language_evaluation(source_model, target_data_dir, target_label_encoder):
    """Evaluates a model trained on one language on another language dataset."""
    print(f"\nEvaluating {source_model} on {target_data_dir} dataset...\n")
    
    # Load trained model
    model = load_model(source_model)

    # Load target language dataset
    X_test, y_test = load_data(target_data_dir)

    # Predict
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Load label encoder for correct emotion labels
    label_encoder = joblib.load(target_label_encoder)
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Print evaluation results
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels))

if __name__ == "__main__":
    # English Model tested on Kannada Data
    cross_language_evaluation(
        "models/emotion_model_english.h5", 
        "processed_data/kannada", 
        "processed_data/kannada/label_encoder.pkl"
    )
    
    # Kannada Model tested on English Data
    cross_language_evaluation(
        "models/emotion_model_kannada.h5", 
        "processed_data/english", 
        "processed_data/english/label_encoder.pkl"
    )
