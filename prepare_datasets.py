import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def prepare_dataset(feature_csv, output_dir):
    """
    Loads features from CSV, encodes labels, normalizes features, and splits into train/test sets.
    """
    # Load dataset
    df = pd.read_csv(feature_csv)

    # Extract features (all columns except last) and labels (last column)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Encode labels (emotion categories)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save processed datasets
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    # Save encoders for later use
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    print(f"Dataset prepared and saved in {output_dir}")

if __name__ == "__main__":
    prepare_dataset("data/english_features.csv", "processed_data/english")
    prepare_dataset("data/kannada_features.csv", "processed_data/kannada")
