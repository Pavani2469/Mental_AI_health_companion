import librosa
import numpy as np
import os
import pandas as pd

def extract_features(audio_path):
    """
    Extracts MFCCs, Chroma, and Mel Spectrogram features from an audio file.
    """
    y, sr = librosa.load(audio_path, sr=None)

    # Extract features
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)

    return np.hstack([mfccs, chroma, mel])

def process_english_dataset(data_dir, output_csv):
    """
    Processes the English dataset where emotions are stored in subdirectories.
    """
    features, labels = [], []

    for speaker in os.listdir(data_dir):  # Iterate through speakers
        speaker_path = os.path.join(data_dir, speaker)
        if not os.path.isdir(speaker_path):
            continue
        
        for emotion in os.listdir(speaker_path):  # Iterate through emotions
            emotion_path = os.path.join(speaker_path, emotion)
            if not os.path.isdir(emotion_path):
                continue
            
            for file in os.listdir(emotion_path):  # Iterate through WAV files
                if file.endswith('.wav'):
                    file_path = os.path.join(emotion_path, file)
                    feature_vector = extract_features(file_path)
                    features.append(feature_vector)
                    labels.append(emotion)  # Emotion from folder name

    # Save extracted features
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(output_csv, index=False)
    print(f"Saved English dataset features to {output_csv}")

def process_kannada_dataset(data_dir, output_csv):
    """
    Processes the Kannada dataset where emotion labels are in filenames.
    """
    features, labels = [], []

    for file in os.listdir(data_dir):
        if file.endswith('.wav'):
            file_path = os.path.join(data_dir, file)

            # Extract emotion from filename format (assuming format "01-01-01.wav")
            emotion_label = file.split("-")[2].split(".")[0]  # Extract last part before ".wav"

            feature_vector = extract_features(file_path)
            features.append(feature_vector)
            labels.append(emotion_label)  # Emotion from filename

    # Save extracted features
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(output_csv, index=False)
    print(f"Saved Kannada dataset features to {output_csv}")

if __name__ == "__main__":
    process_english_dataset("data/English", "data/english_features.csv")
    process_kannada_dataset("data/kannada", "data/kannada_features.csv")
