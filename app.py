import streamlit as st
import librosa
import numpy as np
import tensorflow as tf

from preprocessing import extract_features

model = tf.keras.models.load_model("models/emotion_classifier.h5")

def predict_emotion(audio_file):
    feature_vector = extract_features(audio_file).reshape(1, -1)
    pred = model.predict(feature_vector)
    return np.argmax(pred)

st.title("Speech Emotion Recognition")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file:
    emotion = predict_emotion(uploaded_file)
    st.write(f"Predicted Emotion: {emotion}")
