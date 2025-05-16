import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

# Load training data
X_train = pd.read_csv("data/hindi_train.csv").values
y_train = pd.read_csv("data/hindi_train_labels.csv").values.ravel()

# Reshape input for RNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# LSTM Model
model = keras.Sequential([
    keras.layers.LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    keras.layers.LSTM(64),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(set(y_train)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=30, batch_size=32)

# Save Model
model.save("models/emotion_classifier_rnn.h5")
