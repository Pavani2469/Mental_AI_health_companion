import tensorflow as tf
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import Wav2Vec2ForSequenceClassification

# Load test data (Hindi, Telugu, Kannada, Urdu)
languages = ["hindi", "telugu", "kannada", "urdu"]
models = {
    "MLP": tf.keras.models.load_model("models/emotion_classifier.h5"),
    "CNN": tf.keras.models.load_model("models/emotion_classifier_cnn.h5"),
    "RNN": tf.keras.models.load_model("models/emotion_classifier_rnn.h5")
}

# Evaluate each model on each language dataset
for lang in languages:
    X_test = pd.read_csv(f"data/{lang}_test.csv").values
    y_test = pd.read_csv(f"data/{lang}_test_labels.csv").values.ravel()

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_classes = y_pred.argmax(axis=1)
        accuracy = accuracy_score(y_test, y_pred_classes)
        print(f"{model_name} on {lang}: {accuracy * 100:.2f}%")

# Evaluate Wav2Vec model
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.long)
wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-large-xlsr-53", num_labels=len(set(y_test)))
wav2vec_model.load_state_dict(torch.load("models/emotion_classifier_wav2vec.pth"))

outputs = wav2vec_model(X_test_torch).logits
y_pred_torch = outputs.argmax(dim=1).numpy()
accuracy_wav2vec = accuracy_score(y_test, y_pred_torch)
print(f"Wav2Vec on {lang}: {accuracy_wav2vec * 100:.2f}%")
