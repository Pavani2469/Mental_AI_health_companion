import torch
import pandas as pd
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from torch.utils.data import DataLoader, TensorDataset

# Load training data
X_train = pd.read_csv("data/hindi_train.csv").values
y_train = pd.read_csv("data/hindi_train_labels.csv").values.ravel()

# Convert to Torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# Prepare dataset
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Load Wav2Vec 2.0 model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-large-xlsr-53", num_labels=len(set(y_train)))

# Define optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in dataloader:
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch).logits
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

# Save model
torch.save(model.state_dict(), "models/emotion_classifier_wav2vec.pth")
