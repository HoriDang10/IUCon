import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from nlp import bag_of_words, stem, vietnamese_tokenizer
from model import NeuralNet

# Load intents and questions
with open('Intents.json', 'r') as f:
    intents = json.load(f)
with open('Questions.json', 'r') as f:
    questions_data = json.load(f)

# Prepare data
all_words = []
tags = []
xy = []
# Intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = vietnamese_tokenizer(pattern)
        all_words.extend(w)
        xy.append((w, tag))
# Questions patterns
for question_set in questions_data['questions']:
    tag = question_set['tag']
    tags.append(tag)
    for qa in question_set['questions_and_answers']:
        questions = qa['question']
        if isinstance(questions, list):
            for q in questions:
                w = vietnamese_tokenizer(q)
                all_words.extend(w)
                xy.append((w, tag))
        else:
            w = vietnamese_tokenizer(questions)
            all_words.extend(w)
            xy.append((w, tag))

# Clean words
ignore_words = ['?', '.', ',', '❤']
all_words = sorted(set(stem(w) for w in all_words if w not in ignore_words))
tags = sorted(set(tags))

# Create training data
X = []
Y = []
for (pattern_sentence, tag) in xy:
    X.append(bag_of_words(pattern_sentence, all_words))
    Y.append(tags.index(tag))
X = np.array(X)
Y = np.array(Y)

# Train/Validation split
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.15, random_state=42, stratify=Y
)

class ChatDataset(Dataset):
    def __init__(self, X_data, Y_data):
        self.x_data = torch.from_numpy(X_data).float()
        self.y_data = torch.from_numpy(Y_data).long()
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    def __len__(self):
        return len(self.x_data)

# Hyperparameters
batch_size = 8
hidden_size = 64
output_size = len(tags)
input_size = len(X[0])
learning_rate = 0.001
num_epochs = 100
patience = 3

# DataLoaders
train_dataset = ChatDataset(X_train, Y_train)
val_dataset = ChatDataset(X_val, Y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, output_size, hidden_size).to(device)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# Early stopping variables
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    for words, labels in train_loader:
        words, labels = words.to(device), labels.to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
    train_loss /= len(train_dataset)
    train_acc = train_correct / len(train_dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for words, labels in val_loader:
            words, labels = words.to(device), labels.to(device)
            outputs = model(words)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
    val_loss /= len(val_dataset)
    val_acc = val_correct / len(val_dataset)

    # Scheduler step
    scheduler.step(val_loss)

    print(f'Epoch {epoch+1}/{num_epochs} — '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} — '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}')

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            model.load_state_dict(best_model_state)
            break

# Save model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}
FILE = "data.pth"
torch.save(data, FILE)
print(f'Hoàn thành training. Lưu file vào {FILE}')
