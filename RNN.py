import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import re
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv("data.csv")

# 标签编码
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].values, df['label'].values, test_size=0.2, random_state=42
)

# 文本预处理
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    return text.split()

# 词汇表构建
all_texts = np.concatenate([train_texts, test_texts])
all_tokens = [token for text in all_texts for token in preprocess_text(text)]
word_counts = Counter(all_tokens)
vocab = {word: idx+2 for idx, (word, _) in enumerate(word_counts.items())}
vocab['<unk>'] = 0
vocab['<pad>'] = 1

# 将文本转换为数字索引
def text_to_tensor(text, vocab):
    tokens = preprocess_text(text)
    return [vocab.get(token, vocab['<unk>']) for token in tokens]

MAX_LEN = 100

def pad_sequence(sequence, max_len=MAX_LEN):
    """将序列填充至指定最大长度"""
    if len(sequence) < max_len:
        sequence = sequence + [vocab['<pad>']] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    return sequence

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, vocab, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_tensor = torch.tensor(pad_sequence(text_to_tensor(text, self.vocab), self.max_len), dtype=torch.long)
        return text_tensor, torch.tensor(label, dtype=torch.long)

# 创建数据集
train_dataset = NewsDataset(train_texts, train_labels, vocab)
test_dataset = NewsDataset(test_texts, test_labels, vocab)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# RNN模型定义
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab['<pad>'])
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        hidden_state = rnn_out[:, -1, :]
        output = self.fc(hidden_state)
        return output


# 5GRU模型定义
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab['<pad>'])
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        hidden_state = gru_out[:, -1, :]
        output = self.fc(hidden_state)
        return output

embedding_dim = 100
hidden_dim = 128
output_dim = len(label_encoder.classes_)
model = RNNModel(len(vocab), embedding_dim, hidden_dim, output_dim)
#model = GRUModel(len(vocab), embedding_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

def evaluate(model, test_loader, criterion, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct_predictions / total_predictions

    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    
    return accuracy, mae, mse, rmse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 30
train_losses, train_accuracies = [], []
test_accuracies, test_mae, test_mse, test_rmse = [], [], [], []

for epoch in range(epochs):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    test_accuracy, mae, mse, rmse = evaluate(model, test_loader, criterion, device)
    test_accuracies.append(test_accuracy)
    test_mae.append(mae)
    test_mse.append(mse)
    test_rmse.append(rmse)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

plt.figure(figsize=(12, 8))

# 损失曲线
plt.subplot(2, 2, 1)
plt.plot(range(epochs), train_losses, label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# 准确率曲线
plt.subplot(2, 2, 2)
plt.plot(range(epochs), train_accuracies, label="Train Accuracy")
plt.plot(range(epochs), test_accuracies, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("curve.png")
plt.show()
