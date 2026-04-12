import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# 1. 讀資料
# =========================
df = pd.read_csv(r"/home/snow/DNS_dnsmasq_log/Spoofing_detect/dns_dataset.csv")

# 打亂資料（很重要）
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# =========================
# 2. 特徵選擇
# =========================
features = [
    "response_time",
    "response_count",
    #"ttl",
    "packet_interval",
    "duplicate_txid",
    #"is_private_ip"
]

# 可選加分特徵（如果你有加）
if "is_zero_ttl" in df.columns:
    features.append("is_zero_ttl")
if "is_fast_response" in df.columns:
    features.append("is_fast_response")

X = df[features].values
y = df["label"].values

# =========================
# 3. 標準化
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# 4. 切資料
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 5. Dataset
# =========================
class DNSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(DNSDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(DNSDataset(X_test, y_test), batch_size=32)

# =========================
# 6. MLP 模型
# =========================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = MLP(len(features))

# =========================
# 7. Loss / Optimizer
# =========================
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# 8. Training
# =========================
epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# =========================
# 9. Evaluation
# =========================
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch).squeeze()
        preds = (outputs > 0.5).int()

        y_pred.extend(preds.numpy())
        y_true.extend(y_batch.numpy())

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# =========================
# 🔥 10. 儲存模型（重點）
# =========================

MODEL_PATH = "/home/snow/DNS_dnsmasq_log/Spoofing_detect/dns_model.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "scaler": scaler,
    "features": features
}, MODEL_PATH)

print(f"\n✅ 模型已儲存到: {MODEL_PATH}")


# =========================
# 🔥 11. Feature Importance（重點）
# =========================

def evaluate_model(X, y):
    dataset = DNSDataset(X, y)
    loader = DataLoader(dataset, batch_size=32)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch).squeeze()
            preds = (outputs > 0.5).int()

            correct += (preds == y_batch.int()).sum().item()
            total += len(y_batch)

    return correct / total


# 原始準確率
baseline_acc = evaluate_model(X_test, y_test)
print(f"\n📊 Baseline Accuracy: {baseline_acc:.4f}")

# 計算每個 feature 的重要性
importance = []

for i, feature in enumerate(features):
    X_test_permuted = X_test.copy()

    # 打亂某一個 feature
    np.random.shuffle(X_test_permuted[:, i])

    acc = evaluate_model(X_test_permuted, y_test)

    drop = baseline_acc - acc
    importance.append((feature, drop))

# 排序
importance.sort(key=lambda x: x[1], reverse=True)

print("\n🔥 Feature Importance (越大越重要):")
for f, imp in importance:
    print(f"{f}: {imp:.6f}")


'''
Epoch 1/30 - Loss: 59.7643
Epoch 2/30 - Loss: 40.3294
Epoch 3/30 - Loss: 34.1523
Epoch 4/30 - Loss: 32.8382
Epoch 5/30 - Loss: 32.2059
Epoch 6/30 - Loss: 31.6712
Epoch 7/30 - Loss: 31.8369
Epoch 8/30 - Loss: 31.5748
Epoch 9/30 - Loss: 31.5059
Epoch 10/30 - Loss: 31.3252
Epoch 11/30 - Loss: 31.3055
Epoch 12/30 - Loss: 31.0171
Epoch 13/30 - Loss: 31.3813
Epoch 14/30 - Loss: 31.8602
Epoch 15/30 - Loss: 31.4187
Epoch 16/30 - Loss: 31.0867
Epoch 17/30 - Loss: 31.0899
Epoch 18/30 - Loss: 30.9769
Epoch 19/30 - Loss: 31.2073
Epoch 20/30 - Loss: 31.0532
Epoch 21/30 - Loss: 30.8659
Epoch 22/30 - Loss: 30.9746
Epoch 23/30 - Loss: 30.9383
Epoch 24/30 - Loss: 30.8921
Epoch 25/30 - Loss: 31.0524
Epoch 26/30 - Loss: 30.9340
Epoch 27/30 - Loss: 31.1207
Epoch 28/30 - Loss: 30.7780
Epoch 29/30 - Loss: 30.9541
Epoch 30/30 - Loss: 30.8707

Classification Report:
              precision    recall  f1-score   support

         0.0       0.83      0.88      0.85       399
         1.0       0.92      0.88      0.90       603

    accuracy                           0.88      1002
   macro avg       0.87      0.88      0.88      1002
weighted avg       0.88      0.88      0.88      1002


Confusion Matrix:
[[351  48]
 [ 73 530]]

✅ 模型已儲存到: /home/snow/DNS_dnsmasq_log/Spoofing_detect/dns_model.pth

📊 Baseline Accuracy: 0.8792

🔥 Feature Importance (越大越重要):
response_time: 0.265469
duplicate_txid: 0.133733
packet_interval: 0.080838
response_count: 0.034930

'''
