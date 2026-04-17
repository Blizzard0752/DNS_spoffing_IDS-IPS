import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# 1. 讀資料
# =========================
df = pd.read_csv(r"/home/snow/DNS_dnsmasq_log/Spoofing_detect/spoof_dataset(35000).csv")

# 打亂資料（很重要）
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# =========================
# 2. 特徵選擇
# =========================
features = [
    "response_time",
    #"response_count",
    "ttl",
    "packet_interval",
    "duplicate_txid",
    "is_private_ip"
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
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, 
    y_temp, 
    test_size=0.5, 
    random_state=42, 
    stratify=y_temp
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
val_loader   = DataLoader(DNSDataset(X_val, y_val), batch_size=32)
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
epochs = 100

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

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


# ===== Accuracy =====
accuracy = np.mean(np.array(y_pred) == np.array(y_true))
print(f"\n📊 Accuracy: {accuracy:.4f}")

# ===== F1 Score =====
f1 = f1_score(y_true, y_pred)
print(f"🔥 F1 Score: {f1:.4f}")

# 計算每個 feature 的重要性
importance = []

for i, feature in enumerate(features):
    X_test_permuted = X_test.copy()

    # 打亂某一個 feature
    np.random.shuffle(X_test_permuted[:, i])

    acc = evaluate_model(X_test_permuted, y_test)

    drop = accuracy - acc
    importance.append((feature, drop))

# 排序
importance.sort(key=lambda x: x[1], reverse=True)

print("\n🔥 Feature Importance (越大越重要):")
for f, imp in importance:
    print(f"{f}: {imp:.6f}")


'''
Epoch 1/100 | Train Loss: 0.0794 | Val Loss: 0.0233
Epoch 2/100 | Train Loss: 0.0222 | Val Loss: 0.0208
Epoch 3/100 | Train Loss: 0.0209 | Val Loss: 0.0201
Epoch 4/100 | Train Loss: 0.0210 | Val Loss: 0.0198
Epoch 5/100 | Train Loss: 0.0208 | Val Loss: 0.0190
Epoch 6/100 | Train Loss: 0.0201 | Val Loss: 0.0194
Epoch 7/100 | Train Loss: 0.0200 | Val Loss: 0.0188
Epoch 8/100 | Train Loss: 0.0202 | Val Loss: 0.0199
Epoch 9/100 | Train Loss: 0.0199 | Val Loss: 0.0189
Epoch 10/100 | Train Loss: 0.0193 | Val Loss: 0.0180
Epoch 11/100 | Train Loss: 0.0191 | Val Loss: 0.0178
Epoch 12/100 | Train Loss: 0.0189 | Val Loss: 0.0175
Epoch 13/100 | Train Loss: 0.0183 | Val Loss: 0.0173
Epoch 14/100 | Train Loss: 0.0182 | Val Loss: 0.0167
Epoch 15/100 | Train Loss: 0.0179 | Val Loss: 0.0178
Epoch 16/100 | Train Loss: 0.0175 | Val Loss: 0.0164
Epoch 17/100 | Train Loss: 0.0170 | Val Loss: 0.0170
Epoch 18/100 | Train Loss: 0.0167 | Val Loss: 0.0170
Epoch 19/100 | Train Loss: 0.0163 | Val Loss: 0.0183
Epoch 20/100 | Train Loss: 0.0162 | Val Loss: 0.0156
Epoch 21/100 | Train Loss: 0.0160 | Val Loss: 0.0161
Epoch 22/100 | Train Loss: 0.0157 | Val Loss: 0.0150
Epoch 23/100 | Train Loss: 0.0157 | Val Loss: 0.0145
Epoch 24/100 | Train Loss: 0.0153 | Val Loss: 0.0151
Epoch 25/100 | Train Loss: 0.0154 | Val Loss: 0.0147
Epoch 26/100 | Train Loss: 0.0152 | Val Loss: 0.0145
Epoch 27/100 | Train Loss: 0.0150 | Val Loss: 0.0147
Epoch 28/100 | Train Loss: 0.0148 | Val Loss: 0.0145
Epoch 29/100 | Train Loss: 0.0150 | Val Loss: 0.0156
Epoch 30/100 | Train Loss: 0.0152 | Val Loss: 0.0156
Epoch 31/100 | Train Loss: 0.0149 | Val Loss: 0.0157
Epoch 32/100 | Train Loss: 0.0148 | Val Loss: 0.0153
Epoch 33/100 | Train Loss: 0.0148 | Val Loss: 0.0154
Epoch 34/100 | Train Loss: 0.0147 | Val Loss: 0.0156
Epoch 35/100 | Train Loss: 0.0146 | Val Loss: 0.0148
Epoch 36/100 | Train Loss: 0.0145 | Val Loss: 0.0156
Epoch 37/100 | Train Loss: 0.0148 | Val Loss: 0.0151
Epoch 38/100 | Train Loss: 0.0143 | Val Loss: 0.0156
Epoch 39/100 | Train Loss: 0.0147 | Val Loss: 0.0140
Epoch 40/100 | Train Loss: 0.0142 | Val Loss: 0.0140
Epoch 41/100 | Train Loss: 0.0146 | Val Loss: 0.0144
Epoch 42/100 | Train Loss: 0.0143 | Val Loss: 0.0149
Epoch 43/100 | Train Loss: 0.0143 | Val Loss: 0.0167
Epoch 44/100 | Train Loss: 0.0142 | Val Loss: 0.0151
Epoch 45/100 | Train Loss: 0.0141 | Val Loss: 0.0173
Epoch 46/100 | Train Loss: 0.0145 | Val Loss: 0.0168
Epoch 47/100 | Train Loss: 0.0141 | Val Loss: 0.0197
Epoch 48/100 | Train Loss: 0.0143 | Val Loss: 0.0137
Epoch 49/100 | Train Loss: 0.0137 | Val Loss: 0.0177
Epoch 50/100 | Train Loss: 0.0139 | Val Loss: 0.0155
Epoch 51/100 | Train Loss: 0.0139 | Val Loss: 0.0142
Epoch 52/100 | Train Loss: 0.0140 | Val Loss: 0.0158
Epoch 53/100 | Train Loss: 0.0137 | Val Loss: 0.0217
Epoch 54/100 | Train Loss: 0.0140 | Val Loss: 0.0152
Epoch 55/100 | Train Loss: 0.0140 | Val Loss: 0.0150
Epoch 56/100 | Train Loss: 0.0136 | Val Loss: 0.0164
Epoch 57/100 | Train Loss: 0.0140 | Val Loss: 0.0157
Epoch 58/100 | Train Loss: 0.0139 | Val Loss: 0.0151
Epoch 59/100 | Train Loss: 0.0137 | Val Loss: 0.0154
Epoch 60/100 | Train Loss: 0.0135 | Val Loss: 0.0137
Epoch 61/100 | Train Loss: 0.0138 | Val Loss: 0.0155
Epoch 62/100 | Train Loss: 0.0137 | Val Loss: 0.0173
Epoch 63/100 | Train Loss: 0.0137 | Val Loss: 0.0160
Epoch 64/100 | Train Loss: 0.0137 | Val Loss: 0.0162
Epoch 65/100 | Train Loss: 0.0134 | Val Loss: 0.0140
Epoch 66/100 | Train Loss: 0.0136 | Val Loss: 0.0153
Epoch 67/100 | Train Loss: 0.0135 | Val Loss: 0.0159
Epoch 68/100 | Train Loss: 0.0136 | Val Loss: 0.0150
Epoch 69/100 | Train Loss: 0.0140 | Val Loss: 0.0143
Epoch 70/100 | Train Loss: 0.0141 | Val Loss: 0.0150
Epoch 71/100 | Train Loss: 0.0136 | Val Loss: 0.0156
Epoch 72/100 | Train Loss: 0.0136 | Val Loss: 0.0154
Epoch 73/100 | Train Loss: 0.0135 | Val Loss: 0.0155
Epoch 74/100 | Train Loss: 0.0134 | Val Loss: 0.0147
Epoch 75/100 | Train Loss: 0.0135 | Val Loss: 0.0154
Epoch 76/100 | Train Loss: 0.0133 | Val Loss: 0.0150
Epoch 77/100 | Train Loss: 0.0137 | Val Loss: 0.0150
Epoch 78/100 | Train Loss: 0.0136 | Val Loss: 0.0190
Epoch 79/100 | Train Loss: 0.0137 | Val Loss: 0.0145
Epoch 80/100 | Train Loss: 0.0134 | Val Loss: 0.0167
Epoch 81/100 | Train Loss: 0.0135 | Val Loss: 0.0171
Epoch 82/100 | Train Loss: 0.0136 | Val Loss: 0.0147
Epoch 83/100 | Train Loss: 0.0137 | Val Loss: 0.0154
Epoch 84/100 | Train Loss: 0.0133 | Val Loss: 0.0137
Epoch 85/100 | Train Loss: 0.0134 | Val Loss: 0.0144
Epoch 86/100 | Train Loss: 0.0135 | Val Loss: 0.0182
Epoch 87/100 | Train Loss: 0.0135 | Val Loss: 0.0145
Epoch 88/100 | Train Loss: 0.0135 | Val Loss: 0.0154
Epoch 89/100 | Train Loss: 0.0134 | Val Loss: 0.0157
Epoch 90/100 | Train Loss: 0.0135 | Val Loss: 0.0147
Epoch 91/100 | Train Loss: 0.0135 | Val Loss: 0.0140
Epoch 92/100 | Train Loss: 0.0135 | Val Loss: 0.0146
Epoch 93/100 | Train Loss: 0.0132 | Val Loss: 0.0156
Epoch 94/100 | Train Loss: 0.0134 | Val Loss: 0.0145
Epoch 95/100 | Train Loss: 0.0134 | Val Loss: 0.0145
Epoch 96/100 | Train Loss: 0.0133 | Val Loss: 0.0147
Epoch 97/100 | Train Loss: 0.0133 | Val Loss: 0.0148
Epoch 98/100 | Train Loss: 0.0136 | Val Loss: 0.0134
Epoch 99/100 | Train Loss: 0.0132 | Val Loss: 0.0161
Epoch 100/100 | Train Loss: 0.0135 | Val Loss: 0.0159

Classification Report:
              precision    recall  f1-score   support

         0.0       0.99      1.00      0.99      1155
         1.0       1.00      0.99      1.00      2385

    accuracy                           1.00      3540
   macro avg       0.99      1.00      1.00      3540
weighted avg       1.00      1.00      1.00      3540


Confusion Matrix:
[[1154    1]
 [  14 2371]]

✅ 模型已儲存到: /home/snow/DNS_dnsmasq_log/Spoofing_detect/dns_model.pth

📊 Accuracy: 0.9958
🔥 F1 Score: 0.9968

🔥 Feature Importance (越大越重要):
ttl: 0.203955
is_private_ip: 0.196328
packet_interval: 0.014689
response_time: 0.009605
duplicate_txid: 0.000847
'''
