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
df = pd.read_csv(r"/home/snow/DNS_dnsmasq_log/Spoofing_detect/dns_dataset.csv")

# 打亂資料（很重要）
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# =========================
# 2. 特徵選擇
# =========================
features = [
    "response_time",
    "response_count",
    "ttl",
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

         1.0       0.92      0.88      0.90       603
         0.0       0.83      0.88      0.85       399

    accuracy                           0.88      1002
   macro avg       0.87      0.88      0.88      1002
weighted avg       0.88      0.88      0.88      1002


Confusion Matrix:
[[351  48]
 [ 73 530]]

✅ 模型已儲存到: /home/snow/DNS_dnsmasq_log/Spoofing_detect/dns_model(30).pth

📊 Baseline Accuracy: 0.8792

🔥 Feature Importance (越大越重要):
response_time: 0.265469
duplicate_txid: 0.133733
packet_interval: 0.080838
response_count: 0.034930



Epoch 1/100 | Train Loss: 0.5089 | Val Loss: 0.3114
Epoch 2/100 | Train Loss: 0.3298 | Val Loss: 0.2748
Epoch 3/100 | Train Loss: 0.2840 | Val Loss: 0.2540
Epoch 4/100 | Train Loss: 0.2637 | Val Loss: 0.2397
Epoch 5/100 | Train Loss: 0.2638 | Val Loss: 0.2363
Epoch 6/100 | Train Loss: 0.2596 | Val Loss: 0.2423
Epoch 7/100 | Train Loss: 0.2550 | Val Loss: 0.2291
Epoch 8/100 | Train Loss: 0.2555 | Val Loss: 0.2351
Epoch 9/100 | Train Loss: 0.2575 | Val Loss: 0.2281
Epoch 10/100 | Train Loss: 0.2510 | Val Loss: 0.2255
Epoch 11/100 | Train Loss: 0.2525 | Val Loss: 0.2247
Epoch 12/100 | Train Loss: 0.2526 | Val Loss: 0.2330
Epoch 13/100 | Train Loss: 0.2497 | Val Loss: 0.2265
Epoch 14/100 | Train Loss: 0.2469 | Val Loss: 0.2211
Epoch 15/100 | Train Loss: 0.2480 | Val Loss: 0.2279
Epoch 16/100 | Train Loss: 0.2492 | Val Loss: 0.2309
Epoch 17/100 | Train Loss: 0.2533 | Val Loss: 0.2207
Epoch 18/100 | Train Loss: 0.2481 | Val Loss: 0.2313
Epoch 19/100 | Train Loss: 0.2460 | Val Loss: 0.2183
Epoch 20/100 | Train Loss: 0.2472 | Val Loss: 0.2236
Epoch 21/100 | Train Loss: 0.2464 | Val Loss: 0.2242
Epoch 22/100 | Train Loss: 0.2475 | Val Loss: 0.2202
Epoch 23/100 | Train Loss: 0.2479 | Val Loss: 0.2174
Epoch 24/100 | Train Loss: 0.2447 | Val Loss: 0.2210
Epoch 25/100 | Train Loss: 0.2462 | Val Loss: 0.2179
Epoch 26/100 | Train Loss: 0.2470 | Val Loss: 0.2216
Epoch 27/100 | Train Loss: 0.2460 | Val Loss: 0.2254
Epoch 28/100 | Train Loss: 0.2445 | Val Loss: 0.2156
Epoch 29/100 | Train Loss: 0.2453 | Val Loss: 0.2195
Epoch 30/100 | Train Loss: 0.2477 | Val Loss: 0.2265
Epoch 31/100 | Train Loss: 0.2418 | Val Loss: 0.2172
Epoch 32/100 | Train Loss: 0.2436 | Val Loss: 0.2242
Epoch 33/100 | Train Loss: 0.2427 | Val Loss: 0.2199
Epoch 34/100 | Train Loss: 0.2447 | Val Loss: 0.2292
Epoch 35/100 | Train Loss: 0.2433 | Val Loss: 0.2179
Epoch 36/100 | Train Loss: 0.2432 | Val Loss: 0.2164
Epoch 37/100 | Train Loss: 0.2430 | Val Loss: 0.2339
Epoch 38/100 | Train Loss: 0.2446 | Val Loss: 0.2193
Epoch 39/100 | Train Loss: 0.2456 | Val Loss: 0.2226
Epoch 40/100 | Train Loss: 0.2445 | Val Loss: 0.2257
Epoch 41/100 | Train Loss: 0.2427 | Val Loss: 0.2164
Epoch 42/100 | Train Loss: 0.2450 | Val Loss: 0.2199
Epoch 43/100 | Train Loss: 0.2429 | Val Loss: 0.2214
Epoch 44/100 | Train Loss: 0.2423 | Val Loss: 0.2160
Epoch 45/100 | Train Loss: 0.2426 | Val Loss: 0.2227
Epoch 46/100 | Train Loss: 0.2431 | Val Loss: 0.2182
Epoch 47/100 | Train Loss: 0.2432 | Val Loss: 0.2306
Epoch 48/100 | Train Loss: 0.2424 | Val Loss: 0.2385
Epoch 49/100 | Train Loss: 0.2436 | Val Loss: 0.2302
Epoch 50/100 | Train Loss: 0.2434 | Val Loss: 0.2187
Epoch 51/100 | Train Loss: 0.2432 | Val Loss: 0.2175
Epoch 52/100 | Train Loss: 0.2409 | Val Loss: 0.2180
Epoch 53/100 | Train Loss: 0.2406 | Val Loss: 0.2192
Epoch 54/100 | Train Loss: 0.2448 | Val Loss: 0.2181
Epoch 55/100 | Train Loss: 0.2409 | Val Loss: 0.2145
Epoch 56/100 | Train Loss: 0.2405 | Val Loss: 0.2167
Epoch 57/100 | Train Loss: 0.2445 | Val Loss: 0.2159
Epoch 58/100 | Train Loss: 0.2416 | Val Loss: 0.2202
Epoch 59/100 | Train Loss: 0.2438 | Val Loss: 0.2196
Epoch 60/100 | Train Loss: 0.2428 | Val Loss: 0.2222
Epoch 61/100 | Train Loss: 0.2397 | Val Loss: 0.2242
Epoch 62/100 | Train Loss: 0.2429 | Val Loss: 0.2319
Epoch 63/100 | Train Loss: 0.2440 | Val Loss: 0.2220
Epoch 64/100 | Train Loss: 0.2417 | Val Loss: 0.2181
Epoch 65/100 | Train Loss: 0.2404 | Val Loss: 0.2239
Epoch 66/100 | Train Loss: 0.2412 | Val Loss: 0.2166
Epoch 67/100 | Train Loss: 0.2388 | Val Loss: 0.2170
Epoch 68/100 | Train Loss: 0.2390 | Val Loss: 0.2196
Epoch 69/100 | Train Loss: 0.2445 | Val Loss: 0.2207
Epoch 70/100 | Train Loss: 0.2382 | Val Loss: 0.2397
Epoch 71/100 | Train Loss: 0.2405 | Val Loss: 0.2182
Epoch 72/100 | Train Loss: 0.2398 | Val Loss: 0.2295
Epoch 73/100 | Train Loss: 0.2388 | Val Loss: 0.2207
Epoch 74/100 | Train Loss: 0.2402 | Val Loss: 0.2190
Epoch 75/100 | Train Loss: 0.2394 | Val Loss: 0.2274
Epoch 76/100 | Train Loss: 0.2379 | Val Loss: 0.2176
Epoch 77/100 | Train Loss: 0.2416 | Val Loss: 0.2235
Epoch 78/100 | Train Loss: 0.2394 | Val Loss: 0.2150
Epoch 79/100 | Train Loss: 0.2402 | Val Loss: 0.2165
Epoch 80/100 | Train Loss: 0.2399 | Val Loss: 0.2146
Epoch 81/100 | Train Loss: 0.2388 | Val Loss: 0.2158
Epoch 82/100 | Train Loss: 0.2391 | Val Loss: 0.2224
Epoch 83/100 | Train Loss: 0.2417 | Val Loss: 0.2128
Epoch 84/100 | Train Loss: 0.2370 | Val Loss: 0.2143
Epoch 85/100 | Train Loss: 0.2387 | Val Loss: 0.2131
Epoch 86/100 | Train Loss: 0.2352 | Val Loss: 0.2319
Epoch 87/100 | Train Loss: 0.2362 | Val Loss: 0.2210
Epoch 88/100 | Train Loss: 0.2389 | Val Loss: 0.2159
Epoch 89/100 | Train Loss: 0.2384 | Val Loss: 0.2221
Epoch 90/100 | Train Loss: 0.2360 | Val Loss: 0.2114
Epoch 91/100 | Train Loss: 0.2359 | Val Loss: 0.2190
Epoch 92/100 | Train Loss: 0.2362 | Val Loss: 0.2216
Epoch 93/100 | Train Loss: 0.2384 | Val Loss: 0.2289
Epoch 94/100 | Train Loss: 0.2371 | Val Loss: 0.2362
Epoch 95/100 | Train Loss: 0.2390 | Val Loss: 0.2191
Epoch 96/100 | Train Loss: 0.2375 | Val Loss: 0.2176
Epoch 97/100 | Train Loss: 0.2389 | Val Loss: 0.2170
Epoch 98/100 | Train Loss: 0.2351 | Val Loss: 0.2243
Epoch 99/100 | Train Loss: 0.2355 | Val Loss: 0.2244
Epoch 100/100 | Train Loss: 0.2353 | Val Loss: 0.2249

Classification Report:
              precision    recall  f1-score   support

         1.0       0.90      0.86      0.88       301              
         0.0       0.80      0.86      0.83       200

    accuracy                           0.86       501
   macro avg       0.85      0.86      0.85       501
weighted avg       0.86      0.86      0.86       501


Confusion Matrix:
[[172  28]
 [ 43 258]]

✅ 模型已儲存到: /home/snow/DNS_dnsmasq_log/Spoofing_detect/dns_model(100).pth

📊 Accuracy: 0.8583
🔥 F1 Score: 0.8790

🔥 Feature Importance (越大越重要):
packet_interval: 0.127745
response_time: 0.123752
duplicate_txid: 0.107784
response_count: 0.075848
'''
