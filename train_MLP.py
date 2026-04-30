import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# 1. 讀資料
# =========================
df = pd.read_csv(r"/home/snow/DNS_dnsmasq_log/Spoofing_detect/spoof_dataset(10000).csv")

# 打亂資料（可留）
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# =========================
# 🔥 2. 特徵選擇
# =========================
features = [
    "response_time",
    #"ttl",
    "packet_interval",
    "duplicate_txid",
    "is_private_ip"
]

if "is_zero_ttl" in df.columns:
    features.append("is_zero_ttl")
if "is_fast_response" in df.columns:
    features.append("is_fast_response")

X = df[features].values
y = df["label"].values

# 🔥 關鍵：分組依據（TXID）
groups = df["txid"].values

# =========================
# 🔥 3. 依 TXID 切分資料（無洩漏）
# =========================
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, temp_idx = next(gss1.split(X, y, groups))

X_train, X_temp = X[train_idx], X[temp_idx]
y_train, y_temp = y[train_idx], y[temp_idx]
groups_train, groups_temp = groups[train_idx], groups[temp_idx]

# 再切 val / test（仍然用 group）
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups_temp))

X_val, X_test = X_temp[val_idx], X_temp[test_idx]
y_val, y_test = y_temp[val_idx], y_temp[test_idx]

# =========================
# 🔥 4. 標準化（避免資料洩漏）
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

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
test_loader  = DataLoader(DNSDataset(X_test, y_test), batch_size=32)

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
# 🔥 10. 儲存模型
# =========================
MODEL_PATH = "/home/snow/DNS_dnsmasq_log/Spoofing_detect/dns_model.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "scaler": scaler,
    "features": features
}, MODEL_PATH)

print(f"\n✅ 模型已儲存到: {MODEL_PATH}")

# =========================
# 🔥 11. Feature Importance
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

accuracy = np.mean(np.array(y_pred) == np.array(y_true))
print(f"\n📊 Accuracy: {accuracy:.4f}")

f1 = f1_score(y_true, y_pred)
print(f"🔥 F1 Score: {f1:.4f}")

importance = []

for i, feature in enumerate(features):
    X_test_permuted = X_test.copy()
    np.random.shuffle(X_test_permuted[:, i])

    acc = evaluate_model(X_test_permuted, y_test)
    drop = accuracy - acc
    importance.append((feature, drop))

importance.sort(key=lambda x: x[1], reverse=True)

print("\n🔥 Feature Importance (越大越重要):")
for f, imp in importance:
    print(f"{f}: {imp:.6f}")


'''
Epoch 1/100 | Train Loss: 0.2924 | Val Loss: 0.0978
Epoch 2/100 | Train Loss: 0.0846 | Val Loss: 0.0562
Epoch 3/100 | Train Loss: 0.0631 | Val Loss: 0.0467
Epoch 4/100 | Train Loss: 0.0563 | Val Loss: 0.0433
Epoch 5/100 | Train Loss: 0.0563 | Val Loss: 0.0417
Epoch 6/100 | Train Loss: 0.0532 | Val Loss: 0.0435
Epoch 7/100 | Train Loss: 0.0535 | Val Loss: 0.0450
Epoch 8/100 | Train Loss: 0.0512 | Val Loss: 0.0403
Epoch 9/100 | Train Loss: 0.0522 | Val Loss: 0.0444
Epoch 10/100 | Train Loss: 0.0502 | Val Loss: 0.0444
Epoch 11/100 | Train Loss: 0.0493 | Val Loss: 0.0450
Epoch 12/100 | Train Loss: 0.0486 | Val Loss: 0.0375
Epoch 13/100 | Train Loss: 0.0490 | Val Loss: 0.0393
Epoch 14/100 | Train Loss: 0.0480 | Val Loss: 0.0385
Epoch 15/100 | Train Loss: 0.0470 | Val Loss: 0.0373
Epoch 16/100 | Train Loss: 0.0462 | Val Loss: 0.0374
Epoch 17/100 | Train Loss: 0.0458 | Val Loss: 0.0459
Epoch 18/100 | Train Loss: 0.0448 | Val Loss: 0.0401
Epoch 19/100 | Train Loss: 0.0459 | Val Loss: 0.0415
Epoch 20/100 | Train Loss: 0.0450 | Val Loss: 0.0389
Epoch 21/100 | Train Loss: 0.0442 | Val Loss: 0.0388
Epoch 22/100 | Train Loss: 0.0438 | Val Loss: 0.0381
Epoch 23/100 | Train Loss: 0.0450 | Val Loss: 0.0401
Epoch 24/100 | Train Loss: 0.0429 | Val Loss: 0.0408
Epoch 25/100 | Train Loss: 0.0450 | Val Loss: 0.0381
Epoch 26/100 | Train Loss: 0.0440 | Val Loss: 0.0408
Epoch 27/100 | Train Loss: 0.0431 | Val Loss: 0.0412
Epoch 28/100 | Train Loss: 0.0440 | Val Loss: 0.0409
Epoch 29/100 | Train Loss: 0.0424 | Val Loss: 0.0363
Epoch 30/100 | Train Loss: 0.0426 | Val Loss: 0.0378
Epoch 31/100 | Train Loss: 0.0425 | Val Loss: 0.0378
Epoch 32/100 | Train Loss: 0.0420 | Val Loss: 0.0373
Epoch 33/100 | Train Loss: 0.0404 | Val Loss: 0.0372
Epoch 34/100 | Train Loss: 0.0416 | Val Loss: 0.0369
Epoch 35/100 | Train Loss: 0.0403 | Val Loss: 0.0407
Epoch 36/100 | Train Loss: 0.0409 | Val Loss: 0.0385
Epoch 37/100 | Train Loss: 0.0401 | Val Loss: 0.0387
Epoch 38/100 | Train Loss: 0.0413 | Val Loss: 0.0374
Epoch 39/100 | Train Loss: 0.0411 | Val Loss: 0.0378
Epoch 40/100 | Train Loss: 0.0388 | Val Loss: 0.0376
Epoch 41/100 | Train Loss: 0.0393 | Val Loss: 0.0381
Epoch 42/100 | Train Loss: 0.0386 | Val Loss: 0.0404
Epoch 43/100 | Train Loss: 0.0386 | Val Loss: 0.0386
Epoch 44/100 | Train Loss: 0.0377 | Val Loss: 0.0393
Epoch 45/100 | Train Loss: 0.0389 | Val Loss: 0.0370
Epoch 46/100 | Train Loss: 0.0379 | Val Loss: 0.0458
Epoch 47/100 | Train Loss: 0.0387 | Val Loss: 0.0383
Epoch 48/100 | Train Loss: 0.0381 | Val Loss: 0.0384
Epoch 49/100 | Train Loss: 0.0366 | Val Loss: 0.0367
Epoch 50/100 | Train Loss: 0.0360 | Val Loss: 0.0363
Epoch 51/100 | Train Loss: 0.0385 | Val Loss: 0.0386
Epoch 52/100 | Train Loss: 0.0366 | Val Loss: 0.0361
Epoch 53/100 | Train Loss: 0.0358 | Val Loss: 0.0357
Epoch 54/100 | Train Loss: 0.0362 | Val Loss: 0.0372
Epoch 55/100 | Train Loss: 0.0355 | Val Loss: 0.0366
Epoch 56/100 | Train Loss: 0.0360 | Val Loss: 0.0359
Epoch 57/100 | Train Loss: 0.0359 | Val Loss: 0.0361
Epoch 58/100 | Train Loss: 0.0350 | Val Loss: 0.0414
Epoch 59/100 | Train Loss: 0.0350 | Val Loss: 0.0350
Epoch 60/100 | Train Loss: 0.0347 | Val Loss: 0.0358
Epoch 61/100 | Train Loss: 0.0340 | Val Loss: 0.0358
Epoch 62/100 | Train Loss: 0.0353 | Val Loss: 0.0435
Epoch 63/100 | Train Loss: 0.0346 | Val Loss: 0.0357
Epoch 64/100 | Train Loss: 0.0340 | Val Loss: 0.0348
Epoch 65/100 | Train Loss: 0.0341 | Val Loss: 0.0380
Epoch 66/100 | Train Loss: 0.0338 | Val Loss: 0.0350
Epoch 67/100 | Train Loss: 0.0342 | Val Loss: 0.0341
Epoch 68/100 | Train Loss: 0.0327 | Val Loss: 0.0354
Epoch 69/100 | Train Loss: 0.0341 | Val Loss: 0.0359
Epoch 70/100 | Train Loss: 0.0326 | Val Loss: 0.0336
Epoch 71/100 | Train Loss: 0.0338 | Val Loss: 0.0352
Epoch 72/100 | Train Loss: 0.0337 | Val Loss: 0.0426
Epoch 73/100 | Train Loss: 0.0339 | Val Loss: 0.0367
Epoch 74/100 | Train Loss: 0.0329 | Val Loss: 0.0433
Epoch 75/100 | Train Loss: 0.0336 | Val Loss: 0.0406
Epoch 76/100 | Train Loss: 0.0336 | Val Loss: 0.0345
Epoch 77/100 | Train Loss: 0.0324 | Val Loss: 0.0347
Epoch 78/100 | Train Loss: 0.0320 | Val Loss: 0.0345
Epoch 79/100 | Train Loss: 0.0335 | Val Loss: 0.0335
Epoch 80/100 | Train Loss: 0.0328 | Val Loss: 0.0449
Epoch 81/100 | Train Loss: 0.0320 | Val Loss: 0.0349
Epoch 82/100 | Train Loss: 0.0319 | Val Loss: 0.0372
Epoch 83/100 | Train Loss: 0.0316 | Val Loss: 0.0356
Epoch 84/100 | Train Loss: 0.0312 | Val Loss: 0.0440
Epoch 85/100 | Train Loss: 0.0327 | Val Loss: 0.0342
Epoch 86/100 | Train Loss: 0.0322 | Val Loss: 0.0345
Epoch 87/100 | Train Loss: 0.0318 | Val Loss: 0.0325
Epoch 88/100 | Train Loss: 0.0319 | Val Loss: 0.0363
Epoch 89/100 | Train Loss: 0.0316 | Val Loss: 0.0359
Epoch 90/100 | Train Loss: 0.0321 | Val Loss: 0.0336
Epoch 91/100 | Train Loss: 0.0316 | Val Loss: 0.0338
Epoch 92/100 | Train Loss: 0.0311 | Val Loss: 0.0369
Epoch 93/100 | Train Loss: 0.0313 | Val Loss: 0.0359
Epoch 94/100 | Train Loss: 0.0308 | Val Loss: 0.0358
Epoch 95/100 | Train Loss: 0.0317 | Val Loss: 0.0388
Epoch 96/100 | Train Loss: 0.0316 | Val Loss: 0.0343
Epoch 97/100 | Train Loss: 0.0315 | Val Loss: 0.0346
Epoch 98/100 | Train Loss: 0.0310 | Val Loss: 0.0333
Epoch 99/100 | Train Loss: 0.0308 | Val Loss: 0.0351
Epoch 100/100 | Train Loss: 0.0315 | Val Loss: 0.0376

Classification Report:
              precision    recall  f1-score   support

         0.0       0.99      1.00      0.99       489
         1.0       1.00      0.99      0.99       576

    accuracy                           0.99      1065
   macro avg       0.99      0.99      0.99      1065
weighted avg       0.99      0.99      0.99      1065


Confusion Matrix:
[[488   1]
 [  6 570]]

✅ 模型已儲存到: /home/snow/DNS_dnsmasq_log/Spoofing_detect/dns_model(ttl30).pth

📊 Accuracy: 0.9934
🔥 F1 Score: 0.9939

🔥 Feature Importance (越大越重要):
ttl: 0.304225
response_time: 0.061972
packet_interval: 0.028169
is_private_ip: 0.023474
duplicate_txid: 0.003756

在採用 TXID 分組切分（Group-based splitting）後，
模型表現不降反升，推測原因為：

1. 同一 DNS 查詢事件不再被拆分至不同資料集，
   減少資料分布不一致問題

2. 修正標準化過程中的資料洩漏，使模型學習更貼近實際分布

然而，此結果仍需進一步驗證是否存在特徵過度主導
（如 TTL）或資料生成偏差，以確保模型具備真實泛化能力。




Classification Report:
              precision    recall  f1-score   support

         0.0       0.94      0.94      0.94       489
         1.0       0.95      0.95      0.95       576

    accuracy                           0.95      1065
   macro avg       0.95      0.94      0.95      1065
weighted avg       0.95      0.95      0.95      1065


Confusion Matrix:
[[458  31]
 [ 27 549]]

✅ 模型已儲存到: /home/snow/DNS_dnsmasq_log/Spoofing_detect/dns_model.pth

📊 Accuracy: 0.9455
🔥 F1 Score: 0.9498

🔥 Feature Importance (越大越重要):
response_time: 0.301408
is_private_ip: 0.166197
packet_interval: 0.107042
duplicate_txid: 0.077934

在移除 TTL 特徵後，模型 F1 score 由 0.99 降至約 0.95，
顯示模型並非僅依賴單一特徵進行分類，仍具備一定的泛化能力。

然而，特徵重要性分析顯示 response_time 成為主要判斷依據，
推測模型主要利用 DNS 回應延遲差異來區分正常與攻擊流量。

此結果顯示，在目前實驗環境中，
攻擊封包與正常封包在時間特徵上具有顯著差異，
但亦可能導致模型在面對延遲偽裝攻擊時效能下降。


實際測試時：
[[880, 120],
 [ 50, 950]]
 
 F1 Score ≈ 0.92
 
 
'''

