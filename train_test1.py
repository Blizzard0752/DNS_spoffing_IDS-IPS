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
df = pd.read_csv(r"/home/snow/DNS_dnsmasq_log/Spoofing_detect/output(2:10).csv")

# =========================
# 先計算 answer_ip_variance
# =========================
group_key = ["txid", "qname"]

# 計算每個 query 對應的不同 IP 數量
ip_variance = df.groupby(["txid", "qname"])["answer_ip"].nunique()

# 變成一個欄位
df["answer_ip_variance"] = df.set_index(["txid", "qname"]).index.map(ip_variance)

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
    #"is_private_ip",
    "answer_ip_variance"
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
Epoch 1/100 | Train Loss: 0.3550 | Val Loss: 0.1308
Epoch 2/100 | Train Loss: 0.1223 | Val Loss: 0.1293
Epoch 3/100 | Train Loss: 0.1189 | Val Loss: 0.1289
Epoch 4/100 | Train Loss: 0.1148 | Val Loss: 0.1322
Epoch 5/100 | Train Loss: 0.1131 | Val Loss: 0.1256
Epoch 6/100 | Train Loss: 0.1086 | Val Loss: 0.1199
Epoch 7/100 | Train Loss: 0.1044 | Val Loss: 0.1229
Epoch 8/100 | Train Loss: 0.1034 | Val Loss: 0.1217
Epoch 9/100 | Train Loss: 0.1020 | Val Loss: 0.1150
Epoch 10/100 | Train Loss: 0.1027 | Val Loss: 0.1181
Epoch 11/100 | Train Loss: 0.1010 | Val Loss: 0.1130
Epoch 12/100 | Train Loss: 0.1031 | Val Loss: 0.1139
Epoch 13/100 | Train Loss: 0.1001 | Val Loss: 0.1099
Epoch 14/100 | Train Loss: 0.0988 | Val Loss: 0.1132
Epoch 15/100 | Train Loss: 0.0986 | Val Loss: 0.1089
Epoch 16/100 | Train Loss: 0.0987 | Val Loss: 0.1040
Epoch 17/100 | Train Loss: 0.0972 | Val Loss: 0.1152
Epoch 18/100 | Train Loss: 0.1006 | Val Loss: 0.1170
Epoch 19/100 | Train Loss: 0.0981 | Val Loss: 0.1112
Epoch 20/100 | Train Loss: 0.0979 | Val Loss: 0.1036
Epoch 21/100 | Train Loss: 0.0952 | Val Loss: 0.1040
Epoch 22/100 | Train Loss: 0.0974 | Val Loss: 0.1143
Epoch 23/100 | Train Loss: 0.0970 | Val Loss: 0.1018
Epoch 24/100 | Train Loss: 0.0970 | Val Loss: 0.1019
Epoch 25/100 | Train Loss: 0.0953 | Val Loss: 0.1074
Epoch 26/100 | Train Loss: 0.0980 | Val Loss: 0.1032
Epoch 27/100 | Train Loss: 0.0954 | Val Loss: 0.1102
Epoch 28/100 | Train Loss: 0.0941 | Val Loss: 0.1019
Epoch 29/100 | Train Loss: 0.0947 | Val Loss: 0.1064
Epoch 30/100 | Train Loss: 0.0958 | Val Loss: 0.1041
Epoch 31/100 | Train Loss: 0.0966 | Val Loss: 0.0996
Epoch 32/100 | Train Loss: 0.0963 | Val Loss: 0.1062
Epoch 33/100 | Train Loss: 0.0963 | Val Loss: 0.0951
Epoch 34/100 | Train Loss: 0.0941 | Val Loss: 0.1026
Epoch 35/100 | Train Loss: 0.0951 | Val Loss: 0.1049
Epoch 36/100 | Train Loss: 0.0925 | Val Loss: 0.0970
Epoch 37/100 | Train Loss: 0.0920 | Val Loss: 0.1044
Epoch 38/100 | Train Loss: 0.0954 | Val Loss: 0.0996
Epoch 39/100 | Train Loss: 0.0923 | Val Loss: 0.1033
Epoch 40/100 | Train Loss: 0.0939 | Val Loss: 0.0972
Epoch 41/100 | Train Loss: 0.0923 | Val Loss: 0.1007
Epoch 42/100 | Train Loss: 0.0936 | Val Loss: 0.0939
Epoch 43/100 | Train Loss: 0.0941 | Val Loss: 0.1000
Epoch 44/100 | Train Loss: 0.0929 | Val Loss: 0.0933
Epoch 45/100 | Train Loss: 0.0916 | Val Loss: 0.0930
Epoch 46/100 | Train Loss: 0.0923 | Val Loss: 0.1013
Epoch 47/100 | Train Loss: 0.0917 | Val Loss: 0.0968
Epoch 48/100 | Train Loss: 0.0921 | Val Loss: 0.0921
Epoch 49/100 | Train Loss: 0.0918 | Val Loss: 0.0950
Epoch 50/100 | Train Loss: 0.0922 | Val Loss: 0.1021
Epoch 51/100 | Train Loss: 0.0912 | Val Loss: 0.0935
Epoch 52/100 | Train Loss: 0.0908 | Val Loss: 0.0954
Epoch 53/100 | Train Loss: 0.0905 | Val Loss: 0.0946
Epoch 54/100 | Train Loss: 0.0907 | Val Loss: 0.0924
Epoch 55/100 | Train Loss: 0.0922 | Val Loss: 0.1053
Epoch 56/100 | Train Loss: 0.0923 | Val Loss: 0.1034
Epoch 57/100 | Train Loss: 0.0901 | Val Loss: 0.0924
Epoch 58/100 | Train Loss: 0.0923 | Val Loss: 0.0944
Epoch 59/100 | Train Loss: 0.0906 | Val Loss: 0.0844
Epoch 60/100 | Train Loss: 0.0907 | Val Loss: 0.0931
Epoch 61/100 | Train Loss: 0.0925 | Val Loss: 0.0939
Epoch 62/100 | Train Loss: 0.0897 | Val Loss: 0.0853
Epoch 63/100 | Train Loss: 0.0900 | Val Loss: 0.0861
Epoch 64/100 | Train Loss: 0.0897 | Val Loss: 0.0974
Epoch 65/100 | Train Loss: 0.0889 | Val Loss: 0.0924
Epoch 66/100 | Train Loss: 0.0898 | Val Loss: 0.0930
Epoch 67/100 | Train Loss: 0.0911 | Val Loss: 0.0991
Epoch 68/100 | Train Loss: 0.0898 | Val Loss: 0.0899
Epoch 69/100 | Train Loss: 0.0899 | Val Loss: 0.0906
Epoch 70/100 | Train Loss: 0.0885 | Val Loss: 0.0862
Epoch 71/100 | Train Loss: 0.0912 | Val Loss: 0.1013
Epoch 72/100 | Train Loss: 0.0911 | Val Loss: 0.0878
Epoch 73/100 | Train Loss: 0.0889 | Val Loss: 0.0914
Epoch 74/100 | Train Loss: 0.0883 | Val Loss: 0.0970
Epoch 75/100 | Train Loss: 0.0894 | Val Loss: 0.1030
Epoch 76/100 | Train Loss: 0.0904 | Val Loss: 0.0924
Epoch 77/100 | Train Loss: 0.0906 | Val Loss: 0.0964
Epoch 78/100 | Train Loss: 0.0899 | Val Loss: 0.0903
Epoch 79/100 | Train Loss: 0.0892 | Val Loss: 0.0914
Epoch 80/100 | Train Loss: 0.0889 | Val Loss: 0.1007
Epoch 81/100 | Train Loss: 0.0886 | Val Loss: 0.0925
Epoch 82/100 | Train Loss: 0.0905 | Val Loss: 0.1059
Epoch 83/100 | Train Loss: 0.0905 | Val Loss: 0.0950
Epoch 84/100 | Train Loss: 0.0882 | Val Loss: 0.0945
Epoch 85/100 | Train Loss: 0.0881 | Val Loss: 0.1037
Epoch 86/100 | Train Loss: 0.0898 | Val Loss: 0.1032
Epoch 87/100 | Train Loss: 0.0885 | Val Loss: 0.0905
Epoch 88/100 | Train Loss: 0.0868 | Val Loss: 0.0945
Epoch 89/100 | Train Loss: 0.0883 | Val Loss: 0.0964
Epoch 90/100 | Train Loss: 0.0910 | Val Loss: 0.0976
Epoch 91/100 | Train Loss: 0.0909 | Val Loss: 0.0927
Epoch 92/100 | Train Loss: 0.0882 | Val Loss: 0.0953
Epoch 93/100 | Train Loss: 0.0873 | Val Loss: 0.0981
Epoch 94/100 | Train Loss: 0.0887 | Val Loss: 0.1017
Epoch 95/100 | Train Loss: 0.0892 | Val Loss: 0.0990
Epoch 96/100 | Train Loss: 0.0896 | Val Loss: 0.1066
Epoch 97/100 | Train Loss: 0.0905 | Val Loss: 0.0901
Epoch 98/100 | Train Loss: 0.0876 | Val Loss: 0.0996
Epoch 99/100 | Train Loss: 0.0894 | Val Loss: 0.0942
Epoch 100/100 | Train Loss: 0.0910 | Val Loss: 0.0992

Classification Report:
              precision    recall  f1-score   support

         0.0       0.97      0.92      0.94       200
         1.0       0.95      0.98      0.96       301

    accuracy                           0.96       501
   macro avg       0.96      0.95      0.95       501
weighted avg       0.96      0.96      0.96       501


Confusion Matrix:
[[183  17]
 [  5 296]]

✅ 模型已儲存到: /home/snow/DNS_dnsmasq_log/Spoofing_detect/dns_model2.pth

📊 Accuracy: 0.9561
🔥 F1 Score: 0.9642

🔥 Feature Importance (越大越重要):
response_time: 0.197605
duplicate_txid: 0.145709
ttl: 0.097804
answer_ip_variance: 0.093812
packet_interval: 0.081836
response_count: 0.031936

'''
