from scapy.all import *
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import ipaddress
import threading

# =========================
# 1. 載入模型
# =========================
MODEL_PATH = "/home/snow/DNS_dnsmasq_log/Spoofing_detect/dns_model(35000).pth"

checkpoint = torch.load(MODEL_PATH, weights_only=False)

features = checkpoint["features"]
scaler = checkpoint["scaler"]

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
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("✅ 模型載入完成（純 ML 判斷）")

# =========================
# 2. 狀態追蹤
# =========================
query_tracker = {}
response_counter = defaultdict(int)
last_packet_time = time.time()

# =========================
# 3. 時間窗
# =========================
WINDOW_SIZE = 0.2

current_normal = 0
current_suspicious = 0   # ✅ 新增
current_attack = 0

# =========================
# 4. 畫圖資料
# =========================
time_window = deque(maxlen=50)
normal_counts = deque(maxlen=50)
suspicious_counts = deque(maxlen=50)  # ✅ 新增
attack_counts = deque(maxlen=50)

time_index = 0

# =========================
# 5. 工具
# =========================
def is_private(ip):
    try:
        return int(ipaddress.ip_address(ip).is_private)
    except:
        return 0

# =========================
# 6. ML 預測（只回傳 prob）
# =========================
def predict_ml(record):
    x = np.array([[record[f] for f in features]])
    x = scaler.transform(x)
    x = torch.FloatTensor(x)

    with torch.no_grad():
        prob = model(x).item()

    return prob   # ✅ 只回傳機率

# =========================
# 7. DNS callback
# =========================
def dns_callback(packet):
    global last_packet_time
    global current_normal, current_suspicious, current_attack

    current_time = time.time()
    interval = current_time - last_packet_time
    last_packet_time = current_time

    if packet.haslayer(DNSQR) and packet[DNS].qr == 0:
        qname = packet[DNSQR].qname.decode(errors="ignore").rstrip('.')
        query_tracker[(qname, packet[DNS].id)] = current_time

    elif packet.haslayer(DNSRR) and packet[DNS].qr == 1:
        try:
            qname = packet[DNSQR].qname.decode(errors="ignore").rstrip('.')
        except:
            return

        txid = packet[DNS].id
        key = (qname, txid)

        if key in query_tracker:
            response_time = current_time - query_tracker[key]
        else:
            response_time = 0

        response_counter[key] += 1

        try:
            ttl = packet[DNSRR].ttl
        except:
            ttl = 0

        try:
            answer_ip = packet[DNSRR].rdata
            if isinstance(answer_ip, bytes):
                answer_ip = answer_ip.decode()
        except:
            answer_ip = "0.0.0.0"

        record = {
            "response_time": response_time,
            #"response_count": response_counter[key],
            "ttl": ttl,
            "packet_interval": interval,
            "duplicate_txid": int(response_counter[key] > 1),
            "is_private_ip": is_private(answer_ip),
        }

        # 🔥 ML
        prob = predict_ml(record)

        # 🔥 三分類
        if prob > 0.8:
            color = "\033[91m"
            label_str = "ATTACK"
            current_attack += 1

        elif prob > 0.5:
            color = "\033[93m"
            label_str = "SUSPICIOUS"
            current_suspicious += 1   # ✅ 必須加

        else:
            color = "\033[92m"
            label_str = "NORMAL"
            current_normal += 1

        print(f"{color}[{label_str}] {qname} → {answer_ip} TTL={ttl} ({prob*100:.1f}%)\033[0m")

# =========================
# 8. Sniff Thread
# =========================
def sniff_thread():
    sniff(filter="udp port 53", prn=dns_callback, store=0)

# =========================
# 9. 主執行緒畫圖
# =========================
plt.ion()
fig, ax = plt.subplots()

print("🚀 即時 DNS 偵測器（含 Suspicious）啟動...")

t = threading.Thread(target=sniff_thread, daemon=True)
t.start()

while True:
    time.sleep(WINDOW_SIZE)

    time_index += 1

    time_window.append(time_index)
    normal_counts.append(current_normal)
    suspicious_counts.append(current_suspicious)  # ✅ 新增
    attack_counts.append(current_attack)

    current_normal = 0
    current_suspicious = 0   # ✅ reset
    current_attack = 0

    ax.clear()
    ax.plot(time_window, normal_counts, color="green", label="Normal")
    ax.plot(time_window, suspicious_counts, color="orange", label="Suspicious")  # ✅ 新線
    ax.plot(time_window, attack_counts, color="red", label="Attack")

    ax.set_xlabel("Time (0.2s per step)")
    ax.set_ylabel("Packet Count")
    ax.legend()

    plt.pause(0.001)
