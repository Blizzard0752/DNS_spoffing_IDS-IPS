#dns_detector_realtime.py
from scapy.all import *
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import ipaddress

# =========================
# 1. 載入模型
# =========================
MODEL_PATH = "/home/snow/DNS_dnsmasq_log/Spoofing_detect/dns_model.pth"

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

print("✅ 模型載入完成")

# =========================
# 2. 狀態追蹤
# =========================
query_tracker = {}
response_counter = defaultdict(int)
last_packet_time = time.time()

# 畫圖資料
time_window = deque(maxlen=50)
normal_counts = deque(maxlen=50)
attack_counts = deque(maxlen=50)

normal_total = 0
attack_total = 0

# =========================
# 3. 工具函數
# =========================
def is_private(ip):
    try:
        return int(ipaddress.ip_address(ip).is_private)
    except:
        return 0

# =========================
# 4. 預測函數
# =========================
def predict(record):
    global normal_total, attack_total

    x = np.array([[record[f] for f in features]])
    x = scaler.transform(x)
    x = torch.FloatTensor(x)

    with torch.no_grad():
        prob = model(x).item()

    label = 1 if prob > 0.5 else 0

    if label == 1:
        attack_total += 1
    else:
        normal_total += 1

    return label, prob

# =========================
# 5. 畫圖初始化
# =========================
plt.ion()
fig, ax = plt.subplots()

# =========================
# 6. 更新圖表
# =========================
def update_plot():
    ax.clear()
    ax.plot(time_window, normal_counts, color="green", label="Normal")
    ax.plot(time_window, attack_counts, color="red", label="Attack")

    ax.set_xlabel("Time")
    ax.set_ylabel("Packet Count")
    ax.legend()

    total = normal_total + attack_total
    if total > 0:
        normal_ratio = normal_total / total * 100
        attack_ratio = attack_total / total * 100
        ax.set_title(f"Normal: {normal_ratio:.1f}% | Attack: {attack_ratio:.1f}%")

    plt.pause(0.01)

# =========================
# 7. DNS Callback
# =========================
def dns_callback(packet):
    global last_packet_time

    current_time = time.time()
    interval = current_time - last_packet_time
    last_packet_time = current_time

    if packet.haslayer(DNSQR) and packet[DNS].qr == 0:
        qname = packet[DNSQR].qname.decode().rstrip('.')
        query_tracker[(qname, packet[DNS].id)] = current_time

    elif packet.haslayer(DNSRR) and packet[DNS].qr == 1:
        qname = packet[DNSQR].qname.decode().rstrip('.')
        txid = packet[DNS].id
        key = (qname, txid)

        # response time
        if key in query_tracker:
            response_time = current_time - query_tracker[key]
        else:
            response_time = 0

        response_counter[key] += 1

        try:
            answer_ip = packet[DNSRR].rdata
        except:
            answer_ip = "0.0.0.0"

        record = {
            "response_time": response_time,
            "response_count": response_counter[key],
            "packet_interval": interval,
            "duplicate_txid": int(response_counter[key] > 1),
            "is_private_ip": is_private(answer_ip),
        }

        # 預測
        label, prob = predict(record)

        label_str = "ATTACK" if label == 1 else "NORMAL"
        color = "\033[91m" if label == 1 else "\033[92m"

        print(f"{color}[{label_str}] {qname} → {answer_ip} ({prob*100:.1f}%)\033[0m")

        # 更新圖表資料
        now = len(time_window)

        if label == 1:
            attack_counts.append(attack_counts[-1] + 1 if attack_counts else 1)
            normal_counts.append(normal_counts[-1] if normal_counts else 0)
        else:
            normal_counts.append(normal_counts[-1] + 1 if normal_counts else 1)
            attack_counts.append(attack_counts[-1] if attack_counts else 0)

        time_window.append(now)

        update_plot()

# =========================
# 8. 開始監聽
# =========================
print("🚀 即時 DNS 偵測器啟動...")
sniff(filter="udp port 53", prn=dns_callback, store=0)
