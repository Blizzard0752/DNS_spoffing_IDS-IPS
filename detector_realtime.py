from scapy.all import *
from scapy.layers.dns import DNS, DNSQR, DNSRR
import socket
import time
import threading
import torch
import torch.nn as nn
import numpy as np
import ipaddress
from collections import defaultdict, deque
import matplotlib.pyplot as plt

# =========================
# 設定
# =========================
MODEL_PATH  = "/home/snow/DNS_dnsmasq_log/Spoofing_detect/dns_model(ttl30).pth"
DNS_SERVER  = "192.168.10.10"
INTERFACE   = "enP8p1s0"
PROXY_PORT  = 53
BUFSIZE     = 4096
COLOR = {
    "GREEN":  "\033[92m",
    "YELLOW": "\033[93m",
    "RED":    "\033[91m",
    "WHITE":  "\033[97m",
    "END":    "\033[0m"
}
TARGET_DOMAINS = {
    "google.com", "youtube.com", "github.com",
    "stackoverflow.com", "wikipedia.org", "reddit.com",
    "amazon.com", "facebook.com", "twitter.com",
    "microsoft.com", "example.com"
}

# =========================
# 風險系統參數
# =========================
PROB_WINDOW   = deque(maxlen=4)  # 滑動窗口，最近 20 個封包
ATTACK_TH     = 0.5              # 攻擊判定門檻 θ
RISE_RATE     = 0.15              # 上升速率
DECAY_RATE    = 0.05              # 下降速率
EMA_ALPHA     = 0.7              # EMA 平滑係數

risk_value   = 0.0
ema_risk     = 0.0
risk_history = []
time_history = []
start_time   = time.time()
prob_buffer  = []

# =========================
# 攻擊區間記錄
# =========================
attack_regions    = []
in_attack         = False
attack_start_time = None

# =========================
# 載入模型
# =========================
checkpoint = torch.load(MODEL_PATH, weights_only=False)
features   = checkpoint["features"]
scaler     = checkpoint["scaler"]

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 16),        nn.ReLU(),
            nn.Linear(16, 1),         nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = MLP(len(features))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("✅ ML DNS IPS 啟動")

# =========================
# 狀態
# =========================
query_table      = {}
response_counter = defaultdict(int)
lock             = threading.Lock()

# =========================
# 工具
# =========================
def is_private(ip):
    try:
        return int(ipaddress.ip_address(ip).is_private)
    except:
        return 0

def predict_prob(record):
    x = np.array([[record[f] for f in features]])
    x = scaler.transform(x)
    x = torch.FloatTensor(x)
    with torch.no_grad():
        return model(x).item()

# =========================
# DNS 回應建構
# =========================
def build_response(qname, txid, ip):
    dns = DNS(
        id=txid, qr=1, aa=0, rd=1, ra=1, rcode=0,
        qd=DNSQR(qname=qname),
        an=DNSRR(rrname=qname, ttl=60, rdata=ip),
        ancount=1
    )
    return bytes(dns)

# =========================
# 轉發查詢
# =========================
def forward_query(data, sport):
    pkt = IP(dst=DNS_SERVER) / UDP(sport=sport, dport=53) / DNS(data)
    send(pkt, verbose=0)

# =========================
# 收 DNS 回應（Scapy sniff）
# =========================
def scapy_response_handler(pkt):
    if DNS not in pkt:
        return
    dns = pkt[DNS]
    if dns.qr != 1 or dns.ancount == 0:
        return

    qname = dns.qd.qname.decode(errors="ignore").strip(".")
    if not any(qname.endswith(d) for d in TARGET_DOMAINS):
        return

    txid = dns.id
    key  = (txid, qname)

    with lock:
        if key not in query_table:
            return

        now       = time.time()
        last_time = query_table[key].get("last_time", query_table[key]["time"])
        interval  = now - last_time
        query_table[key]["last_time"] = now

        response_time = now - query_table[key]["time"]
        response_counter[key] += 1

        ttl       = getattr(dns.an, "ttl", 0)
        answer_ip = dns.an.rdata
        if isinstance(answer_ip, bytes):
            answer_ip = answer_ip.decode()

        record = {
            "response_time":   response_time,
            "ttl":             ttl,
            "packet_interval": interval,
            "duplicate_txid":  int(response_counter[key] > 1),
            "is_private_ip":   is_private(answer_ip),
        }

    prob = predict_prob(record)

    with lock:
        prob_buffer.append(prob)

    if prob > 0.8:
        label = "ATTACK";     color = COLOR["RED"]
    elif prob > 0.4:
        label = "SUSPICIOUS"; color = COLOR["YELLOW"]
    else:
        label = "NORMAL";     color = COLOR["GREEN"]

    print(f"{color}[{label}] {qname} → {answer_ip} TTL={ttl} ({prob*100:.1f}%){COLOR['END']}")

    with lock:
        if label == "NORMAL":
            if not query_table[key]["done"]:
                reply = build_response(qname, txid, answer_ip)
                query_table[key]["sock"].sendto(reply, query_table[key]["client_addr"])
                print(f"{COLOR['GREEN']}[ALLOW] {qname}{COLOR['END']}")
                query_table[key]["done"] = True
        else:
            print(f"{COLOR['RED']}[BLOCK] {qname}{COLOR['END']}")

# =========================
# 處理 Client 查詢
# =========================
def handle_query(data, client_addr, sock):
    dns = DNS(data)
    if dns.qd is None:
        return
    qname = dns.qd.qname.decode(errors="ignore").strip(".")
    txid  = dns.id
    key   = (txid, qname)
    with lock:
        query_table[key] = {
            "time":        time.time(),
            "last_time":   time.time(),
            "client_addr": client_addr,
            "sock":        sock,
            "done":        False
        }
    print(f"{COLOR['WHITE']}[QUERY] {qname}{COLOR['END']}")
    forward_query(data, client_addr[1])

# =========================
# UDP Socket 監聽
# =========================
def socket_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PROXY_PORT))
    while True:
        data, addr = sock.recvfrom(BUFSIZE)
        threading.Thread(target=handle_query, args=(data, addr, sock), daemon=True).start()

# =========================
# main
# =========================
if __name__ == "__main__":
    threading.Thread(
        target=lambda: sniff(
            iface=INTERFACE,
            prn=scapy_response_handler,
            store=0,
            filter="udp port 53"
        ),
        daemon=True
    ).start()
    threading.Thread(target=socket_listener, daemon=True).start()

    plt.ion()
    fig, ax = plt.subplots()

    while True:
        time.sleep(0.2)

        with lock:
            current_time = time.time() - start_time

            # Step 1：把新封包機率加入滑動窗口
            for p in prob_buffer:
                PROB_WINDOW.append(p)
            prob_buffer.clear()

            # Step 2：計算滑動窗口平均機率
            # avg_prob = (1/N) * sum(p_i), N = min(|buffer|, 20)
            if len(PROB_WINDOW) == 0:
                avg_prob = 0.0
            else:
                avg_prob = sum(PROB_WINDOW) / len(PROB_WINDOW)

            # Step 3：線性風險增量
            # avg_prob > θ → Δ = +RISE_RATE × avg_prob
            # avg_prob ≤ θ → Δ = -DECAY_RATE × (1 - avg_prob)
            if avg_prob > ATTACK_TH:
                delta = RISE_RATE * avg_prob

                if not in_attack:
                    in_attack         = True
                    attack_start_time = current_time
            else:
                # 🔥 新增：快速下降條件
                if avg_prob < 0.3:
                    delta = -0.05   # ← 直接快速掉
                else:
                    delta = -DECAY_RATE * (1 - avg_prob)

                if in_attack:
                    in_attack = False
                    attack_regions.append((attack_start_time, current_time))

            # Step 4：更新風險值，限制在 [0, 1]
            # risk_t = clip(risk_{t-1} + Δ, 0, 1)
            risk_value = max(0.0, min(1.0, risk_value + delta))

            # Step 5：EMA 平滑
            # EMA_t = α × risk_t + (1 - α) × EMA_{t-1}
            ema_risk = EMA_ALPHA * risk_value + (1 - EMA_ALPHA) * ema_risk

            # Step 6：輸出成百分比
            risk_history.append(ema_risk * 100)
            time_history.append(current_time)

            x = list(time_history)
            y = list(risk_history)

        ax.clear()
        ax.plot(x, y, color='black', label="Risk (%)")

        for start, end in attack_regions:
            ax.axvspan(start, end, alpha=0.2, color='red')
        if in_attack:
            ax.axvspan(attack_start_time, current_time, alpha=0.2, color='red')

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Risk (%)")
        ax.set_title("DNS Attack Risk (Linear)")
        ax.set_ylim(0, 110)

        plt.pause(0.001)


'''
sudo $(which python) /home/snow/VScode_programing/Python/畢業專題/detect/defence_realtime.py
'''
