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
# 攻擊偵測參數
# =========================
WINDOW_SIZE      = 10
ATTACK_THRESHOLD = 7
NORMAL_THRESHOLD = 7
ATTACK_SEQ_TH    = 3
NORMAL_SEQ_TH    = 5

# ===== 攻擊偵測 =====
attack_state           = False
attack_seq_count       = 0
normal_seq_count       = 0
recent_labels          = deque(maxlen=10)
attack_spans           = []
current_attack_start   = None
first_attack_candidate = None
last_attack_seen       = None

# =========================
# ⭐ 新增：風險系統（累積版本）
# =========================
risk_value   = 0.0
risk_history = []
time_history = []
start_time   = time.time()

prob_buffer = []

# ⭐ 改成 max-based
RISK_GAIN = 0.08          # 👉 建議調小（因為現在是每次都加）
RISK_DECAY_NORMAL = 0.08

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
print("✅ ML DNS IPS（風險累積版本）啟動")

# =========================
# 狀態
# =========================
query_table      = {}
response_counter = defaultdict(int)
lock             = threading.Lock()

# =========================
# 原本圖表資料（保留，不刪）
# =========================
query_order_counter = 0
key_to_qidx         = {}
plot_data           = {}
attack_spans_qidx   = []
current_attack_start_qidx  = None

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
# DNS 回應
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
# forward query
# =========================
def forward_query(data, sport):
    pkt = IP(dst=DNS_SERVER) / UDP(sport=sport, dport=53) / DNS(data)
    send(pkt, verbose=0)

# =========================
# 收 response
# =========================
def scapy_response_handler(pkt):
    global attack_state, attack_seq_count, normal_seq_count
    global first_attack_candidate, last_attack_seen
    global prob_buffer

    if DNS not in pkt:
        return
    dns = pkt[DNS]
    if dns.qr != 1 or dns.ancount == 0:
        return

    qname = dns.qd.qname.decode(errors="ignore").strip(".")
    # ⭐ 過濾非目標 domain（關鍵修正）
    if not any(qname.endswith(d) for d in TARGET_DOMAINS):
        return
    txid  = dns.id
    key   = (txid, qname)

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

    # ⭐ 只收集，不直接算 risk
    with lock:
        prob_buffer.append(prob)

    if prob > 0.8:
        label = "ATTACK";     color = COLOR["RED"]
    elif prob > 0.4:
        label = "SUSPICIOUS"; color = COLOR["YELLOW"]
    else:
        label = "NORMAL";     color = COLOR["GREEN"]

    print(f"{color}[{label}] {qname} → {answer_ip} TTL={ttl} ({prob*100:.1f}%){COLOR['END']}")

    # ===== IPS 行為（完全保留）=====
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
# query handler
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
# socket server
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

    # ===== ⭐ 新圖：Risk 曲線 =====
    plt.ion()
    fig, ax = plt.subplots()

    # ⭐ 攻擊區塊偵測
    attack_regions = []
    in_attack = False
    attack_start_time = None

    while True:
        time.sleep(0.2)

        with lock:
            # =========================
            # ⭐ 改成 max 判斷（核心）
            # =========================
            if len(prob_buffer) == 0:
                max_prob = None
            else:
                max_prob = max(prob_buffer)

            # ⭐ 一定要清空（關鍵修正）
            prob_buffer.clear()
            # ⭐ 先算時間（⚠️ 提前）
            current_time = time.time() - start_time
            # ⭐ 風險更新
            if max_prob is None:
                pass

            elif max_prob > 0.8:
                risk_value += RISK_GAIN

                # ⭐ 開始攻擊區間
                if not in_attack:
                    in_attack = True
                    attack_start_time = time.time() - start_time
            else:
                risk_value -= RISK_DECAY_NORMAL

                # ⭐ 結束攻擊區間
                if in_attack:
                    in_attack = False
                    attack_regions.append((attack_start_time, current_time))

             # ⭐ 限制範圍
            risk_value = max(0, min(1, risk_value))
            risk_percent = risk_value * 100

            risk_history.append(risk_percent)
            time_history.append(current_time)

            x = list(time_history)
            y = list(risk_history)

        ax.clear()
        ax.plot(x, y, label="Raw Risk (%)", color='black')

        # ⭐ 畫攻擊區塊
        for start, end in attack_regions:
            ax.axvspan(start, end, alpha=0.2, color='red')

        # ⭐ 如果攻擊還沒結束（正在進行）
        if in_attack:
            ax.axvspan(attack_start_time, current_time, alpha=0.2, color='red')

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Risk (%)")
        ax.set_title("DNS Attack Accumulated Risk")
        ax.set_ylim(0, 110)

        plt.pause(0.001)


'''
即時威脅強度：
prob = σ(w1*ttl + w2*response_time + w3*interval + w4*is_private + ...)
risk_t = risk_{t-1} * decay(0.9) + prob * gain(0.25)
'''
