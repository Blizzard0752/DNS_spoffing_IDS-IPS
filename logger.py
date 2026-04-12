# dns_logger_advanced.py
from scapy.all import *
import csv
import time
from datetime import datetime
from collections import defaultdict
import ipaddress

# 記錄 query 發送時間
query_tracker = {}

# 記錄 response 數量
response_counter = defaultdict(int)

# 記錄上一個封包時間（算 interval）
last_packet_time = time.time()

# 判斷是否 private IP
def is_private(ip):
    try:
        return int(ipaddress.ip_address(ip).is_private)
    except:
        return 0

def dns_callback(packet):
    global last_packet_time

    current_time = time.time()
    interval = current_time - last_packet_time
    last_packet_time = current_time

    # =========================
    # DNS QUERY
    # =========================
    if packet.haslayer(DNSQR) and packet[DNS].qr == 0:
        qname = packet[DNSQR].qname.decode().rstrip('.')

        query_tracker[(qname, packet[DNS].id)] = current_time

        print(f"[QUERY] {qname}")

    # =========================
    # DNS RESPONSE
    # =========================
    elif packet.haslayer(DNSRR) and packet[DNS].qr == 1:
        qname = packet[DNSQR].qname.decode().rstrip('.')
        txid = packet[DNS].id

        key = (qname, txid)

        # 計算 response time
        if key in query_tracker:
            response_time = current_time - query_tracker[key]
        else:
            response_time = -1  # 沒對到

        # 計算 response 次數
        response_counter[key] += 1

        # 取 answer IP
        try:
            answer_ip = packet[DNSRR].rdata
        except:
            answer_ip = "0.0.0.0"

        # TTL
        ttl = packet[DNSRR].ttl if packet.haslayer(DNSRR) else 0

        record = {
            "timestamp": datetime.now().isoformat(),
            "qname": qname,
            "txid": txid,

            # ===== 原本的 =====
            "qname_length": len(qname),
            "dot_count": qname.count("."),

            # ===== 新增關鍵特徵 =====
            "response_time": response_time,
            "response_count": response_counter[key],
            "ttl": ttl,
            "answer_ip": answer_ip,
            "is_private_ip": is_private(answer_ip),

            # ===== 行為特徵 =====
            "packet_interval": interval,
            "duplicate_txid": int(response_counter[key] > 1),

            # ===== network =====
            "src_ip": packet[IP].src,
            "dst_ip": packet[IP].dst,
        }

        write_to_csv(record)

        print(f"[RESP] {qname} -> {answer_ip} | ttl={ttl} | t={response_time:.4f} | count={response_counter[key]}")

def write_to_csv(record):
    filename = r"/home/snow/DNS_dnsmasq_log/Spoofing_detect/spoof_dataset.csv"

    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(record)

# 啟動
print("開始監聽 DNS（進階特徵版）...")
sniff(filter="udp port 53", prn=dns_callback, store=0)
