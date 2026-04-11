# dns_logger.py - 运行在客户端 10.11
from scapy.all import *
import csv
import time
from datetime import datetime

# 存储最近查询的时间，用于计算频率
recent_queries = {}

def dns_callback(packet):
    if packet.haslayer(DNSQR):  # 有DNS查询
        qname = packet[DNSQR].qname.decode('utf-8').rstrip('.')
        query_time = time.time()
        
        # 计算这个域名在最近5秒内的查询次数
        if qname not in recent_queries:
            recent_queries[qname] = []
        recent_queries[qname] = [t for t in recent_queries[qname] if query_time - t < 5]
        recent_queries[qname].append(query_time)
        freq_5s = len(recent_queries[qname])
        
        # 记录特征
        record = {
            'timestamp': datetime.now().isoformat(),
            'qname': qname,
            'qname_length': len(qname),
            'dot_count': qname.count('.'),
            'freq_last_5s': freq_5s,
            'src_ip': packet[IP].src,
            'dst_ip': packet[IP].dst
        }
        
        # 写入CSV
        with open('dns_normal.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(record)
        
        print(f"[NORMAL] {qname} - freq:{freq_5s}")

# 开始抓包（需要sudo）
sniff(filter="udp port 53", prn=dns_callback, store=0)
