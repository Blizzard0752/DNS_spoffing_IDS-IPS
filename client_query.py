import socket
import time
import random

domains = ["google.com", "youtube.com", "github.com", 
           "stackoverflow.com", "wikipedia.org", "reddit.com",
           "amazon.com", "facebook.com", "twitter.com", "microsoft.com", "example.com"]

while True:
    # 随机选一个域名
    domain = random.choice(domains)
    try:
        # 发起DNS查询
        ip = socket.gethostbyname(domain)
        print(f"Query {domain} -> {ip}")
    except:
        print(f"Query {domain} failed")
    
    # 人类行为：有时快有时慢，有时连续查几次
    if random.random() < 0.3:  # 30%概率连续查
        time.sleep(random.uniform(0.1, 1))
    else:
        time.sleep(random.uniform(3, 5))

'''
sudo python3 /home/snow/VScode_programing/Python/畢業專題/detect/dns_query_simulator.py

'''
