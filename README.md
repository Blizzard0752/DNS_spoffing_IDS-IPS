# DNS_spoffing_detect

第一步：封包抓什么？（最关键的差异点）
根据最新的学术研究，恶意DNS流量和正常流量的核心差异不在单个封包，而在"行为模式"。下表是你实验中最容易抓到的具体差异：

对比维度	正常封包的特征	被Spoofing的封包特征	如何抓到这个差异
查询频率	访问一个域名是零散的，比如查google.com后过很久才再查	攻击时短时间内连续大量查询同一个域名（如每秒几十次）	在10.11上用tshark记录时间戳，统计同一个域名的请求间隔
域名的"长相"	正常：www.google.com、api.github.com（有意义的单词）	恶意：2po3asvtjvfkebjuke4qsvf3ja6agsznrt.tunnel.com（随机字符串）	记录QNAME字段，计算其中的字母随机程度
TTL存活时间	正常：300秒、3600秒（比较长）	恶意：常常设为0秒（不让缓存，以便随时换IP）	抓响应封包里的TTL字段
响应码	大部分是NOERROR(0)	如果是DGA攻击，会出现大量NXDOMAIN(3)错误	记录RCODE字段
📊 你的实验场景中，这4个差异会如何体现？
攻击方式：假设攻击端(10.13)用Ettercap做DNS欺骗，它会拦截客户端(10.11)的查询，返回伪造的IP地址

客户端现象：10.11原本查demo.test应该得到10.10的IP，攻击后会变成10.13的IP

封包特征：你会看到同一个域名demo.test在短时间内被反复查询，而且TTL可能异常

🔧 第二步：实验流程（手把手操作）
阶段一：建立"正常"封包基线
在客户端(10.11)上运行抓包命令：

bash
sudo tshark -i eth0 -f "udp port 53" -w normal_traffic.pcap
执行正常行为：用浏览器访问几个真实网站、ping几个域名

抓5-10分钟，产生几百到几千条DNS查询就够了

阶段二：制造并抓取"被欺骗"封包
在攻击端(10.13)启动欺骗（参考你之前的环境配置）

同时在客户端(10.11)抓包：

bash
sudo tshark -i eth0 -f "udp port 53" -w spoofed_traffic.pcap
在客户端反复访问被攻击的域名（比如nslookup demo.test）

抓5-10分钟

阶段三：从pcap文件中提取特征（这一步最关键）
写一个Python脚本（我后面可以帮你写），把每个DNS封包转成一行数字特征：

text
timestamp | src_ip | qname | qname_entropy | ttl | rcode | query_frequency
具体计算方式：

qname_entropy：用信息熵公式算域名的随机程度，正常域名熵值低，随机字符串熵值高

query_frequency：统计过去5秒内同一个域名的查询次数

ttl：直接从响应封包里读

🧠 第三步：神经网络架构建议（别想太复杂）
考虑到你是自己跑数据，数据集可能不会太大（几百到几千条），我建议用最简单但有效的架构：

方案A：多层感知机(MLP)——最稳妥的选择
text
输入层(特征数) → 隐藏层1(64个神经元) → 隐藏层2(32个) → 输出层(2分类：正常/攻击)
优点：数据量小也能训练，容易调试，足够应付你的实验规模



你的实验流程（完整版）
阶段一：产生正常流量
客户端(10.11)运行上面的dns_logger.py开始记录

运行查询脚本（模拟人类行为）

记录5-10分钟，得到dns_normal.csv

阶段二：产生攻击流量
攻击端(10.13)启动DNS欺骗（Ettercap或arpspoof）

客户端(10.11)运行另一个监听程序（同样的代码，但输出到dns_spoofed.csv）

客户端继续查询（脚本不变）

记录5-10分钟

你最终得到的CSV格式：
timestamp	qname	qname_length	dot_count	freq_last_5s	src_ip	dst_ip
2024-01-15T10:00:01	google.com	10	1	1	192.168.10.11	192.168.10.10
2024-01-15T10:00:02	google.com	10	1	2	192.168.10.11	192.168.10.10

