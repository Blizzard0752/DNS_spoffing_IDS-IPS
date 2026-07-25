# Machine Learning-based DNS Spoofing Detection and Defense System

A real-time DNS Spoofing detection and defense system based on **Machine Learning (PyTorch MLP)**. This project detects forged DNS responses from the **client-side** by analyzing DNS packet characteristics and automatically blocks suspicious responses through a DNS Proxy.

> 🎓 Graduation Project, Department of Computer Science and Information Engineering

---

# 📖 Overview

DNS Spoofing (DNS Cache Poisoning) is one of the most common attacks against the Domain Name System. Attackers forge fake DNS responses before the legitimate DNS server replies, causing victims to resolve malicious IP addresses.

Traditional defense mechanisms such as **DNSSEC** require server-side deployment and are not universally adopted. Therefore, this project proposes a lightweight **client-side detection mechanism** that combines packet analysis with machine learning to identify spoofed DNS responses in real time.

The proposed system intercepts DNS packets through a DNS Proxy, extracts network features, predicts attack probabilities using a trained MLP model, and decides whether to forward or block each DNS response.

---

# ✨ Features

- Real-time DNS Proxy
- DNS Packet Sniffing
- DNS Feature Extraction
- Machine Learning-based Detection
- Real-time DNS Response Filtering
- Dynamic Risk Score Mechanism
- Live Risk Visualization
- Lightweight Edge Deployment

---

# 🏗 System Architecture

```
                +------------------+
                |      Client      |
                +------------------+
                         |
                    DNS Query
                         |
                         ▼
                +------------------+
                |    DNS Proxy     |
                +------------------+
                  │            │
                  │            │
                  ▼            ▼
          Forward Query   Packet Sniffer
                  │            │
                  │            ▼
                  │     Feature Extraction
                  │            │
                  ▼            ▼
             DNS Server    MLP Detection
                               │
                    ┌──────────┴──────────┐
                    │                     │
                 Normal              Suspicious
                    │                     │
                    ▼                     ▼
             Return Response        Block Response
```

---

# 🔄 Workflow

1. Client sends a DNS Query.
2. DNS Proxy intercepts the request.
3. Proxy forwards the query to the DNS Server.
4. Packet Sniffer captures DNS responses.
5. Features are extracted from captured packets.
6. The trained MLP predicts the probability of DNS Spoofing.
7. The proxy decides whether to:
   - Forward the response
   - Block the response
8. Update the dynamic Risk Score.

---

# 📂 Project Structure

```
.
├── dataset/                # Training datasets
├── model/                  # Trained machine learning models
├── proxy/                  # DNS Proxy implementation
├── sniffer/                # Packet capture module
├── training/               # Model training scripts
├── utils/                  # Feature extraction utilities
├── visualization/          # Risk score visualization
├── results/                # Experimental results
├── README.md
└── requirements.txt
```

---

# ⚙️ Technologies

## Programming Language

- Python

## Machine Learning

- PyTorch
- Scikit-learn

## Network Analysis

- Scapy
- Socket Programming

## Data Processing

- NumPy
- Pandas

## Visualization

- Matplotlib

---

# 📊 Feature Engineering

The final model uses five lightweight network features that can be extracted in real time without modifying DNS servers.

| Feature | Description |
|----------|-------------|
| ttl | DNS Time-To-Live value |
| response_time | DNS response latency |
| packet_interval | Interval between packets |
| is_private_ip | Whether the response IP is private |
| duplicate_txid | Duplicate DNS Transaction ID |

These features were selected after multiple rounds of feature engineering and experimental evaluation.

---

# 🤖 Machine Learning Model

The project adopts a **Multi-Layer Perceptron (MLP)** implemented with PyTorch.

## Network Architecture

```
Input Layer
      │
      ▼
Linear (64)
      │
    ReLU
      │
      ▼
Linear (32)
      │
    ReLU
      │
      ▼
Linear (16)
      │
    ReLU
      │
      ▼
Linear (1)
      │
   Sigmoid
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | BCELoss |
| Learning Rate | 0.001 |
| Epoch | 100 |

---

# 📈 Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | **98.40%** |
| F1 Score | **98.66%** |

The trained model successfully distinguishes legitimate DNS responses from spoofed responses while maintaining high detection accuracy in real-time environments.

---

# 🛡 Dynamic Risk Score

Instead of relying solely on a single prediction, the system continuously accumulates attack probabilities into a dynamic Risk Score.

The mechanism provides:

- Continuous attack monitoring
- Reduced false positives
- Better stability during transient network fluctuations
- Automatic recovery after attacks disappear

Risk Score behavior:

```
Attack Probability
        │
        ▼
+--------------------+
|  Risk Accumulator  |
+--------------------+
        │
        ▼
Current Risk Value
        │
        ▼
Threshold Decision
        │
   ┌────┴────┐
   │         │
Allow      Block
```

---

# 🌐 Experimental Environment

The experiments were conducted in an isolated LAN environment.

| Device | Role |
|---------|------|
| Ubuntu VM | DNS Server (dnsmasq) |
| Windows | DNS Spoofing Attacker |
| Jetson Orin Nano | Client |
| Jetson Orin Nano | DNS Proxy |

The attacker injects forged DNS responses while the proxy analyzes and filters packets in real time.

---

# 🚀 Project Highlights

✅ Client-side DNS Spoofing Detection

✅ Real-time DNS Proxy Protection

✅ Lightweight PyTorch MLP Model

✅ Packet Feature Engineering

✅ Dynamic Risk Score Mechanism

✅ Edge Device Deployment

✅ Real-time Packet Filtering

---

# 📷 Demonstration

The project includes:

- Real-time DNS packet capture
- Live attack probability prediction
- Risk Score visualization
- Automatic spoofed response blocking

*(Add screenshots or GIF demonstrations here.)*

---

# 🔮 Future Work

Possible future improvements include:

- DNS over HTTPS (DoH) detection
- DNS over TLS (DoT) support
- Federated Learning-based collaborative detection
- Online Learning
- Multi-class DNS attack classification
- Transformer-based anomaly detection
- Containerized deployment with Docker

---

# 📚 Citation

If you use this project for academic or research purposes, please cite this repository appropriately.

---

# 👨‍💻 Author

**Chen Kuan-Hsuan**

Department of Computer Science and Information Engineering

Graduation Project

Machine Learning-based DNS Spoofing Detection and Defense System

---

# 📄 License

This project is released for academic and educational purposes.
