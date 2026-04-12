#merge_data.py
import pandas as pd

# =========================
# 1. 讀資料
# =========================
df_normal = pd.read_csv("/home/snow/DNS_dnsmasq_log/Spoofing_detect/1000N_spoof_dataset.csv")
df_attack = pd.read_csv("/home/snow/DNS_dnsmasq_log/Spoofing_detect/1000A_spoof_dataset.csv")

# =========================
# 2. 正常資料 → label = 0
# =========================
df_normal['label'] = 0

# =========================
# 3. 攻擊資料 → 根據 TTL 判斷
# =========================
# ⚠️ 重點：不要全部設 1！

df_attack['label'] = df_attack['ttl'].apply(lambda x: 1 if x == 0 else 0)

# =========================
# 4. 合併
# =========================
df = pd.concat([df_normal, df_attack], ignore_index=True)

# =========================
# 5. 清理資料（建議）
# =========================

# 移除 response_time = -1（無法匹配的）
df = df[df['response_time'] >= 0]

# 移除異常值（可選）
df = df[df['response_time'] < 1]

# =========================
# 6. 打亂
# =========================
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# =========================
# 7. 存檔
# =========================
output_path = "/home/snow/DNS_dnsmasq_log/Spoofing_detect/dns_dataset.csv"
df.to_csv(output_path, index=False)

# =========================
# 8. 統計
# =========================
print("📊 合併完成")
print(df['label'].value_counts())

print("\n📈 檢查 TTL 分布:")
print(df.groupby('label')['ttl'].mean())

print("\n📈 檢查 response_count:")
print(df.groupby('label')['response_count'].mean())
