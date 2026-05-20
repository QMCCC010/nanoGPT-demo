"""
下载 Dolly 15K 指令微调数据，格式化 Q&A
"""
import os, random

# Dolly 15K 的原始 JSONL URL
url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
output = os.path.join("raw_data", "sft_data.txt")

import urllib.request
import json

print("下载 Dolly 15K...")
# Dolly 约 55MB，用流式下载
urllib.request.urlretrieve(url, os.path.join("raw_data", "dolly.jsonl"))

pairs = []
with open(os.path.join("raw_data", "dolly.jsonl"), "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        q = item["instruction"].strip()
        a = item["response"].strip()
        # 只取有质量的问答对
        if len(q) > 5 and len(a) > 5 and len(a) < 500:
            pairs.append(f"Q: {q}\nA: {a}")

random.shuffle(pairs)
print(f"有效问答对: {len(pairs)}")

with open(output, "w", encoding="utf-8") as f:
    f.write("\n".join(pairs))
print(f"→ {output} ({len(pairs)} 条)")

# 样本
for i in range(3):
    print(f"\n--- 样本 {i+1} ---")
    print(pairs[i])
