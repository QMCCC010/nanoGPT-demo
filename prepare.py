import os
import pickle
import numpy as np

# 读取文本
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 字符级分词
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}

# 编码全部文本 → 整数数组 → uint16
data = np.array([stoi[c] for c in text], dtype=np.uint16)

# 9:1 划分
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 创建目录
data_dir = os.path.join('data', 'shakespeare_char')
os.makedirs(data_dir, exist_ok=True)

# 写入 bin 文件
train_data.tofile(os.path.join(data_dir, 'train.bin'))
val_data.tofile(os.path.join(data_dir, 'val.bin'))

# 存入元信息
meta = {'vocab_size': vocab_size}
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"vocab_size: {vocab_size}")
print(f"train: {len(train_data)} tokens → {os.path.join(data_dir, 'train.bin')}")
print(f"val:   {len(val_data)} tokens → {os.path.join(data_dir, 'val.bin')}")
print(f"meta:  {meta} → {os.path.join(data_dir, 'meta.pkl')}")
print("Done.")
