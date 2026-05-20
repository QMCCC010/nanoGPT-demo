"""
混合对话数据 + 指令微调数据 → .bin 训练文件
比例 60% 对话 + 40% 指令
"""
import os, pickle, random
import numpy as np
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer_movie.json")

# ==============================================
# 1. 加载对话数据
# ==============================================
dialogue_path = os.path.join("raw_data", "dialogue_text.txt")
with open(dialogue_path, "r", encoding="utf-8") as f:
    dialogue_pairs = [line for line in f.read().split("\n") if line.startswith("Q:") or line.startswith("A:")]

# 按 Q: + A: 组装
dialogues = []
for i in range(0, len(dialogue_pairs) - 1, 2):
    if dialogue_pairs[i].startswith("Q:") and dialogue_pairs[i + 1].startswith("A:"):
        dialogues.append(dialogue_pairs[i] + "\n" + dialogue_pairs[i + 1])

# 限制对话数量，防止淹没指令数据
random.shuffle(dialogues)
dialogues = dialogues[:50000]
print(f"对话: {len(dialogues)} 条")

# ==============================================
# 2. 加载指令数据
# ==============================================
sft_path = os.path.join("raw_data", "sft_data.txt")
with open(sft_path, "r", encoding="utf-8") as f:
    instructions = [line for line in f.read().split("\n") if line.startswith("Q:")]
# 构造完整的 Q+A 对
inst_pairs = []
for i, line in enumerate(instructions):
    if i + 1 < len(instructions):
        # SFT数据格式: 每行是 "Q: xxx\nA: xxx" 的完整对
        pass
# 实际上 sft_data.txt 每行就是 "Q: xxx\nA: xxx" 的完整对
with open(sft_path, "r", encoding="utf-8") as f:
    raw = f.read()
inst_pairs = [p.strip() for p in raw.split("\n\n") if "Q:" in p and "A:" in p]
# 如果有空行分隔，用上法；否则每两行为一组
if len(inst_pairs) < 1000:
    lines = raw.split("\n")
    inst_pairs = []
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith("Q:") and lines[i + 1].startswith("A:"):
            inst_pairs.append(lines[i] + "\n" + lines[i + 1])

print(f"指令: {len(inst_pairs)} 条")

# ==============================================
# 3. 混合
# ==============================================
all_data = dialogues + inst_pairs
random.shuffle(all_data)
text = "\n".join(all_data)
print(f"混合后: {len(all_data)} 条, {len(text)} 字符")

# ==============================================
# 4. 编码
# ==============================================
ids = tokenizer.encode(text).ids
data = np.array(ids, dtype=np.uint16)
n = int(0.95 * len(data))  # 5% 验证集
train_data = data[:n]
val_data = data[n:]

# ==============================================
# 5. 输出
# ==============================================
data_dir = os.path.join("data", "movie_sft")
os.makedirs(data_dir, exist_ok=True)
train_data.tofile(os.path.join(data_dir, "train.bin"))
val_data.tofile(os.path.join(data_dir, "val.bin"))

meta = {"vocab_size": tokenizer.get_vocab_size()}
with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print(f"train: {len(train_data)} tokens → {os.path.join(data_dir, 'train.bin')}")
print(f"val:   {len(val_data)} tokens → {os.path.join(data_dir, 'val.bin')}")
print("Done.")
