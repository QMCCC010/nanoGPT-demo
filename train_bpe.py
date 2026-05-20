"""
在对话数据上训练 BPE 分词器，生成 .bin 训练数据
"""
import os
import pickle
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# ==============================================
# 1. 读取对话文本
# ==============================================
text_path = os.path.join("raw_data", "dialogue_text.txt")
with open(text_path, "r", encoding="utf-8") as f:
    text = f.read()
print(f"文本: {len(text)} 字符")

# ==============================================
# 2. 训练 BPE 分词器
# ==============================================
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=4096,
    special_tokens=["<|endoftext|>", "Q:", "A:"],
)

tokenizer.train_from_iterator([text], trainer)
tokenizer.save("tokenizer_movie.json")
vocab_size = tokenizer.get_vocab_size()
print(f"BPE 词表大小: {vocab_size}")

# 测试编码
enc = tokenizer.encode("Q: What is your name?\nA: My name is Anna.")
print(f"测试编码: {enc.tokens}")

# ==============================================
# 3. 编码全部文本
# ==============================================
ids = tokenizer.encode(text).ids
data = np.array(ids, dtype=np.uint16)
print(f"编码后 token 数: {len(data)}")

# ==============================================
# 4. 训练/验证集划分
# ==============================================
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ==============================================
# 5. 输出 .bin 文件
# ==============================================
data_dir = os.path.join("data", "movie_dialogs")
os.makedirs(data_dir, exist_ok=True)

train_data.tofile(os.path.join(data_dir, "train.bin"))
val_data.tofile(os.path.join(data_dir, "val.bin"))

meta = {"vocab_size": vocab_size}
with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print(f"train: {len(train_data)} tokens → {os.path.join(data_dir, 'train.bin')}")
print(f"val:   {len(val_data)} tokens → {os.path.join(data_dir, 'val.bin')}")
print("Done.")
