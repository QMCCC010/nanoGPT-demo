import urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参数
batch_size = 32
block_size = 8
max_iters = 10000
eval_interval = 500
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # 一次用200个loss的平均值来评估
n_embd = 32 # 词嵌入大小（即每个token转化为向量的大小）
# ---------------------------------------------------------

torch.manual_seed(42)

# ---------------------------------------------------------
# 下载数据集
# url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# filename = "input.txt"
#
# urllib.request.urlretrieve(url, filename)
# print(f"文件已下载并保存为: {filename}")
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# ---------------------------------------------------------

# 创建token集
chars = sorted(list(set(text)))
vocab_size = len(chars)
# 字符编码器和解码器
stoi = { ch : i for i, ch in enumerate(chars)}
itos = { i : ch for i, ch in enumerate(chars)}
def encode(s):
    return [stoi[c] for c in s]
def decode(ids):
    return ''.join(itos[i] for i in ids)
# 编码整个莎士比亚数据集
data = torch.tensor(encode(text), dtype=torch.long)

# ---------------------------------------------------------

# 划分训练集和验证集
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# 读取数据函数
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# 评估损失函数
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out

# ---------------------------------------------------------

# 二元语言模型
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # 词嵌入表
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # 位置嵌入表
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # pytorch中交叉熵损失函数要求通道数在第二维表示，因此对张量形状进行改变
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx是初始文本的token ID
        for _ in range(max_new_tokens):
            # 前向传播
            logits, loss = self(idx)
            # 只取最后一个位置的预测
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# ---------------------------------------------------------

# 训练
model = BigramLanguageModel()
m = model.to(device)

# 优化器
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# 训练过程
print("训练设备: ", device)
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 使用模型生成文本
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))