import torch
from model import GPTConfig, GPT

# 小参数
block_size = 128
batch_size = 32
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_layer = 4
n_head = 8
n_embd = 256
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 字符级数据加载
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
  return [stoi[c] for c in s]

def decode(ids):
  return ''.join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  d = train_data if split == 'train' else val_data
  ix = torch.randint(len(d) - block_size, (batch_size,))
  x = torch.stack([d[i:i + block_size] for i in ix])
  y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
  return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
      losses = torch.zeros(eval_iters)
      for k in range(eval_iters):
          X, Y = get_batch(split)
          _, loss = model(X, Y)
          losses[k] = loss.item()
      out[split] = losses.mean()
  model.train()
  return out

# 创建模型
model_args = dict(
  n_layer=n_layer, n_head=n_head, n_embd=n_embd,
  block_size=block_size, bias=False, vocab_size=vocab_size,
  dropout=dropout,
)
model = GPT(GPTConfig(**model_args)).to(device)
print(f"vocab_size: {vocab_size}")
print(f"参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 训练
for iter in range(max_iters):
  if iter % eval_interval == 0:
      losses = estimate_loss()
      print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  xb, yb = get_batch('train')
  _, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

# 生成
context = torch.tensor([encode("FIRST CITIZEN:")], device=device)
output = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
print(decode(output[0].tolist()))