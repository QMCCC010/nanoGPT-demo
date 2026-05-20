import time, torch
from model import GPTConfig, GPT

# 和之前训练一样的配置
config = GPTConfig(
    n_layer=4, n_head=8, n_embd=256, block_size=256,
    bias=False, vocab_size=4096, dropout=0.1,
)

steps = 1000
warmup = 50

model = GPT(config)
model.train()
model = model.to('cuda')

if torch.cuda.is_available():
    model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# 随机数据模拟训练
B, T = 32, 256
x = torch.randint(0, 4096, (B, T), device='cuda')
y = torch.randint(0, 4096, (B, T), device='cuda')

torch.cuda.synchronize()
t0 = time.time()

for i in range(steps + warmup):
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if i == warmup:
        torch.cuda.synchronize()
        t0 = time.time()

torch.cuda.synchronize()
elapsed = time.time() - t0

print(f"  {steps}步:  {elapsed:.1f}s  =  {elapsed/steps*1000:.1f}ms/步")