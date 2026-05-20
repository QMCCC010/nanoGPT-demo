"""
交互式对话机器人
"""
import torch
from tokenizers import Tokenizer
from model import GPTConfig, GPT

# 加载分词器
tokenizer = Tokenizer.from_file("tokenizer_movie.json")

# 加载模型
vocab_size = tokenizer.get_vocab_size()
model = GPT(GPTConfig(
    n_layer=4, n_head=8, n_embd=256, block_size=256,
    bias=False, vocab_size=vocab_size, dropout=0.1,
))
model.load_state_dict(torch.load("out/ckpt.pt", map_location="cpu")["model"])
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f"设备: {device}, 参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print("输入 'quit' 退出\n")

while True:
    user = input("你: ").strip()
    if user.lower() == "quit":
        break
    if not user:
        continue

    prompt = f"Q: {user}\nA:"
    ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    input_len = len(ids[0])

    output = model.generate(
        ids, max_new_tokens=50,
        temperature=0.6, top_k=20,
    )

    # 只解码生成的部分（砍掉输入）
    reply_ids = output[0, input_len:].tolist()
    reply = tokenizer.decode(reply_ids)

    # 如果模型生成了后续轮次，截断到第一个 Q 之前
    if "\nQ:" in reply:
        reply = reply.split("\nQ:")[0]
    print(f"机器人: {reply.strip()}\n")
