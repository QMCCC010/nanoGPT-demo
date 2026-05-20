import torch

# 1. PyTorch 版本 >= 2.0？
print(f"PyTorch {torch.__version__}")  # 你的是 2.10 → ✅

# 2. 实际编译一小段测试
try:
    @torch.compile
    def f(x):
        return x + 1
    f(torch.randn(3))
    print("torch.compile: OK")
except Exception as e:
    print(f"torch.compile: 不行 ({e})")