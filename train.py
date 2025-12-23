import os
import time
import torch

gpu = int(os.environ.get("GPU", "5"))
device = torch.device(f"cuda:{gpu}")

# 先初始化一下
torch.cuda.set_device(device)
torch.randn(1, device=device)

# 目标：占用接近全部显存（留一点点余量避免 OOM）
free, total = torch.cuda.mem_get_info(device)
target = int(free * 0.92)

# 用 uint8 占显存：1 byte / element
buf = torch.empty(target, dtype=torch.uint8, device=device)

print(f"[OK] Holding ~{target/1024**3:.2f} GB on {device}. PID={os.getpid()}")
while True:
    time.sleep(12000)