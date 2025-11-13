import torch
import os
from pathlib import Path
from omegaconf import OmegaConf

# 设置路径
hf_model_path = (
    "/nas/data/jhkuang/projects/AntiDesign_related/dplm/weights/DPLM2"
)
output_dir = (
    "/nas/data/jhkuang/projects/AntiDesign_related/dplm/weights/DPLM2_lightning"
)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(output_dir, ".hydra"), exist_ok=True)

print("=" * 60)
print("Converting DPLM2 from Hugging Face format to Lightning format")
print("=" * 60)

# 1. 加载HF格式的权重
print(f"\n1. Loading weights from: {hf_model_path}")
state_dict = torch.load(
    os.path.join(hf_model_path, "pytorch_model.bin"), map_location="cpu"
)
print(f"   Found {len(state_dict)} weight tensors")

# 2. 创建Lightning checkpoint格式
print("\n2. Converting to Lightning checkpoint format...")
lightning_checkpoint = {
    "state_dict": {},
    "epoch": 0,
    "global_step": 0,
    "pytorch-lightning_version": "2.0.0",
}

# 添加"model."前缀（Lightning格式要求）
for k, v in state_dict.items():
    new_key = f"model.{k}" if not k.startswith("model.") else k
    lightning_checkpoint["state_dict"][new_key] = v

# 保存checkpoint
checkpoint_path = os.path.join(output_dir, "checkpoints", "last.ckpt")
torch.save(lightning_checkpoint, checkpoint_path)
print(f"   Saved checkpoint to: {checkpoint_path}")
print(
    f"   Checkpoint size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB"
)
