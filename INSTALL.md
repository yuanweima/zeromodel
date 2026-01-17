# 安装指南 | Installation Guide

本文档提供 ZeroModel 的详细安装说明。

## 系统要求

### 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | NVIDIA GPU (8GB+ VRAM) | NVIDIA A100/H100 (40GB+) |
| 内存 | 32GB RAM | 64GB+ RAM |
| 存储 | 50GB 可用空间 | 100GB+ SSD |

### 软件要求

- **操作系统**: Linux (Ubuntu 20.04+) 或 macOS
- **Python**: 3.9 - 3.11
- **CUDA**: 11.8 或 12.1 (用于 GPU 加速)
- **Git**: 2.0+

## 安装步骤

### 1. 创建 Conda 环境

```bash
# 创建新环境
conda create -n zeromodel python=3.9
conda activate zeromodel
```

### 2. 安装 PyTorch

根据你的 CUDA 版本选择对应的安装命令:

**CUDA 12.1 (推荐)**:
```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8**:
```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

**CPU Only (仅用于测试)**:
```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

### 3. 安装 vLLM 和 Ray

```bash
pip install vllm==0.6.3
pip install ray
```

### 4. 安装 veRL (本项目)

```bash
cd /path/to/zeromodel
pip install -e .
```

### 5. 安装 Flash Attention 2

Flash Attention 2 对于高效训练至关重要:

```bash
pip install flash-attn --no-build-isolation
```

如果安装失败，请确保:
- CUDA toolkit 已正确安装
- gcc/g++ 版本 >= 7.0

### 6. 安装其他依赖

```bash
pip install wandb IPython matplotlib
pip install datasets transformers
```

## 验证安装

运行测试套件验证安装:

```bash
# 运行所有测试
python -m pytest tests/test_mla_lru.py -v

# 预期输出:
# tests/test_mla_lru.py::TestConfig::test_mla_config PASSED
# tests/test_mla_lru.py::TestConfig::test_lru_config PASSED
# tests/test_mla_lru.py::TestRoPE::test_decoupled_rope PASSED
# tests/test_mla_lru.py::TestLRU::test_lru_forward PASSED
# tests/test_mla_lru.py::TestMLA::test_mla_attention PASSED
# tests/test_mla_lru.py::TestLosses::test_losses PASSED
# tests/test_mla_lru.py::TestHalting::test_halting_units PASSED
# tests/test_mla_lru.py::TestCausalLoop::test_data_generation PASSED
# tests/test_mla_lru.py::TestCausalLoop::test_reward_function PASSED
```

## 快速验证

```python
# 验证 MLA 模块
import torch
from verl.models.mla import MLAConfig, MLAAttention

config = MLAConfig(hidden_size=256, num_attention_heads=4, kv_latent_dim=64)
mla = MLAAttention(config, layer_idx=0)
x = torch.randn(2, 16, 256)
pos_ids = torch.arange(16).unsqueeze(0).expand(2, -1)
out, _, _ = mla(x, position_ids=pos_ids)
print(f"MLA output shape: {out.shape}")  # [2, 16, 256]

# 验证 LRU 模块
from verl.models.mla import LRUConfig, LatentReasoningUnit

lru_config = LRUConfig(latent_dim=64, max_iterations=4)
lru = LatentReasoningUnit(lru_config)
latent = torch.randn(2, 16, 64)
refined, info = lru(latent)
print(f"LRU output shape: {refined.shape}")  # [2, 16, 64]
print(f"Ponder cost: {info.ponder_cost.item():.4f}")
```

## 数据准备

### 因果环路数据集

```bash
# 生成训练数据
python examples/data_preprocess/causal_loop.py \
    --local_dir ~/data/causal_loop \
    --num_samples 100000 \
    --levels 1,2,3,4

# 验证数据
ls -la ~/data/causal_loop/
# train.parquet
# test.parquet
```

## 常见问题

### Q: Flash Attention 安装失败

**A**: 确保 CUDA toolkit 和 PyTorch CUDA 版本匹配:
```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

### Q: OOM (内存不足)

**A**: 尝试以下方法:
1. 减小 batch size
2. 启用梯度检查点:
   ```yaml
   critic.model.enable_gradient_checkpointing: True
   ```
3. 使用混合精度训练

### Q: vLLM 导入错误

**A**: 确保 vLLM 版本与 PyTorch 兼容:
```bash
pip install vllm==0.6.3 --force-reinstall
```

### Q: Ray 集群问题

**A**: 清理 Ray 进程并重启:
```bash
ray stop --force
ray start --head
```

## 开发模式安装

如果你需要修改代码进行开发:

```bash
# 克隆仓库
git clone https://github.com/lt0440/zeromodel.git
cd zeromodel

# 以可编辑模式安装
pip install -e ".[dev]"

# 安装预提交钩子 (可选)
pip install pre-commit
pre-commit install
```

## Docker 安装 (可选)

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-devel

WORKDIR /app
COPY . .

RUN pip install -e . && \
    pip install vllm==0.6.3 ray flash-attn --no-build-isolation

CMD ["bash"]
```

构建并运行:
```bash
docker build -t zeromodel .
docker run --gpus all -it zeromodel
```

## 下一步

安装完成后，请参阅:
- [README.md](README.md) - 项目概述和快速入门
- [scripts/train_lru.sh](scripts/train_lru.sh) - 训练脚本示例
