# ZeroModel

**潜空间逻辑推理系统 | Latent Space Logical Reasoning System**

ZeroModel 是一个基于 DeepSeek-V3 的 MLA（多头潜在注意力）架构，并注入 LRU（潜空间推理单元）的创新性研究项目。本项目旨在验证"潜空间震荡"式的迭代推理是否比简单增加参数量具有更强的逻辑压缩能力。

## 核心创新

1. **MLA (Multi-head Latent Attention)**: 在注意力机制中引入 KV 压缩潜空间
2. **LRU (Latent Reasoning Unit)**: 在潜空间中进行递归推理，使用 GRU 风格门控
3. **ACT (Adaptive Computation Time)**: 自适应停机机制，根据问题复杂度动态调整计算量
4. **Decoupled RoPE**: 解耦位置编码，只对部分 head 维度应用 RoPE

## 架构图

```
输入 x [B, S, H]
       │
       ▼
   ┌─────────────────┐
   │ W_DKV 下投影     │ → c_kv [B, S, d_c]  (潜空间压缩)
   └─────────────────┘
       │
       ▼
   ┌─────────────────┐
   │ LRU 迭代推理     │ → c_kv' [B, S, d_c] (N次迭代直到停机)
   │ (GRU门控+ACT)   │
   └─────────────────┘
       │
       ├──→ W_UK 上投影 → K [B, S, n_kv * head_dim]
       │
       └──→ W_UV 上投影 → V [B, S, n_kv * head_dim]

   ┌─────────────────┐
   │ W_DQ/W_UQ      │ → Q [B, S, n_heads * head_dim]
   └─────────────────┘
       │
       ▼
   Flash Attention → 输出
```

## 安装

详见 [INSTALL.md](INSTALL.md)

快速安装:

```bash
conda create -n zeromodel python=3.9
conda activate zeromodel

# 安装 PyTorch (CUDA 12.1)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 安装 vLLM 和 Ray
pip install vllm==0.6.3 ray

# 安装 veRL
pip install -e .

# 安装 Flash Attention 2
pip install flash-attn --no-build-isolation

# 安装其他依赖
pip install wandb IPython matplotlib
```

## 项目结构

```
zeromodel/
├── verl/
│   ├── models/
│   │   ├── mla/                        # MLA + LRU 核心实现
│   │   │   ├── config.py               # 配置类 (MLAConfig, LRUConfig)
│   │   │   ├── rope.py                 # 解耦 RoPE 实现
│   │   │   ├── mla_attention.py        # MLA 注意力机制
│   │   │   ├── lru.py                  # 潜空间推理单元
│   │   │   └── modeling_deepseek_mla.py # 完整模型
│   │   └── registry.py                 # 模型注册表
│   ├── trainer/
│   │   ├── lru/                        # LRU 训练组件
│   │   │   ├── losses.py               # 损失函数
│   │   │   └── halting.py              # 停机单元
│   │   └── config/
│   │       └── lru_trainer.yaml        # 训练配置
│   └── utils/
│       └── reward_score/
│           └── causal_loop.py          # 因果环路奖励函数
├── examples/
│   └── data_preprocess/
│       └── causal_loop.py              # 因果环路数据生成
├── scripts/
│   └── train_lru.sh                    # 训练脚本
└── tests/
    └── test_mla_lru.py                 # 测试套件
```

## 因果环路任务

因果环路预测任务是用于验证 LRU 架构逻辑推理能力的核心基准测试。

### 任务难度级别

| 级别 | 描述 | 示例 |
|------|------|------|
| Level 1 | 线性链 | A=1 → B=A+2 → C=B*2 → D=C-1 |
| Level 2 | 简单环 | A=1 → B=A+2 → C=B*2 → A=C%5 (迭代到收敛) |
| Level 3 | 多路径汇聚 | A→C, B→C 同时影响 C |
| Level 4 | 复杂嵌套环 | 多个相互影响的变量循环 |

### 数据生成

```bash
# 生成 100K 样本数据集
python examples/data_preprocess/causal_loop.py \
    --local_dir ~/data/causal_loop \
    --num_samples 100000 \
    --levels 1,2,3,4
```

数据生成速度约 150K 样本/秒。

## 训练

### 单 GPU 训练 (小模型调试)

```bash
export N_GPUS=1
export BASE_MODEL=Qwen/Qwen2.5-0.5B
export DATA_DIR=~/data/causal_loop
export EXPERIMENT_NAME=zeromodel-0.5b-lru

bash scripts/train_lru.sh
```

### 多 GPU 训练

```bash
export N_GPUS=4
export BASE_MODEL=Qwen/Qwen2.5-3B
export DATA_DIR=~/data/causal_loop
export EXPERIMENT_NAME=zeromodel-3b-lru

bash scripts/train_lru.sh
```

### 实验对照组

我们设计了 4 组实验进行对比：

| 组别 | 配置 | 说明 |
|------|------|------|
| Group A | 原始 Qwen-3B | 基线 |
| Group B | + MLA (无 LRU) | 验证 MLA 压缩效果 |
| Group C | + MLA + LRU (固定迭代) | 验证 LRU 推理能力 |
| Group D | + MLA + LRU (自适应停机) | 完整架构 |

## 损失函数

| 损失 | 权重 | 作用 |
|------|------|------|
| L_pred | 1.0 | 下一 token 预测（主目标）|
| L_stability | 0.1 | 表示收敛性（迭代后残差递减）|
| L_sparsity | 0.01 | 激活稀疏性（Hoyer-Square）|
| L_ponder | 0.001 | 计算代价惩罚（ACT 风格）|

## 测试

```bash
# 运行完整测试套件
python -m pytest tests/test_mla_lru.py -v

# 运行特定测试
python -m pytest tests/test_mla_lru.py::TestMLA -v
```

## API 使用

### MLA 注意力

```python
from verl.models.mla import MLAConfig, MLAAttention

config = MLAConfig(
    hidden_size=2048,
    num_attention_heads=16,
    kv_latent_dim=512,
    q_latent_dim=1536,
    rope_head_dim=64,
)

mla = MLAAttention(config, layer_idx=0)
output, attn_weights, past_kv = mla(hidden_states, position_ids=position_ids)
```

### LRU 推理单元

```python
from verl.models.mla import LRUConfig, LatentReasoningUnit

lru_config = LRUConfig(
    latent_dim=512,
    max_iterations=8,
    halt_threshold=0.99,
)

lru = LatentReasoningUnit(lru_config)
refined_latent, lru_output = lru(latent_kv)

# lru_output 包含:
# - halt_probs: 每步停机概率
# - ponder_cost: 计算代价
# - num_iterations: 实际迭代次数
```

### 完整模型

```python
from verl.models.mla import DeepSeekMLAConfig, DeepSeekMLAForCausalLM

config = DeepSeekMLAConfig(
    vocab_size=32000,
    hidden_size=2048,
    num_hidden_layers=24,
    use_lru=True,
    lru_max_iterations=8,
)

model = DeepSeekMLAForCausalLM(config)
outputs = model(input_ids, labels=labels)
```

## 评估指标

- **准确率 vs 难度级别**: 不同任务难度下的表现
- **Few-shot 效率**: 示范数量与性能的关系
- **平均迭代次数 vs 问题复杂度**: 验证自适应计算
- **收敛比**: 最后/首次残差，验证推理收敛性

## 致谢

- [TinyZero](https://github.com/Jiayi-Pan/TinyZero) - 本项目基于 TinyZero 构建
- [veRL](https://github.com/volcengine/verl) - PPO 训练框架
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) - MLA 架构参考
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) - 基座模型

## 引用

```bibtex
@misc{zeromodel2025,
    author = {ZeroModel Authors},
    title = {ZeroModel: Latent Space Logical Reasoning with MLA and LRU},
    howpublished = {https://github.com/lt0440/zeromodel},
    year = {2025}
}
```

## 许可证

本项目采用 Apache License 2.0 许可证。详见 [LICENSE](LICENSE) 文件。
