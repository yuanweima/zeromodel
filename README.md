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

### Docker 安装（推荐）

本项目提供两种 Docker 镜像：

**1. 完整版（GPU 训练环境）**

```bash
# 构建镜像
docker build -t zeromodel:latest .

# 启动容器（需要 NVIDIA GPU）
docker run --gpus all -it \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd)/data:/workspace/zeromodel/data \
    zeromodel:latest

# 或使用 docker-compose
docker compose up zeromodel
```

**2. 轻量版（消融实验工具，无需 GPU）**

```bash
# 构建轻量镜像
docker build -f Dockerfile.ablation -t zeromodel-ablation:latest .

# 运行测试
docker run zeromodel-ablation:latest pytest tests/ -v

# 运行参数计数
docker run zeromodel-ablation:latest python verl/utils/param_counter.py

# 运行消融实验（dry-run）
docker run zeromodel-ablation:latest python scripts/run_ablation.py --dry-run

# 或使用 docker-compose
docker compose run ablation-test      # 运行测试
docker compose run param-counter      # 参数计数
docker compose run ablation-run       # 消融实验 dry-run
```

**Docker Compose 服务一览**

| 服务 | 用途 | GPU |
|------|------|-----|
| `zeromodel` | 完整训练环境 | ✅ |
| `ablation` | 消融工具交互环境 | ❌ |
| `ablation-test` | 运行测试套件 | ❌ |
| `ablation-run` | 消融实验 dry-run | ❌ |
| `param-counter` | 参数计数工具 | ❌ |

### 本地安装

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
│   │       └── lru_trainer.yaml        # 训练配置 (含17个消融实验)
│   └── utils/
│       ├── param_counter.py            # 参数计数工具
│       ├── statistical_tests.py        # 统计显著性检验
│       ├── evaluation_metrics.py       # 评估指标体系
│       └── reward_score/
│           └── causal_loop.py          # 因果环路奖励函数
├── examples/
│   └── data_preprocess/
│       └── causal_loop.py              # 因果环路数据生成
├── scripts/
│   ├── train_lru.sh                    # 训练脚本
│   └── run_ablation.py                 # 自动化消融实验脚本
└── tests/
    ├── test_mla_lru.py                 # MLA+LRU 测试
    ├── test_param_counter.py           # 参数计数测试
    ├── test_statistical_tests.py       # 统计检验测试
    └── test_evaluation_metrics.py      # 评估指标测试
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

## 消融实验工具

本项目提供完整的消融实验工具链，包括参数计数、统计检验、评估指标和自动化实验脚本。

### 项目结构（工具部分）

```
verl/utils/
├── param_counter.py        # 参数计数工具
├── statistical_tests.py    # 统计显著性检验
└── evaluation_metrics.py   # 评估指标体系

scripts/
└── run_ablation.py         # 自动化消融实验脚本

tests/
├── test_param_counter.py       # 参数计数测试 (16 tests)
├── test_statistical_tests.py   # 统计检验测试 (32 tests)
└── test_evaluation_metrics.py  # 评估指标测试 (18 tests)
```

### 1. 参数计数工具

用于计算不同模型配置的参数量，确保消融实验的公平对比。

```python
from verl.utils.param_counter import (
    count_standard_transformer_params,
    count_mla_params,
    count_mla_lru_params,
    design_matched_baseline,
    compare_configurations,
)

# 计算标准 Transformer 参数量
baseline = count_standard_transformer_params(
    vocab_size=151936,
    hidden_size=896,
    intermediate_size=4864,
    num_layers=24,
    num_attention_heads=14,
    num_kv_heads=2,
)
print(f"Baseline: {baseline.total:,} params")

# 计算 MLA + LRU 参数量
mla_lru = count_mla_lru_params(
    vocab_size=151936,
    hidden_size=896,
    intermediate_size=4864,
    num_layers=24,
    num_attention_heads=14,
    num_kv_heads=2,
    kv_latent_dim=256,
    q_latent_dim=512,
    rope_head_dim=64,
)
print(f"MLA+LRU: {mla_lru.total:,} params")

# 设计参数匹配的 baseline
target_params = mla_lru.total
intermediate_size, matched = design_matched_baseline(
    target_params=target_params,
    vocab_size=151936,
    hidden_size=896,
    num_layers=24,
    num_attention_heads=14,
    num_kv_heads=2,
)
print(f"Matched baseline (intermediate_size={intermediate_size}): {matched.total:,} params")

# 生成对比表格
configs = {
    "baseline": baseline,
    "mla_lru": mla_lru,
    "matched": matched,
}
print(compare_configurations(configs, reference="mla_lru"))
```

命令行快速使用：

```bash
python verl/utils/param_counter.py
```

### 2. 统计显著性检验

提供严格的统计检验工具，支持多种检验方法和多重比较校正。

```python
from verl.utils.statistical_tests import (
    paired_comparison,
    bootstrap_ci,
    multiple_comparison_correction,
    StatisticalReport,
    power_analysis,
)
import numpy as np

# 假设有两组实验结果
baseline_scores = np.array([0.72, 0.75, 0.71, 0.73, 0.74])  # 5 seeds
mla_lru_scores = np.array([0.81, 0.83, 0.79, 0.82, 0.80])

# 配对统计检验（自动选择合适的检验方法）
result = paired_comparison(baseline_scores, mla_lru_scores, test_type="auto")
print(f"P-value: {result.p_value:.6f}")
print(f"Effect size (Cohen's d): {result.effect_size:.3f}")
print(f"Significant: {result.significant}")

# Bootstrap 置信区间
ci = bootstrap_ci(mla_lru_scores, confidence_level=0.95)
print(f"95% CI: [{ci.lower:.4f}, {ci.upper:.4f}]")

# 多重比较校正（多个实验对比时使用）
p_values = [0.01, 0.03, 0.04, 0.08]
adjusted_p, significant = multiple_comparison_correction(
    p_values, method="holm", alpha=0.05
)
print(f"Adjusted p-values: {adjusted_p}")
print(f"Significant: {significant}")

# 生成完整统计报告
report = StatisticalReport(baseline_scores, baseline_name="baseline")
report.add_comparison("mla_lru", mla_lru_scores)
print(report.generate())

# 生成 LaTeX 表格
print(report.to_latex())

# Power analysis：计算所需样本量
power = power_analysis(effect_size=0.5, alpha=0.05, power=0.8)
print(f"Required sample size: {power['required_n']}")
```

支持的统计检验方法：
- **Paired t-test**: 正态分布数据
- **Wilcoxon signed-rank test**: 非参数检验
- **Permutation test**: 小样本或非正态数据
- **Bootstrap CI**: BCa、percentile、basic 方法

支持的多重比较校正：
- **Bonferroni**: 最保守
- **Holm**: 比 Bonferroni 更有 power
- **Benjamini-Hochberg**: FDR 控制

### 3. 评估指标体系

提供全面的评估指标收集和分析工具。

```python
from verl.utils.evaluation_metrics import (
    MetricsCollector,
    EvaluationReport,
    compute_difficulty_curve,
    compare_experiments,
)

# 创建指标收集器
collector = MetricsCollector(max_iterations=8)

# 在评估循环中添加样本
for batch in eval_dataloader:
    outputs = model(batch)

    collector.add_batch(
        predictions=outputs.predictions,
        targets=batch.targets,
        correct=outputs.correct,
        metadata=[{
            'level': m['level'],
            'num_vars': m['num_vars'],
            'num_steps': m['num_steps'],
        } for m in batch.metadata],
        lru_outputs=[{
            'avg_iterations': o.avg_iterations,
            'converged': o.converged,
        } for o in outputs.lru_outputs],
    )

# 计算所有指标
accuracy = collector.compute_accuracy_metrics()
lru = collector.compute_lru_metrics()
efficiency = collector.compute_efficiency_metrics()

# 生成评估报告
report = EvaluationReport(
    experiment_name="mla_lru_adaptive",
    accuracy=accuracy,
    lru=lru,
    efficiency=efficiency,
)
print(report)

# 保存为 JSON
report.to_json("results/eval_report.json")

# 分析难度曲线
curve = compute_difficulty_curve(accuracy.accuracy_by_level)
print(f"Accuracy slope: {curve['slope']:.4f} (per level)")
print(f"Difficulty resilience: {curve['difficulty_resilience']:.4f}")
```

指标类别：

| 类别 | 指标 | 说明 |
|------|------|------|
| **准确率** | overall_accuracy | 总体准确率 |
| | accuracy_by_level | 按难度级别分解 |
| | accuracy_by_num_vars | 按变量数分解 |
| | accuracy_by_num_steps | 按步数分解 |
| **LRU 行为** | avg_iterations | 平均迭代次数 |
| | convergence_ratio | 提前收敛比例 |
| | early_halt_ratio | 早停比例 (≤2 iter) |
| | ponder_cost | 归一化计算代价 |
| **效率** | flops_reduction | 相对固定迭代的 FLOPs 减少 |
| | speedup_vs_fixed | 加速比 |
| | kv_compression_ratio | KV 缓存压缩率 |

### 4. 自动化消融实验

一键运行完整消融实验并生成报告。

```bash
# 查看所有可用实验配置
python scripts/run_ablation.py --dry-run

# 运行所有实验（每个配置 3 seeds）
python scripts/run_ablation.py \
    --config verl/trainer/config/lru_trainer.yaml \
    --output-dir outputs/ablation \
    --seeds 42 123 456

# 只运行特定实验
python scripts/run_ablation.py \
    --experiments baseline mla_only mla_lru_adaptive \
    --seeds 42 123 456

# 从已有结果生成报告
python scripts/run_ablation.py \
    --report-only \
    --results-dir outputs/ablation

# 生成 LaTeX 表格
python scripts/run_ablation.py \
    --report-only \
    --results-dir outputs/ablation \
    --latex
```

### 5. 预定义实验配置

`verl/trainer/config/lru_trainer.yaml` 包含 17 个预定义实验：

| 类别 | 实验名 | 说明 |
|------|--------|------|
| **Baselines** | `baseline` | 原始 Qwen-0.5B (630M) |
| | `baseline_matched` | 参数匹配 baseline (646M) |
| **架构** | `mla_only` | 仅 MLA，无 LRU |
| | `mla_lru_fixed` | MLA + LRU 固定迭代 |
| | `mla_lru_adaptive` | MLA + LRU 自适应停机 (**主实验**) |
| **LRU 组件** | `mla_lru_no_pos_mix` | 无位置混合 |
| | `mla_lru_no_global_halt` | 无全局停机 |
| | `mla_lru_minimal` | 最小 LRU |
| **迭代次数** | `mla_lru_iter_2/4/8/16` | 固定 2/4/8/16 次迭代 |
| **损失权重** | `mla_lru_no_stability` | 无稳定性损失 |
| | `mla_lru_high_stability` | 高稳定性权重 (0.5) |
| | `mla_lru_no_ponder` | 无 ponder 损失 |
| | `mla_lru_high_ponder` | 高 ponder 权重 (0.01) |
| | `mla_lru_ce_only` | 仅交叉熵损失 |

### 6. 运行测试

```bash
# 运行所有工具测试
python -m pytest tests/test_param_counter.py tests/test_statistical_tests.py tests/test_evaluation_metrics.py -v

# 快速验证
python -m pytest tests/test_param_counter.py tests/test_statistical_tests.py tests/test_evaluation_metrics.py --tb=no -q
```

### 7. 论文写作建议

基于工具的输出，论文中应包含：

1. **参数公平性声明**: 使用 `param_counter.py` 的输出说明 baseline 与实验组参数量匹配
2. **统计显著性表格**: 使用 `StatisticalReport.to_latex()` 生成
3. **多重比较校正**: 明确使用的校正方法（推荐 Holm）
4. **难度曲线图**: 展示 accuracy vs difficulty level
5. **效率分析**: 报告 speedup、convergence ratio、ponder cost

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
