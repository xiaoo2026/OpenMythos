# OpenMythos 源码研究笔记

Forked from: https://github.com/kyegomez/OpenMythos
User fork: https://github.com/xiaoo2026/OpenMythos
Last updated: 2026-04-25

## 项目概述

**OpenMythos** 是一个开源的理论复现项目，试图从公开论文和逆向分析角度重建 Claude Mythos（Anthropic 的推理模型）的架构。10.2k stars，2.2k forks。

## 核心架构：三阶段循环深度变换器

```
Input
  ↓
[Prelude P]        — 标准 transformer 层，运行一次
  ↓
[Recurrent Block R] — 循环 T 次（每次循环 = 一步隐式推理）
  ↑______↓
  ↓
[Coda C]           — 标准 transformer 层，运行一次
  ↓
Output
```

### 关键创新点

**1. 循环推理（Recurrent Depth）**
- 不是堆叠几百层，而是同一组权重循环使用
- 推理时循环次数越多，推理深度越深
- 每次循环 = 在连续潜空间里的一步思维链（不输出 token）
- `max_loop_iters` 从 16B 模型的 16 次到 1T 模型的 64 次

**2. MoE FFN（专家混合）**
- FFN 被分成 N 个细粒度专家（fine-grained experts）
- Router 选择 top-K 个专家处理每个 token
- 少量"共享专家"始终激活，吸收跨领域通用知识
- 总参数量大，但每次推理只激活一小部分（~5% 估计）

**3. 注意力机制（两种可选）**
- `MLA`（Multi-Latent Attention）：DeepSeek-V2 技术，缓存压缩的 KV 潜向量而非完整 K/V，大幅降低 KV-cache 内存
- `GQA`（Grouped Query Attention）：标准技术，Flash Attention 2 加速

**4. LTI 注入稳定性**
- 循环训练容易不稳定（hidden state 爆炸或消失）
- 通过将注入参数参数化为连续负对角矩阵，保证谱半径 < 1，训练稳定

**5. LoRA 深度适应**
- 每次循环有小幅 LoRA adapter 调整行为，但不增加太多参数
- 平衡权重共享（高效）与各层差异化（表达力）

**6. ACT  halting（自适应计算时间）**
- 动态决定何时停止循环
- 简单问题少循环，难问题多循环
- 使得模型图灵完备

## 核心文件结构

```
open_mythos/
  main.py      — 核心模型实现（1085行）
  moda.py      — MoDA（Mixture of Depth-wise Attention）实现
  variants.py  — 1B/3B/10B/50B/100B/500B/1T 各规模配置
  tokenizer.py — tokenizer
training/
  3b_fine_web_edu.py — 3B 模型训练脚本（FineWeb-Edu 数据集）
tests/
  bench_vs_transformer.py — 与标准 transformer 对比基准
```

## 各规模配置

| Variant | dim | n_experts | n_shared | loop_iters | context | 备注 |
|---------|-----|-----------|---------|------------|---------|------|
| 1B | 2048 | 64 | 2 | 16 | 4K | 小型研究/微调 |
| 3B | 3072 | 64 | 2 | 16 | 4K | 紧凑推理模型 |
| 10B | 4096 | 128 | 2 | 24 | 8K | 中等通用模型 |
| 50B | 6144 | 256 | 4 | 32 | 8K | 大型推理模型 |
| 100B | 8192 | 256 | 4 | 32 | 1M | 前沿级，128K 输出 |
| 500B | 12288 | 512 | 8 | 48 | 1M | 超大规模 MoE |
| 1T | 16384 | 512 | 8 | 64 | 1M | 最大规模 |

## 关键超参解读

```python
# MLA（Multi-Latent Attention）压缩
kv_lora_rank: 压缩后的 KV 缓存维度（越大压缩越少越精准）
q_lora_rank: Q 的压缩维度
qk_rope_head_dim: 应用 RoPE 的维度（位置编码）
qk_nope_head_dim: 不应用 RoPE 的维度

# MoE
n_experts: 总专家数（越多路由越细粒度）
n_shared_experts: 始终激活的共享专家数
n_experts_per_tok: 每个 token 激活的 top-K 专家数
expert_dim: 每个专家的隐层维度

# 循环
max_loop_iters: 推理时最大循环深度
act_threshold: ACT 停止阈值（累积概率）
lora_rank: 每次循环的 LoRA adapter 秩
```

## 技术亮点

1. **推理时计算扩展**：更多的循环次数 = 更强的推理能力，遵循可预测的指数衰减规律
2. **参数量与推理质量解耦**：同样权重循环使用，内存占用不随推理深度增长
3. **隐式思维链**：循环在连续潜空间进行，不像 CoT 那样输出中间 token，可同时探索多个推理方向
4. **系统泛化能力**：训练时见过的推理链长度可以直接外推到更长的测试链

## 对 OPC 的潜在用途

1. **本地推理引擎**：如果能在 J4105 或 wecleanlinux 上跑 1B-3B 版本，可以做私有化 AI 推理
2. **香薰行业知识库**：结合行业数据微调一个小型专家模型
3. **学习参考**：作为 AI Agent 系统设计的参考架构

## 待深入点

1. moda.py 的 MoDA（Depth-wise Attention）实现细节
2. 训练脚本 3b_fine_web_edu.py 的完整流程
3. ACT halting 机制的具体实现
4. 循环索引嵌入（loop index embedding）的实现方式

## 监测计划

- 每周一凌晨 4 点检查上游仓库更新
- 跟踪新 commits、新 issues、新 PR
- 评估是否有重要新功能或架构变更
