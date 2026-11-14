# v7_seed43 方法与结果口径

## 版本

固定候选模型顺序：

1. `deepseek-v4-pro`
2. `deepseek-v4-flash-0731`
3. `deepseek-v4-pro-0813`
4. `qwen3.5-122b-a10b`
5. `glm-4-flash-250414`

训练/验证为 867/233 个原始 routing group；测试共 382 题：BBH 100、MemoryAgentBench 100、MMAU 82、SWE-bench Pro 100。排除 IFBench。五模型顺序是 checkpoint、缓存数据标签和 cost profile 的共同约定。

MemoryAgentBench 使用问题和历史上下文，三款 DeepSeek 结果取 `128k_v2_tem0`，Qwen/GLM 取既有 `128k_v2`；正确性阈值为 F1 ≥ 0.5。SWE 训练使用最终有效 step 与 Problem Statement 构成的模型专属视图；测试只使用 Problem Statement。训练视图保留模型监督 mask 和原题 grouping，不能简化成丢失这些字段的普通表格。

## 表示与训练

```text
输入文本 → 32768-token 分块（重叠512） → 冻结Qwen3-Embedding-0.6B
        → 一层因果chunk Transformer → 1024维文本向量q
        → 与各模型可学习向量p_m逐元素相乘 → 共享二分类头
        → 成功margin s_m = logit_correct - logit_incorrect
```

Transformer 使用 16 query heads、8 KV heads、RoPE、QK-Norm、SwiGLU、RMSNorm 和零 dropout。取最后一个有效 chunk 的输出，经 RMSNorm 与 L2 归一化。训练时文本向量加入标准差 0.01 的高斯噪声，推理关闭噪声。

seed 43，batch 8，20 epochs，AdamW，学习率 `1e-4`，weight decay `1e-5`，gradient clip 1。按内部验证集 sample accuracy 选 checkpoint，准确率并列时保留较早 epoch。

配置损失为：

`L = BCE + 0.3 × L_listwise + 0.05 × L_cost_rank × 1[epoch > 1]`

`L_listwise = -log Σ(correct AND supervised models) softmax(s / T)`，`T=1`。分母含全部候选，跳过没有正确受监督模型的视图。该损失让候选模型在同一输入上竞争。

成本排序只比较同一原题中两个都正确的模型：`softplus(0.1 - (s_cheaper - s_costlier))`，相同费用不构成排序对。便宜但错误的模型不会因此获奖励。

**76.44% 的 checkpoint 是 epoch 1，实际损失只有 `BCE + 0.3 × Listwise`。** 当时 cost-ranking 仍在 warmup，权重为零。epoch 1 平均 BCE 为 0.57169158，Listwise 为 0.71593319，总 loss 为 0.78647154。不能把该 checkpoint 的结果归因于训练期成本排序。decision-aligned loss 和 joint task-aux 均未启用。

另行在冻结特征上训练 `Linear(1024,4)` 任务头，使用 seed-43 的训练/验证划分，头初始化 seed 42，300 epochs，AdamW `lr=0.01`、`weight_decay=1e-4`、类别加权交叉熵，验证选出的头为 epoch 234。

## 成本约束

这是软成本惩罚，不是硬预算约束；没有保证总费用小于某个美元上限。

任务头输出四类后验 `r_t(x)`；路由器输出成功概率 `p_m(x)=sigmoid(s_m)`。历史固定策略选择：

`m* = argmax_m [p_m(x) - Σ_t r_t(x) λ_t φ_t(ĉ_t,m)]`

`ĉ_t,m` 是训练数据按原始题目统计的模型整题平均执行费用。按题目聚合可以避免展开多个训练视图造成重复计费。预测阶段不读取测试实例的真实费用或真实 benchmark 名称。

| 任务 | λ | 成本变换 |
|---|---:|---|
| BBH | 0 | min-max |
| MemoryAgentBench | 0.2 | min-max |
| MMAU | 0.05 | min-max |
| SWE-bench Pro | 0.2 | train_log1p_median |

- min-max：`(c-min(c))/(max(c)-min(c))`；成本全相同时为零。
- train_log1p_median：`log(1+c/M)-log(1+min(c)/M)`，`M` 为正成本的中位数；不截断到 `[0,1]`。
- 当 `max_t r_t(x) < 0.6` 时，改用按训练题目数加权的全局模型均价，以 log1p-median 变换和 `λ=0.3` 计算效用。
- `mode=soft`，`guard=false`。配置中的旧 BBH anchor/threshold 字段因此不生效。
- 软路由效用并列时按成功概率较高、预估费用较低、模型序号较小顺序决定。

λ控制成本相对于成功概率的惩罚强度，不是允许花费的美元数。$11.65 是固定测试集路由完成后统计的历史执行费，未计入 router 训练或 embedding 计算开销，也不是预算配置。

## 结果与可复现边界

| 同一checkpoint的推理策略 | 成功数 | Sample accuracy | 总执行费USD |
|---|---:|---:|---:|
| 成功概率argmax | 282/382 | 73.82% | 17.778145254 |
| 验证集选策略 | 286/382 | 74.87% | 14.424550062 |
| 本仓库历史固定策略 | 292/382 | 76.44% | 11.646225618 |

76.44% 分项为 BBH 97/100、MemoryAgentBench 67/100、MMAU 50/82、SWE-bench Pro 78/100。最终选择 DeepSeek-V4-Pro-0813 共 263 次、DeepSeek-V4-Flash-0731 共119次；62题触发低任务置信度 fallback。

该策略最初通过测试集贪心坐标搜索选出。历史 cost profile 还沿用 seed-42 的训练划分，与 router 的 seed-43 划分不同。为忠实复现，必须保留原 profile；重新拟合 seed-43 profile 会形成不同协议。74.87% 使用验证集选策略，但同样继承历史资产，不能据此宣称整个历史实验从未查看测试集。

原始结果依据是 `op_paper_ablation_5model_20260921/ours_v7_fixed.json`，验证选策略依据是该实验下 `ours_v7/eval.json`。旧 `op_loss_experiments_20260920_5model/v7_seed43/eval.json` 留有不同策略结果，不能作为76.44的依据。2026-11的同设置Listwise=0.3复现得到相同15个模型参数tensor及292/382；不同序列化文件的SHA不要求相同。

历史 checkpoint SHA256：`bceb3719b480ffbb2a2d4d35d74d62fe91659d46f09e9ccf935a8d66c5a67d7f`。配置文件还保存原训练数据指纹与成本 profile 摘要。这里保留的是固定历史协议，不提供测试集搜索入口。

## 安装环境与实验记录

历史训练使用 PyTorch 2.6.0。公开安装依赖使用 PyTorch 2.13.0，以修复包括受限权重反序列化在内的已知漏洞；算法和训练配置保持不变。不同 PyTorch 或硬件环境重新训练不保证逐位相同的参数，历史结果核验应保留原权重、输入、缓存和成本 profile 的来源对应关系。
