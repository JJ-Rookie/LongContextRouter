# SWE-Router：探索后模型路由

根据 SWE-Router 的探索后路由思路实现的五模型方法扩展。本目录是本地复现与扩展代码，并非官方实现。候选模型、数据划分和计费记录固定在外部数据包中，训练与评估通过离线轨迹重放完成。

## 方法

SWE-bench Pro 中，探索模型先执行三个智能体动作，分类器读取原任务和已经观察到的推理、动作和环境反馈，预测五个候选模型各自的最终成功概率。策略可以继续探索模型，也可以从原任务重新启动另一个候选模型。分类器以 Qwen2.5-Coder-7B-Instruct 为底座，采用 NF4 格式的四位量化低秩微调（QLoRA）和五路分类头，通过二值交叉熵训练，以验证损失选择模型检查点。

同时提供四个评测基准的混合扩展：SWE-bench Pro 仍在三个动作后路由；BBH、MemoryAgentBench、MMAU 使用探索模型完整运行后记录的回答作为探针，再决定保留结果或从原题重启其他模型。后三项属于回答后的级联路由，不能解释成同样的三步智能体机制。

成本规则与路由时机配套：继续探索模型计其整次运行费用；换模型计探索前缀费用加上新模型完整运行费用。非 SWE 探针已经完成，所以其探针费用为完整探索运行费用。成本先验只用训练集估计，成本惩罚权重只用验证集选择，之后固定权重评估测试集。验证费用上限由固定探索模型在验证集上的费用乘以 `--budget-multiplier` 给定；这不是对未知任务的绝对费用保证。

## 数据

数据下载链接：

<!-- 数据下载链接暂留空，后续由项目维护者补充。 -->

数据包需要预处理后的 CSV 数据表、来源清单、轨迹、日志和固定费用记录。格式与目录约定见 [数据与计费说明](../../docs/SWE_ROUTER.md)。数据与权重不在 Git 仓库中。

## 环境

在仓库根目录执行：

```bash
python -m venv .venv-swe
source .venv-swe/bin/activate
python -m pip install -r baselines/SWERouter5/requirements.txt
python -m baselines.SWERouter5.run --help
```

拟合分类器需要 CUDA GPU。`audit` 检查数据和精确费用，并使用分词器检查输入长度；`--base-model` 可以指定已下载的本地模型目录。默认模型标识可由 Transformers 下载所需分词器和模型，完整训练仅在主动运行 `fit` 时启动。

## SWE-bench Pro

```bash
python -m baselines.SWERouter5.run audit \
  --prepared-dir data/swe_router/prepared --swe-root data/swe_agent \
  --k 3 --output-dir outputs/swe_audit

python -m baselines.SWERouter5.run fit \
  --prepared-dir data/swe_router/prepared --swe-root data/swe_agent \
  --k 3 --output-dir outputs/swe_router \
  --device cuda:0 --budget-multiplier 1.0
```

该入口使用 `deepseek-v4-flash-0731` 探索。训练/验证/测试分别为160/40/100题。

## 四个评测基准混合扩展

```bash
python -m baselines.SWERouter5.four_bench_run audit \
  --prepared-dir data/swe_router/prepared --swe-root data/swe_agent \
  --k 3 --output-dir outputs/swe_four_bench_audit

python -m baselines.SWERouter5.four_bench_run fit \
  --prepared-dir data/swe_router/prepared --swe-root data/swe_agent \
  --k 3 --output-dir outputs/swe_four_bench \
  --device cuda:0 --budget-multiplier 1.0 \
  --epochs 5 --seed 43 --max-length 8192
```

默认仍由 Flash 探索；增加 `--explorer-model deepseek-v4-pro-0813` 可以使用 Pro-0813 的对应轨迹和计费记录。训练/验证/测试分别为867/233/382题。

`audit` 返回0表示输入与费用满足拟合条件，2表示不满足。审计不会按词元数量比例猜测缺失的前缀费用。`fit` 输出数据与输入哈希、审计报告、验证集选择的策略、分类器适配器和逐题测试决策；输出目录已存在时拒绝覆盖。失败的训练保留临时目录用于排查。

原始任务、推理和动作保持完整；超长的环境反馈只允许确定性的首尾节选，并明确标记省略部分。无法满足输入长度时中止，避免静默截断决策时可见信息。
