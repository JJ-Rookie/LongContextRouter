# LongContextRouter

面向长上下文任务的成本感知模型路由。仓库提供 LongContextRouter 的训练与推理代码，以及 SWE-Router 的本地方法实现。

LongContextRouter 将输入文本分块，通过冻结的 Qwen3-Embedding-0.6B 编码与因果 Transformer 聚合，预测各候选模型的成功概率。训练采用正确性损失和 Listwise 选择损失；推理结合任务分类器和训练数据的成本统计，在成功概率与执行费用之间进行权衡。

## 数据

数据下载链接：

<!-- Intentionally empty. The owner will provide the data/artifact download URL. -->

数据、模型权重和 embedding 缓存通过外部链接提供。数据格式与文件布局见 [LongContextRouter 数据说明](docs/DATA.md) 和 [SWE-Router 使用说明](baselines/SWERouter5/README.md)。

## 安装

LongContextRouter 的缓存训练和离线推理使用 Python 3.11：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m baselines.LongContextRouter.release --help
```

SWE-Router 的 QLoRA 训练使用独立环境，安装方法见其[使用说明](baselines/SWERouter5/README.md)。

## LongContextRouter

命令均在仓库根目录运行，默认读取 `configs/` 中的配置。训练使用预先划分的训练集和验证集，验证集不得替换为官方测试集。

若下载的 embedding 缓存绑定了原机器路径，先转换缓存元数据。转换保留全部 embedding 数值与来源记录；已转换的缓存直接使用。

```bash
python -m baselines.LongContextRouter.portable_cache \
  --source data/trainval_embeddings_original.pt \
  --output data/trainval_embeddings.pt
```

对任务头 query 缓存和测试缓存采用同样的转换方式。

```bash
python -m baselines.LongContextRouter.release train \
  --train-csv data/train.csv --validation-csv data/validation.csv \
  --cache data/trainval_embeddings.pt --output-dir outputs/router --device cuda
```

使用已有权重可跳过训练。CPU 可通过 `--device cpu` 运行，但跨设备训练不保证得到相同参数。重新训练的 checkpoint 应保留原文件名和同目录生成的 `release_manifest.json`。

任务头在冻结路由器的 query 特征上独立训练：

```bash
python -m baselines.LongContextRouter.release fit-task-head \
  --checkpoint data/router.pth --cache data/task_query_embeddings.pt \
  --train-rows data/task_train.jsonl --validation-rows data/task_validation.jsonl \
  --output outputs/task_head.pth
```

离线评估与模型选择：

```bash
python -m baselines.LongContextRouter.release evaluate \
  --checkpoint data/router.pth --cache data/test_embeddings.pt \
  --task-head outputs/task_head.pth --cost-profile data/train_cost_profile.json \
  --rows data/test.jsonl --output outputs/evaluation.json

python -m baselines.LongContextRouter.release route \
  --checkpoint data/router.pth --cache data/test_embeddings.pt \
  --task-head outputs/task_head.pth --cost-profile data/train_cost_profile.json \
  --inputs data/inputs.jsonl --output outputs/decisions.json
```

`route` 只根据输入及模型预测选择候选模型；`evaluate` 在选择后才使用真实标签和执行费用进行统计。缓存必须包含输入的精确文本；任务头必须与生成它的 router checkpoint 配套。上述命令不调用候选模型 API。

## SWE-Router

实现位于 [`baselines/SWERouter5`](baselines/SWERouter5/README.md)，提供轨迹与费用审计、五路 value head 训练、验证集预算下的策略选择和离线测试。

SWE-bench Pro 在探索模型执行前三个动作后作出路由决策：继续探索模型，或从原任务重新启动其他候选模型。四任务扩展中，其他任务在探索模型完成回答后进行级联选择。这是本地方法实现；四任务扩展并不把所有任务都视为同一种多步 agent 决策流程。具体输入、费用计算及运行命令见子目录说明。

## 项目结构

| 路径 | 内容 |
|---|---|
| `baselines/LongContextRouter/` | 长文本表示、训练损失、任务头、成本策略及运行入口 |
| `baselines/SWERouter5/` | SWE-Router 轨迹适配、费用审计、value head 与路由策略 |
| `configs/` | 训练、推理及资产校验配置 |
| `docs/` | 方法、数据格式与评测口径 |

LongContextRouter 的成本控制是成功概率中的软惩罚，不保证总费用低于硬预算。SWE-Router 在验证集预算内选择策略，测试费用不受同一硬上限保证。详细公式及实验口径见 [方法说明](docs/METHOD.md)。
