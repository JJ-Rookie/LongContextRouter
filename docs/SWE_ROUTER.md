# SWE-Router 输入与计费

## 数据边界

该实现使用五模型已有运行的结果进行离线策略重放，不会调用候选模型 API 或重新执行 benchmark。模型标签是完整历史运行的固定结果。四项混合扩展使用不同的观察时点：SWE-bench Pro 是三次可执行动作后的前缀，其余三项是完整回答后的探针。

五模型顺序必须保持：

```text
deepseek-v4-pro
deepseek-v4-flash-0731
deepseek-v4-pro-0813
qwen3.5-122b-a10b
glm-4-flash-250414
```

## Prepared CSV

`--prepared-dir` 包含：

```text
manifest.json
question_order_ours.csv
train_ours.csv
validation_ours.csv
test_ours.csv
exports/train/selected_prompts.csv
exports/test/selected_prompts.csv
reference_rows.json
cost_overrides.json
```

`question_order_ours.csv` 的列为 `prompt_id,prompt`。三个数据CSV的列与顺序为：

```text
prompt_id,model_id,category_id,label,model_name,category,prompt,cost
```

每题五行，标签为0/1，费用为非负有限数。问题ID与完整问题表一致，训练/验证/测试分别包含867/233/382题，ID无重复。完整源问题表与来源选择表必须一起下载，不能用截断文本或重排后的表替代。

`selected_prompts.csv` 的列为：

```text
prompt_id,prompt_key,bench,instance_id,category,prompt
```

来源清单 `manifest.json` 至少包含：

```json
{
  "model_order": ["deepseek-v4-pro", "deepseek-v4-flash-0731", "deepseek-v4-pro-0813", "qwen3.5-122b-a10b", "glm-4-flash-250414"],
  "prompt_counts": {"train": 867, "validation": 233, "test": 382},
  "prepared_sha256": {
    "question_order_ours.csv": "<SHA256>",
    "train_ours.csv": "<SHA256>",
    "validation_ours.csv": "<SHA256>",
    "test_ours.csv": "<SHA256>"
  },
  "official_train_export": "exports/train",
  "official_test_export": "exports/test",
  "official_selected_sha256": {"train": "<SHA256>", "test": "<SHA256>"},
  "reference_rows": "reference_rows.json",
  "reference_rows_sha256": "<SHA256>",
  "cost_overlay": "cost_overrides.json",
  "cost_overlay_sha256": "<SHA256>"
}
```

路径相对于 `manifest.json` 所在目录解析，与启动命令时的目录无关。也可明确使用本机绝对路径。示例中的SHA占位符必须由实际数据文件摘要替换；解析器拒绝不匹配的文件。

## 原始运行文件

`--swe-root` 指向下载数据包中的源运行树：

```text
data/benchmark_splits/ifbench/train.jsonl
sweagent_results/<explorer>/<benchmark>/<train_split|test_split>/<instance_id>/
    <instance_id>.traj
    <instance_id>.trace.log
    <instance_id>.pred
```

`.pred`用于非SWE的完整响应探针。SWE前缀来自`.traj`，精确费用由`.trace.log`的逐次调用记录核对。MemoryAgentBench的三款DeepSeek结果目录为`memory_agent_bench_128k_v2_tem0`，其他模型使用`memory_agent_bench_128k_v2`，不能替换为同名但不同来源的运行。

IFBench仅用于复原固定分层划分时的随机序列，不参与五模型四benchmark分类器训练或测试。首先在1300条源问题上按类别、seed43与比例0.8切分，再排除IFBench，得到867/233。SWE子集的训练/验证题目集合还需匹配代码中固定的键哈希。

## 费用与选择

SWE前缀计费先核对完整日志的API调用数、token统计、缓存命中和整次运行冻结费用，再按动作边界选择实际调用。无法唯一绑定或对账失败时标记`unavailable`，不按token比例估计。非SWE探针费用直接取冻结表中完整探索运行费用，并验证与成本覆盖文件一致。

训练集中仍有完整文本但前缀成本不可核对的SWE条目可参与分类器训练，费用先验只用费用可核对条目；验证与测试要求全部费用准确。四项扩展最多允许一条SWE训练费用缺失，并要求每个benchmark仍有可用费用先验。

分类器输出五个成功概率。路由根据预测成功概率减去训练集估计的剩余费用惩罚选择候选；继续探索模型的剩余费用扣除已经发生的前缀，重启其他模型则使用该模型整题费用。执行统计仍将已经发生的探索费用计入。

成本权重在预先声明的验证预算内按正确数最大化选择，正确数并列选费用更低者。锁定权重之后才重放固定测试集。非SWE的已完成探针若被保留，其额外剩余费用为零；这与SWE的中途继续运行不同。

某些模型的冻结账面费用可能为零。此处保持数据包的计价约定，不将零账面费用解释为零资源消耗。最终报告同时记录各模型选择次数和费用来源。

## 模型资产

默认底座为`Qwen/Qwen2.5-Coder-7B-Instruct`，代码固定模型仓库revision以保持输入与权重来源一致。`--base-model`也可指向完整的本地目录；数据发布方应附该目录的模型来源或文件摘要。LoRA适配器、tokenizer和数据不随本源码仓库分发。
