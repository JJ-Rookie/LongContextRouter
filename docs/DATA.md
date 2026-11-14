# 外部数据包约定

下载链接：

<!-- Owner-provided URL intentionally left empty. -->

公开 Git 仓库不包含下面任何数据或权重。数据包应提供可复现的固定拆分和成本来源，不能只提供 benchmark 的上游链接。所有路径均可通过命令参数指定。

| 建议文件 | 内容 |
|---|---|
| `train.csv`、`validation.csv` | seed-43 的867/233个原题分组，保留模型专属视图、硬标签、监督mask和修正后费用 |
| `trainval_embeddings.pt` | 对应上述训练文本的冻结chunk缓存 |
| `task_train.jsonl`、`task_validation.jsonl` | 同一seed-43分组的canonical query文本与四类任务标签 |
| `task_query_embeddings.pt` | 任务头query文本的冻结chunk缓存 |
| `test.jsonl` | 382题官方测试的文本、模型标签与历史执行费 |
| `test_embeddings.pt` | 测试文本的冻结chunk缓存，SWE仅problem statement |
| `inputs.jsonl` | 需要路由的文本；必须在所选缓存内 |
| `router.pth` | 历史v7 epoch-1 checkpoint或明确标注的新训练checkpoint |
| `train_cost_profile.json` | 历史seed-42训练整题平均成本profile，不重建成seed43 |

## CSV

每行为输入视图与一个模型的监督记录，必须使用 UTF-8 并正确引用长文本。列为：

```text
prompt_id,model_id,model_name,category,prompt,label,cost,model_specific,routing_group_id,is_canonical,loss_target,loss_weight,loss_group_id,tail_offset
```

`model_id` 为0到4，与固定配置完全同序。`label` 是硬正确性标签；`cost` 是非负整题执行费。`model_specific`、`is_canonical` 和分组字段决定监督、路由与损失的mask，不能丢弃。`tail_offset`无值时留空。保持原始视图与记录顺序以复现shuffle行为。训练和验证原题集合不相交。

转换CSV后，`training_io.dataset_fingerprint` 必须仍为 `1ae7f0462324387a25301fb66fbf0f8f807ee0a416437b4d454d01e240922715`。训练入口对此检查，避免把其他拆分当作v7。

## JSONL

任务头每行：

```json
{"prompt_id":"unique-id","prompt":"exact canonical query text","task":"bbh"}
```

`task`只能是配置的四个名称。训练/验证query属于各自的原题分组；SWE使用problem statement，不能使用训练阶段模型产生的最后step。每个集合内ID唯一。

离线评估每行（示意格式，不是实际样本）：

```json
{"prompt_id":"unique-id","prompt":"exact text","category":"bbh/task","labels":[1,0,1,0,0],"costs":[0.1,0.02,0.12,0.03,0.01]}
```

数组长度必须为5；费用有限且非负。标签和实际费用只用于选择后的评估。

纯路由输入每行：

```json
{"prompt_id":"unique-id","prompt":"exact text"}
```

## 缓存与迁移

格式为 `ChunkEmbeddingCache` v2。每个精确文本的SHA256映射到 `[chunks,1024]` float32 tensor，包含配置、文本集合和tensor内容的完整指纹；窗口32768、overlap512、backend vLLM。

历史缓存还包含原机器编码器路径和mtime指纹。`portable_cache`验证原缓存后，仅改为固定模型标识 `Qwen/Qwen3-Embedding-0.6B`，记录 `source_model_name`、`source_model_source_fingerprint`、`source_cache_fingerprint`、`source_file_sha256`，再计算可移植缓存指纹。所有tensor数值不变；因此新旧cache fingerprint不同是预期现象。不要手动编辑缓存或移除校验字段。

原训练cache fingerprint为 `b71586554338b8b5ca172a1bc139d0735394dab9ee2d2460ff3ffee26b438929`。仅迁移路径不能把错误文本视图或不同encoder输出变成兼容缓存。公开数据包应附文件SHA256清单。

配置记录三个原缓存的SHA256与共同的编码器来源指纹。历史测试缓存还含有其他benchmark文本；缓存包含额外文本不等于纳入评估，实际测试清单必须严格为四个benchmark的382题。

## 成本profile

使用 `causal_cost` v1格式：`mode=train_task_mean`、`cost_unit=whole_problem`、固定`models`、四类`by_benchmark`下的`mean_costs[5]`与`counts[5]`、`provenance`。必须提供历史原文件，不可用测试费用拟合。

历史文件SHA256：`97a532b6bc157cc7e01b833aeac83fd96c4c598dc531fed91378bd3354f28548`。

该profile的训练数据指纹：`f1a56e5b4895ad0b2ca3daaaa7247ae0b3545bac0f8da422d571938a92a13eb1`。它与v7训练指纹不同是历史协议的一部分。
