# External Data and Artifacts

Data download link:

<!-- The project maintainer will add the download URL here. -->

Data, checkpoints, and embedding caches are distributed separately from this repository. Use the complete data bundle, including fixed splits, labels, costs, and provenance records. Paths are supplied through the command-line arguments shown in the [main README](../README.md). SWE-Router uses the separate [input and billing specification](SWE_ROUTER.md).

## Required files

| Suggested filename | Contents |
|---|---|
| `train.csv`, `validation.csv` | Training and validation views for 867 and 233 original questions, including model-specific supervision and costs |
| `trainval_embeddings.pt` | Frozen chunk embeddings for the exact training and validation texts |
| `task_train.jsonl`, `task_validation.jsonl` | Canonical queries and task labels belonging to the corresponding question groups |
| `task_query_embeddings.pt` | Frozen chunk embeddings for the task-head queries |
| `test.jsonl` | Evaluation texts, five-model correctness labels, and recorded execution costs for 382 test questions |
| `test_embeddings.pt` | Frozen chunk embeddings for the evaluation texts |
| `inputs.jsonl` | Routing inputs whose exact text is present in the selected cache |
| `router.pth` | Router checkpoint; CSV-trained checkpoints also require their `release_manifest.json` |
| `train_cost_profile.json` | Training-derived, per-task mean whole-problem costs with provenance |

Model IDs and vector entries must follow this order:

```text
deepseek-v4-pro
deepseek-v4-flash-0731
deepseek-v4-pro-0813
qwen3.5-122b-a10b
glm-4-flash-250414
```

## Training CSV

Use UTF-8 CSV with correctly quoted multiline text. Each row supplies supervision for one input view and one candidate model. Preserve these columns:

```text
prompt_id,model_id,model_name,category,prompt,label,cost,model_specific,routing_group_id,is_canonical,loss_target,loss_weight,loss_group_id,tail_offset
```

`model_id` ranges from 0 to 4. `label` stores binary correctness; MemoryAgentBench uses F1 >= 0.5. The CSV loader applies a label threshold of 0.5. `cost` is the finite, nonnegative charge for the full problem run.

The model-specific, canonical, and grouping fields define supervision and routing masks. Preserve them together with the exact text and row order. `loss_target` must be finite and in `[0, 1]`; `loss_weight` must be finite and positive when provided. Leave `tail_offset` empty when unused; otherwise it must be a positive integer. Training and validation must not share an original question group.

## JSONL inputs

Each file must be nonempty and contain unique, nonempty string `prompt_id` values and nonempty `prompt` text.

Task-head training and validation rows:

```json
{"prompt_id":"unique-id","prompt":"exact canonical query text","task":"bbh"}
```

Allowed task names are `bbh`, `memory_agent_bench`, `mmau`, and `swe_bench_pro`. Queries must belong to their respective training or validation question groups. SWE queries and evaluation inputs use the problem statement; the model-specific step text in router-training views is not a substitute.

Evaluation rows, with illustrative labels and costs:

```json
{"prompt_id":"unique-id","prompt":"exact text","category":"bbh/task","labels":[1,0,1,0,0],"costs":[0.1,0.02,0.12,0.03,0.01]}
```

The part of `category` before the first `/` must be an allowed task name. `labels` must contain five binary integers; `costs` must contain five finite, nonnegative numbers in model order. Labels and realized costs are read for reporting after model selection.

Routing rows accept only these two fields:

```json
{"prompt_id":"unique-id","prompt":"exact text"}
```

## Embedding caches

The cache uses `ChunkEmbeddingCache` format 2. SHA256 hashes of exact input texts map to finite float32 tensors of shape `[chunks, 1024]`. The configured encoder is `Qwen/Qwen3-Embedding-0.6B`, with a 32768-token window, 512-token overlap, and the `vllm` backend. Cache fingerprints cover the configuration, text hashes, and tensor contents.

For a cache bound to a machine-local encoder path, run `portable_cache` as shown in the [main README](../README.md#prepare-embedding-caches). It validates the source, rebinds the model identifier, and retains `source_model_name`, `source_model_source_fingerprint`, `source_cache_fingerprint`, and `source_file_sha256`. Embedding values remain unchanged; the new metadata produces a new cache fingerprint. Do not edit caches or remove validation fields manually.

Conversion cannot make different text views or encoder outputs compatible. The configuration records the accepted source fingerprints and file hashes for the training, task-query, and test caches. Extra cached texts do not become evaluation examples: the evaluation JSONL determines which questions are scored. A task head must also match the router checkpoint used to produce its training features.

## Cost profile and integrity checks

The cost profile uses `schema_version: 1`, `mode: "train_task_mean"`, and `cost_unit: "whole_problem"`. It contains the ordered `models`, a `by_benchmark` entry for each of the four tasks, and `provenance` with `fit_split: "train"`. Each benchmark entry contains five finite, nonnegative `mean_costs` and five equal, positive integer `counts`. Preserve the supplied profile and its provenance; do not fit it from test costs.

The default configuration binds the artifacts to the following identities:

| Artifact identity | Expected digest |
|---|---|
| Parsed training and validation dataset fingerprint | `1ae7f0462324387a25301fb66fbf0f8f807ee0a416437b4d454d01e240922715` |
| Source training-cache fingerprint | `b71586554338b8b5ca172a1bc139d0735394dab9ee2d2460ff3ffee26b438929` |
| Cost-profile file SHA256 | `97a532b6bc157cc7e01b833aeac83fd96c4c598dc531fed91378bd3354f28548` |
| Cost-profile source-dataset fingerprint | `f1a56e5b4895ad0b2ca3daaaa7247ae0b3545bac0f8da422d571938a92a13eb1` |

The training entrypoint checks the parsed dataset fingerprint, which includes text, order, labels, costs, grouping, and supervision semantics. It is not a raw CSV file hash. The data bundle should also include a file-level SHA256 manifest. Keep artifact provenance intact and resolve a mismatch before running training or evaluation.
