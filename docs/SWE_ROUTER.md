# SWE-Router Inputs and Billing

## Scope and model order

The implementation replays recorded candidate-model runs offline. It does not call candidate-model APIs or rerun benchmarks. Labels describe the final outcomes of those runs. SWE-bench Pro uses the explorer's first three executable actions; the four-benchmark extension uses complete explorer responses for BBH, MemoryAgentBench, and MMAU.

All model IDs, labels, and cost vectors use this order:

```text
deepseek-v4-pro
deepseek-v4-flash-0731
deepseek-v4-pro-0813
qwen3.5-122b-a10b
glm-4-flash-250414
```

See the [SWE-Router README](../baselines/SWERouter5/README.md) for environment setup and commands. The source data and model assets are supplied separately.

## Prepared data bundle

`--prepared-dir` contains:

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

Use UTF-8 CSV and preserve the column order. `question_order_ours.csv` has columns `prompt_id,prompt`, with 1482 complete questions in order and IDs from 0 to 1481. The three data CSVs use:

```text
prompt_id,model_id,category_id,label,model_name,category,prompt,cost
```

Each question has five rows, one for each model. `category_id` must equal `category`, and the prompt text must match the question table exactly. Labels are binary; MemoryAgentBench uses F1 >= 0.5. Costs must be finite and nonnegative. The prepared training, validation, and test files contain 867, 233, and 382 distinct questions respectively, with no overlap.

The source selection tables use:

```text
prompt_id,prompt_key,bench,instance_id,category,prompt
```

Each `prompt_key` equals `<bench>:<instance_id>`. The training source table contains 1100 rows and the test source table contains 382; their local `prompt_id` values each start at zero. They must retain the supplied order and full text. Do not replace them with truncated or reordered exports.

`manifest.json` must provide these fields:

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

Paths resolve relative to `manifest.json`; explicit absolute paths are also accepted. Replace the SHA256 placeholders with the actual file digests. The loader rejects hash mismatches.

`reference_rows.json` contains `pool[model]["swe_bench_pro"] = [count, successes, total_cost]` for checking the test records. `cost_overrides.json` supplies the frozen billing records and metadata. Its `metadata.prices[explorer]` contains `input_per_million`, `cached_input_per_million`, and `output_per_million`; `splits[source_split][model][prompt_key]` contains the full-run charge. Keep the supplied benchmark metadata and billing records intact.

## Source run artifacts

`--swe-root` points to the source tree in the downloaded bundle:

```text
data/benchmark_splits/ifbench/train.jsonl
sweagent_results/<explorer>/<benchmark-directory>/<train_split|test_split>/<instance_id>/
    <instance_id>.traj
    <instance_id>.trace.log
    <instance_id>.pred
```

Preserve the bundle's directory layout. SWE prefixes come from `.traj` files; exact billing is reconciled against per-call records in `.trace.log`. Non-SWE probes require all three files, including `.pred` for the submitted response. Instance IDs, submissions, and call counts must agree across the artifacts. Evaluator labels and reference answers are excluded from classifier inputs.

The IFBench source file is needed to reconstruct the fixed training and validation membership; IFBench is excluded from the routing datasets. The loader checks source coverage and fixed hashes of the SWE question keys. The resulting SWE-only splits contain 160, 40, and 100 questions; the four-benchmark splits contain 867, 233, and 382. Retain the complete source files even when running only the SWE subset.

## Billing and policy selection

SWE prefix billing first reconciles full-run API-call counts, token usage, cache hits, and the frozen total charge. It then selects the billed calls through the action boundary. Ambiguous call binding or failed reconciliation produces an `unavailable` status with no numeric estimate. Prefix costs are never estimated by a token ratio. Non-SWE probes use the explorer's frozen full-run cost, checked against the cost overlay.

SWE training examples with complete input text may still train the classifier when their prefix cost is unavailable; cost priors use only examples with exact costs. Every validation and test example requires an exact cost. The four-benchmark runner allows at most one unavailable SWE training cost and requires usable cost priors for every benchmark.

The classifier predicts five success probabilities. The policy subtracts a weighted, training-derived remaining-cost estimate from each probability. For SWE, the explorer's estimate is its training mean full-run cost minus its training mean prefix cost; other candidates use their training mean full-run costs. The four-benchmark runner fits these estimates separately for each benchmark. Keeping a completed non-SWE probe has zero remaining cost.

Reported execution costs include the exploration: continuing or keeping the explorer incurs its full-run charge, while switching incurs the prefix or probe charge plus the selected model's full-run charge. Zero recorded charges retain the bundle's billing convention and do not imply zero resource use.

The validation budget is the explorer's validation cost multiplied by `--budget-multiplier`. Among candidate cost weights within that budget, the runner selects the one with the most validation successes, breaking ties by lower cost and then lower weight. It fixes that weight before test replay. This validation constraint does not guarantee the same spending limit on test data. Reports retain model selections and billing provenance.

## Model assets

The default base model is `Qwen/Qwen2.5-Coder-7B-Instruct`, with an upstream revision pinned in code. `--base-model` also accepts a complete local model directory. Include model provenance or file hashes with distributed model assets. Tokenizer files, model weights, LoRA adapters, and datasets are not included in this source repository.
