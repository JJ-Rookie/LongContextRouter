# LongContextRouter implementation guide

This guide describes the code components and their inputs. Installation and runnable commands are in the [README](../README.md); external artifact formats are in the [data guide](DATA.md).

## Candidate models and tasks

All model-indexed tensors, labels, costs, and saved model parameters use this order:

1. `deepseek-v4-pro`
2. `deepseek-v4-flash-0731`
3. `deepseek-v4-pro-0813`
4. `qwen3.5-122b-a10b`
5. `glm-4-flash-250414`

The task classifier uses four task labels: `bbh`, `memory_agent_bench`, `mmau`, and `swe_bench_pro`.

## Long-context representation

```text
Input text
  -> 32,768-token chunks with 512-token overlap
  -> frozen Qwen3-Embedding-0.6B embeddings
  -> causal chunk Transformer
  -> 1,024-dimensional text representation
  -> elementwise product with each learned candidate-model vector
  -> shared binary classification head
```

The chunk Transformer uses one layer, 16 query heads, eight key/value heads, RoPE, QK normalization, SwiGLU, RMSNorm, and zero dropout. The final valid chunk supplies the text representation, followed by RMSNorm and L2 normalization. Gaussian representation noise is enabled during training and disabled during inference.

For candidate model `m`, the head produces a success margin `s_m = logit_correct - logit_incorrect`. Its success probability is `sigmoid(s_m)`.

## Router training

The router training configuration uses seed **43**, batch size 8, 20 epochs, AdamW, learning rate `1e-4`, weight decay `1e-5`, and gradient clipping at 1.

The configured losses are supervised binary cross-entropy, listwise selection loss with weight `0.3`, and cost-ranking loss with weight `0.05` and a warmup schedule. Listwise supervision rewards probability mass assigned to correct, supervised candidates. Cost ranking compares candidates that are both correct for the same routing group, favoring the cheaper candidate; it does not reward incorrect candidates for being cheap.

Training examples preserve model-specific supervision masks and routing-group identifiers. See [train.py](../baselines/LongContextRouter/train.py) for CSV loading, supervised loss, and cost ranking, and [op_losses.py](../baselines/LongContextRouter/op_losses.py) for listwise selection loss.

## Task classification and routing

A separate linear task classifier maps frozen router features to four task probabilities. Routing combines candidate success probabilities with a cost penalty derived from these task probabilities, the supplied cost profile, and the configured policy.

Cost weights set the strength of the penalty relative to predicted success. They are not dollar-denominated budgets, and the policy does not enforce a hard spending cap. Cost profiles aggregate execution costs by original routing group so that multiple training views do not duplicate the same task's cost.

The complete decision logic is implemented in [task_router.py](../baselines/LongContextRouter/task_router.py). The `policy` configuration supplies its parameters through [release.py](../baselines/LongContextRouter/release.py). Use these together when inspecting or changing the routing rule.

## Command-line interfaces

| Command | Purpose |
|---|---|
| `train` | Fit the router using prepared training and validation CSV files and cached embeddings |
| `fit-task-head` | Fit the task classifier on frozen router features |
| `route` | Select a candidate from input text and cached representations |
| `evaluate` | Report selected-model correctness and execution cost from labeled evaluation rows |

The selection function does not receive ground-truth outcomes or realized per-instance costs. The evaluation command reads these fields for reporting after selection. Input text, candidate-model order, cache settings, and artifact fingerprints must match the supplied configuration.

The [SWE-Router documentation](SWE_ROUTER.md) covers its separate trajectory-based and completed-response workflows.
