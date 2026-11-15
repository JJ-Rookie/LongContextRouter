# LongContextRouter

Cost-aware model routing for long-context tasks. This repository provides LongContextRouter training and inference code together with a local SWE-Router implementation.

LongContextRouter splits input text into chunks, encodes them with a frozen Qwen3-Embedding-0.6B model, and aggregates them with a causal Transformer. It predicts each candidate model's probability of success using a correctness loss and a listwise selection loss. At inference time, a task classifier and training-set cost statistics balance predicted success against execution cost.

## Data and model artifacts

Data download link:

<!-- Intentionally left empty. The project maintainer will add the download URL. -->

Datasets, model weights, and embedding caches are distributed separately. See the [LongContextRouter data specification](docs/DATA.md) and the [SWE-Router guide](baselines/SWERouter5/README.md) for the required files. Training and evaluation require these external artifacts; installing dependencies alone does not supply them.

## Environment setup

The instructions below target Linux with Python 3.11 and Bash. Download the repository as a ZIP archive from its hosting page, extract it, and open a terminal in the extracted directory containing this README and `requirements.txt`.

Run the remaining commands from that repository root. Use separate virtual environments for the two methods.

| Workflow | Compute requirements | Dependencies |
|---|---|---|
| LongContextRouter training with cached embeddings | CPU or a CUDA-enabled GPU | Root `requirements.txt` |
| LongContextRouter task-head fitting, evaluation, and routing | CPU; these entrypoints load the router on CPU | Root `requirements.txt` |
| SWE-Router data audit | CPU and access to the base-model tokenizer | `baselines/SWERouter5/requirements.txt` |
| SWE-Router QLoRA training | NVIDIA GPU with CUDA and support for the BF16 computation used by the loader | `baselines/SWERouter5/requirements.txt` |

### LongContextRouter: CPU environment

The pinned dependencies are PyTorch `2.13.0` and NumPy `2.2.6`. Install the CPU wheel explicitly to avoid downloading GPU libraries:

```bash
python3.11 -m venv .venv/lcr
source .venv/lcr/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

Check the environment without loading a dataset or model:

```bash
python -c "import sys, torch, numpy; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('NumPy:', numpy.__version__); print('CUDA available:', torch.cuda.is_available())"
python -m baselines.LongContextRouter.release --help
python -m baselines.LongContextRouter.portable_cache --help
```

`CUDA available: False` is expected for the CPU environment. The cached workflow does not load the embedding encoder and does not require Transformers or vLLM. Generating new embedding caches requires a separate encoder environment and matching encoder settings; it is not part of this installation recipe.

### LongContextRouter: optional CUDA training environment

Choose a PyTorch CUDA wheel supported by your NVIDIA driver using the [official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/). For example, on a machine compatible with the CUDA 12.6 wheel:

```bash
python3.11 -m venv .venv/lcr-cuda
source .venv/lcr-cuda/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
nvidia-smi
python -c "import torch; print('PyTorch CUDA runtime:', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is unavailable in this environment'; print('GPU:', torch.cuda.get_device_name(0))"
```

Use `--device cuda` for router training in this environment. The CPU-only wheel does not enable CUDA training. Changing the device or dependency environment does not guarantee identical trained parameters.

### SWE-Router environment

SWE-Router additionally uses Transformers, PEFT, Accelerate, and bitsandbytes. Follow its [environment setup instructions](baselines/SWERouter5/README.md#environment-setup) to install the pinned dependencies, verify GPU access, and configure downloaded model assets. A CPU environment can audit inputs, but cannot run the QLoRA training entrypoint.

## LongContextRouter usage

The commands use the default configuration under `configs/`. Training inputs must already be split into training and validation sets; do not substitute the official test set for validation.

### Prepare embedding caches

If a downloaded cache is bound to paths on the original machine, convert its metadata first. Conversion preserves the embedding values and source provenance. Already converted caches can be used directly.

```bash
python -m baselines.LongContextRouter.portable_cache \
  --source data/trainval_embeddings_original.pt \
  --output data/trainval_embeddings.pt
```

Apply the same conversion to the task-head query cache and the test cache when needed.

### Train the router

```bash
python -m baselines.LongContextRouter.release train \
  --train-csv data/train.csv --validation-csv data/validation.csv \
  --cache data/trainval_embeddings.pt --output-dir outputs/router --device cpu
```

Training outputs are written to `outputs/router/`. Keep the generated weights and `release_manifest.json` together. In the commands below, use the path to your weights for `--checkpoint`.

### Fit the task classifier

The task classifier is trained separately on frozen router features from canonical query inputs:

```bash
python -m baselines.LongContextRouter.release fit-task-head \
  --checkpoint data/router.pth --cache data/task_query_embeddings.pt \
  --train-rows data/task_train.jsonl --validation-rows data/task_validation.jsonl \
  --output outputs/task_head.pth
```

### Evaluate or select a model

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

`route` selects a candidate using the input and model predictions. `evaluate` reads ground-truth labels and realized execution costs only after selection, for reporting. The cache must contain the exact input text, and the task head must match the router checkpoint used to produce its features. These commands do not call candidate-model APIs.

## SWE-Router

The implementation in [`baselines/SWERouter5`](baselines/SWERouter5/README.md) provides trajectory and cost auditing, five-output success prediction, policy selection under a validation budget, and offline evaluation.

For SWE-bench Pro, routing happens after the explorer's first three agent actions: continue the explorer or restart another candidate from the original task. In the four-benchmark extension, the other benchmarks use a complete explorer response before making a cascade decision. This is a local implementation and extension, not the official implementation; the four benchmarks do not share an identical multi-step agent decision point.

## Repository layout

| Path | Contents |
|---|---|
| `baselines/LongContextRouter/` | Long-context representation, losses, task classifier, cost policy, and command-line entrypoints |
| `baselines/SWERouter5/` | Trajectory adapters, cost accounting, success prediction, and routing policies |
| `configs/` | Training, inference, and artifact-validation configuration |
| `docs/` | Implementation guide, data formats, and SWE-Router inputs |

LongContextRouter uses a soft cost penalty and does not guarantee a hard spending cap. SWE-Router selects a policy within a validation-set budget; that cap does not guarantee the same bound on test spending. See the [implementation guide](docs/METHOD.md) for the components and configuration entrypoints.
