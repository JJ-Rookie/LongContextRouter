# SWE-Router: Model Routing After Exploration

A local implementation and five-model extension of SWE-Router's exploration-based routing approach. This is not the official implementation. The external data bundle fixes the candidate models, data splits, and billing records; training and evaluation use offline trajectory replay.

## Method

For SWE-bench Pro, an explorer model first performs three agent actions. A classifier reads the original task and the observed thoughts, actions, and environment responses, then predicts the final success probability of each candidate model. The policy either continues the explorer or restarts another candidate from the original task. The classifier uses Qwen2.5-Coder-7B-Instruct with 4-bit NF4 QLoRA and a five-output classification head. It is trained with binary cross-entropy.

The four-benchmark extension uses a different observation point for the other tasks. SWE-bench Pro still routes after three actions; BBH, MemoryAgentBench, and MMAU use the explorer's completed response as a probe, then keep that response or restart another model. These three tasks use a cascade after the initial answer, not the same three-action agent mechanism.

Costs follow the decision point: continuing the explorer incurs its full run cost; switching incurs the exploration prefix cost plus the selected model's full run cost. For non-SWE tasks, the probe has already finished, so its cost is the explorer's full run cost. Cost priors are fitted only on training data. The cost penalty is selected on validation data and then fixed for test evaluation. The validation budget equals the explorer's validation cost multiplied by `--budget-multiplier`; it does not guarantee an absolute spending cap on unseen tasks.

## Data

Data download link:

<!-- The project maintainer will add the data download URL here. -->

The data bundle must contain prepared CSV files, source manifests, trajectories, logs, and fixed billing records. Data and model weights are distributed separately from this Git repository.

## Environment Setup

Use Python 3.11 and a separate virtual environment. The commands below use a Linux shell and run from the repository root.

| Package | Pinned dependency |
|---|---:|
| PyTorch | 2.13.0 |
| Transformers | 4.57.3 |
| PEFT | 0.18.0 |
| Accelerate | 1.11.0 |
| bitsandbytes | 0.48.2 |

The dependency pins are maintained in [`requirements.txt`](requirements.txt).

```bash
python3.11 -m venv .venv/swe
source .venv/swe/bin/activate
python -m pip install --upgrade pip
```

`audit` checks input coverage, exact costs, and tokenizer input lengths. It can run on CPU and does not load the 7B model weights. `fit` trains and evaluates the quantized value head and requires an NVIDIA GPU with a working CUDA-enabled PyTorch installation. The loader uses bfloat16 computation and NF4 quantization; a CPU-only PyTorch installation cannot run `fit`.

Install a CUDA build compatible with your NVIDIA driver, then install the remaining dependencies. The example below uses the CUDA 12.6 build; use it only when your driver supports it. See the [official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/) for other builds matching the pinned dependency.

```bash
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r baselines/SWERouter5/requirements.txt
python -m pip check
python -m baselines.SWERouter5.run --help
python -m baselines.SWERouter5.four_bench_run --help
```

For a machine used only for auditing, replace the first command above with the CPU wheel installation:

```bash
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
```

Check the environment before starting a run:

```bash
python - <<'PYTHON'
import sys
from importlib.metadata import version

import torch
import transformers
import peft
import accelerate
import bitsandbytes

print("Python:", sys.version.split()[0])
for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
    print(f"{package}: {version(package)}")
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("bfloat16 supported:", torch.cuda.is_bf16_supported())
PYTHON
```

For training, CUDA must be available and the device must support the configured computation. If CUDA is unavailable, check the driver and the installed PyTorch build before running `fit`. The `--device` index refers to the GPUs visible to the process; after selecting one GPU with `CUDA_VISIBLE_DEVICES`, use `--device cuda:0`. Memory use depends on input length and the local environment, so no fixed GPU memory capacity is assumed here.

### Model Cache and Offline Use

The default model identifier is `Qwen/Qwen2.5-Coder-7B-Instruct`. The code pins its upstream model revision and disables remote model code. With network access, Transformers downloads missing tokenizer files for `audit` and the model files needed by `fit`.

Choose a writable Hugging Face cache directory before running Python:

```bash
export HF_HOME="$HOME/.cache/huggingface"
```

To work offline, provide a local model directory and enable offline mode. The tokenizer files must already be present for `audit`; `fit` also requires the model configuration and all safetensors weight files. Replace the example model path with your own directory. See the [Hugging Face environment-variable reference](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) for cache and offline settings.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m baselines.SWERouter5.run audit \
  --prepared-dir data/swe_router/prepared --swe-root data/swe_agent \
  --base-model /path/to/Qwen2.5-Coder-7B-Instruct \
  --k 3 --output-dir outputs/swe_offline_audit
```

The same `--base-model` option is supported by both runners and by `fit`. Keep the tokenizer and model weights from the same model source. Full training starts only when you run `fit`; it does not call the candidate-model APIs or rerun the benchmark agents.

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

This entry point uses `deepseek-v4-flash-0731` as the explorer. The training, validation, and test sets contain 160, 40, and 100 tasks respectively.

## Four-Benchmark Hybrid Extension

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

Flash remains the default explorer. Add `--explorer-model deepseek-v4-pro-0813` to use Pro-0813 and its corresponding trajectories and billing records. The training, validation, and test sets contain 867, 233, and 382 tasks respectively.

`audit` returns exit code 0 when the data and costs satisfy the fitting requirements, and 2 otherwise. Missing prefix costs are not estimated from token ratios. `fit` writes input hashes, audit reports, the validation-selected policy, the classifier adapter, and per-task test decisions. Existing output directories are rejected; failed training attempts retain a temporary directory for inspection.

The original task, thoughts, and actions remain complete. Long observations may be shortened only through deterministic head-and-tail excerpts with an explicit omission marker. Inputs that still exceed the length limit cause the run to stop rather than silently removing information available at the decision point.
