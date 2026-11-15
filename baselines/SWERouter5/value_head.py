from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Mapping, Sequence


MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
LABEL_WIDTH = 5


def effective_warmup_steps(total_steps: int, requested_steps: int) -> int:
    if total_steps < 1 or requested_steps < 0:
        raise ValueError("invalid optimizer schedule")
    return min(requested_steps, max(1, total_steps // 10))


def _excerpt_observation(observation: str, limit: int | None) -> str:
    if limit is None or len(observation) <= limit:
        return observation
    head = limit // 2
    tail = limit - head
    omitted = len(observation) - limit
    return (observation[:head] + f"\n[... observation middle omitted: {omitted} characters ...]\n"
            + observation[-tail:])


def _mmau_problem_view(problem: str, limit: int | None) -> tuple[str, int]:
    if limit is None:
        return problem, 0
    try:
        payload = json.loads(problem)
    except json.JSONDecodeError as exc:
        raise ValueError("MMAU task is not valid JSON") from exc
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list) or any(not isinstance(item, dict) for item in messages):
        raise ValueError("MMAU task has malformed messages")
    compact = []
    omitted = 0
    for item in messages:
        content = item.get("content")
        if item.get("role") == "tool":
            if not isinstance(content, str):
                raise ValueError("MMAU tool observation is not text")
            omitted += len(content) > limit
            compact.append({**item, "content": _excerpt_observation(content, limit)})
        else:
            compact.append(item)
    return json.dumps({**payload, "messages": compact}, ensure_ascii=False,
                      sort_keys=True), omitted


def format_value_input(row: Mapping, *, observation_chars: int | None = None) -> str:
    problem, prefix = row.get("problem"), row.get("prefix_text")
    explorer = row.get("explorer_model", "deepseek-v4-flash-0731")
    labels = {"deepseek-v4-flash-0731": "Flash",
              "deepseek-v4-pro-0813": "DeepSeek-V4-Pro-0813"}
    if explorer not in labels:
        raise ValueError("unknown explorer model in value-head input")
    explorer_label = labels[explorer]
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("original SWE task is empty")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("observed trajectory prefix is empty")
    if row.get("probe_kind") == "complete_answer":
        bench = row.get("bench")
        if bench not in ("bbh", "memory_agent_bench", "mmau"):
            raise ValueError("complete-answer probe has an invalid benchmark")
        if bench == "mmau":
            problem, _ = _mmau_problem_view(problem, observation_chars)
        return (f"Benchmark: {bench}\nOriginal task:\n{problem}\n\n"
                f"Observed {explorer_label} recorded response:\n{prefix}")
    if row.get("probe_kind") not in (None, "agent_prefix"):
        raise ValueError("unknown Flash probe kind")
    steps = row.get("prefix_steps")
    if steps:
        blocks = []
        for number, step in enumerate(steps, 1):
            if not isinstance(step, Mapping) or any(
                not isinstance(step.get(name), str)
                for name in ("thought", "action", "observation")
            ):
                raise ValueError("structured trajectory step is malformed")
            blocks.append(
                f"Step {number}\nThought: {step['thought']}\n"
                f"Action: {step['action']}\nObservation: "
                + _excerpt_observation(step["observation"], observation_chars)
            )
        prefix = "\n\n".join(blocks)
    return f"SWE task:\n{problem}\n\nObserved {explorer_label} agent steps:\n{prefix}"


def _encode_one(row: Mapping, tokenizer, max_length: int) -> tuple[dict, dict]:
    if max_length < 128 or max_length > 8192:
        raise ValueError("max_length must be between 128 and 8192")
    steps = row.get("prefix_steps")
    for limit in (None, 8192, 4096, 2048, 1024, 512, 256, 128):
        text = format_value_input(row, observation_chars=limit)
        encoded = tokenizer(text, truncation=False, add_special_tokens=True)
        token_count = len(encoded["input_ids"])
        if token_count <= max_length:
            truncated = sum(len(step["observation"]) > limit for step in steps) if (
                steps and limit is not None) else 0
            if row.get("probe_kind") == "complete_answer" and row.get("bench") == "mmau":
                truncated += _mmau_problem_view(row["problem"], limit)[1]
            return dict(encoded), {
                "prompt_id": row["prompt_id"], "instance_id": row["instance_id"],
                "input_tokens": token_count,
                "observation_char_limit": limit,
                "truncated_observations": truncated,
            }
        if not steps and not (row.get("probe_kind") == "complete_answer"
                              and row.get("bench") == "mmau"):
            break
    raise ValueError(f"{row['instance_id']}: value-head input exceeds "
                     f"max_length={max_length}; observed steps must not be cut")


def preflight_value_inputs(examples: Mapping[str, Sequence[Mapping]], *,
                           base_model: str = MODEL_ID, max_length: int = 8192) -> dict:
    tokenizer = _load_tokenizer(base_model)
    return {split: [_encode_one(row, tokenizer, max_length)[1] for row in examples[split]]
            for split in ("train", "validation", "test")}


def validate_examples(train: Sequence[Mapping], validation: Sequence[Mapping]) -> None:
    if not train or not validation:
        raise ValueError("training and validation examples must be nonempty")
    train_ids = {row["instance_id"] for row in train}
    validation_ids = {row["instance_id"] for row in validation}
    if len(train_ids) != len(train) or len(validation_ids) != len(validation):
        raise ValueError("duplicate instance ID within a split")
    if train_ids & validation_ids:
        raise ValueError("training and validation instance IDs overlap")
    for row in (*train, *validation):
        format_value_input(row)
        if len(row["labels"]) != LABEL_WIDTH or any(value not in (0, 1) for value in row["labels"]):
            raise ValueError("five binary outcome labels are required")


def _load_tokenizer(base_model: str):
    from transformers import AutoTokenizer

    revision = MODEL_REVISION if base_model == MODEL_ID else None
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True,
                                              revision=revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_base(base_model: str, tokenizer, *, device: str):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, BitsAndBytesConfig

    if not torch.cuda.is_available() or not device.startswith("cuda:"):
        raise RuntimeError("the 7B QLoRA value head requires a CUDA device")
    revision = MODEL_REVISION if base_model == MODEL_ID else None
    config = AutoConfig.from_pretrained(
        base_model, revision=revision, trust_remote_code=False,
        num_labels=LABEL_WIDTH, problem_type="multi_label_classification",
    )
    if config.model_type != "qwen2":
        raise ValueError("the SWE-Router value head requires a qwen2 configuration")
    config._attn_implementation = "sdpa"
    quantization = BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.bfloat16,
                                      bnb_4bit_use_double_quant=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, config=config,
        quantization_config=quantization, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": device}, trust_remote_code=False, use_safetensors=True,
        revision=revision,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    return model


def _encode_rows(rows: Sequence[Mapping], tokenizer, max_length: int):
    import torch
    from torch.utils.data import Dataset

    encoded_rows = []
    view_records = []
    for row in rows:
        encoded, record = _encode_one(row, tokenizer, max_length)
        encoded_rows.append({**encoded, "labels": torch.tensor(row["labels"],
                                                        dtype=torch.float32)})
        view_records.append(record)

    class Encoded(Dataset):
        def __len__(self):
            return len(encoded_rows)

        def __getitem__(self, index):
            return encoded_rows[index]

    dataset = Encoded()
    dataset.view_records = view_records
    return dataset


def _collator(tokenizer):
    import torch

    def collate(items):
        labels = torch.stack([item["labels"] for item in items])
        encoded = tokenizer.pad([{key: value for key, value in item.items() if key != "labels"}
                                 for item in items], return_tensors="pt")
        return {**encoded, "labels": labels}

    return collate


def _predict(model, dataset, tokenizer) -> list[list[float]]:
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(dataset, batch_size=1,
                        shuffle=False, collate_fn=_collator(tokenizer))
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels")
            batch = {key: value.to(model.device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=model.device.type == "cuda"):
                logits = model(**batch).logits
            predictions.append(torch.sigmoid(logits.float())[0].cpu().tolist())
    return predictions


def train_value_head(
    train: Sequence[Mapping], validation: Sequence[Mapping], output_dir: Path,
    *, base_model: str = MODEL_ID, max_length: int = 8192,
    epochs: int = 5, learning_rate: float = 5e-5, warmup_steps: int = 50,
    gradient_accumulation: int = 16, seed: int = 43, device: str = "cuda:0",
) -> dict:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader

    validate_examples(train, validation)
    if epochs < 1 or gradient_accumulation < 1 or learning_rate <= 0 or warmup_steps < 0:
        raise ValueError("invalid training schedule")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"training output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    torch.manual_seed(seed)
    tokenizer = _load_tokenizer(base_model)
    train_dataset = _encode_rows(train, tokenizer, max_length)
    val_dataset = _encode_rows(validation, tokenizer, max_length)
    model = _load_base(base_model, tokenizer, device=device)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    adapter = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=32, lora_alpha=64, lora_dropout=0.05,
        bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, adapter)
    model.config.use_cache = False
    train_loader = DataLoader(train_dataset, batch_size=1,
                              shuffle=True, collate_fn=_collator(tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, collate_fn=_collator(tokenizer))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=learning_rate)
    total_steps = math.ceil(len(train_loader) / gradient_accumulation) * epochs
    effective_warmup = effective_warmup_steps(total_steps, warmup_steps)

    def schedule(step):
        if step < effective_warmup:
            return (step + 1) / max(1, effective_warmup)
        fraction = (step - effective_warmup) / max(1, total_steps - effective_warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1, fraction)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    best_loss = math.inf
    best_epoch = 0
    history = []
    adapter_dir = output_dir / "adapter"
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for index, batch in enumerate(train_loader, 1):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=model.device.type == "cuda"):
                loss = model(**batch).loss
                train_loss += float(loss.detach())
                group_start = ((index - 1) // gradient_accumulation) * gradient_accumulation
                group_size = min(gradient_accumulation, len(train_loader) - group_start)
                (loss / group_size).backward()
            if index % gradient_accumulation == 0 or index == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                batch = {key: value.to(model.device) for key, value in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=model.device.type == "cuda"):
                    val_loss += float(model(**batch).loss)
        val_loss /= len(val_loader)
        history.append({"epoch": epoch, "train_bce": train_loss / len(train_loader),
                        "validation_bce": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir, save_jinja_files=False)
        print(f"epoch={epoch} train_bce={history[-1]['train_bce']:.6f} "
              f"validation_bce={val_loss:.6f}", flush=True)
    return {"best_epoch": best_epoch, "best_validation_bce": best_loss,
            "history": history, "adapter_dir": str(adapter_dir),
            "base_model": base_model, "max_length": max_length,
            "optimizer_steps": total_steps, "effective_warmup_steps": effective_warmup}


def predict_from_adapter(
    rows: Sequence[Mapping], adapter_dir: Path, *, base_model: str = MODEL_ID,
    max_length: int = 8192, device: str = "cuda:0",
) -> list[list[float]]:
    from peft import PeftModel

    tokenizer = _load_tokenizer(str(adapter_dir))
    dataset = _encode_rows(rows, tokenizer, max_length)
    base = _load_base(base_model, tokenizer, device=device)
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    return _predict(model, dataset, tokenizer)
