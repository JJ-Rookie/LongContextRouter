import atexit
import contextlib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple

try:
    from .embedding_cache import ChunkEmbeddingCache
except ImportError:
    from embedding_cache import ChunkEmbeddingCache


_VLLM_CLEANUP_REGISTERED = False


def _cleanup_vllm_distributed() -> None:
    with contextlib.suppress(Exception):
        from vllm.distributed import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
        return

    with contextlib.suppress(Exception):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _register_vllm_distributed_cleanup() -> None:
    global _VLLM_CLEANUP_REGISTERED
    if _VLLM_CLEANUP_REGISTERED:
        return
    atexit.register(_cleanup_vllm_distributed)
    _VLLM_CLEANUP_REGISTERED = True


def _select_attention_backend() -> str:
    if os.environ.get("LCR_DISABLE_FLASH", "0") == "1":
        print("[LongContextRouter] LCR_DISABLE_FLASH=1, skipping flash-attn probe")
    elif torch.cuda.is_available():
        try:
            import flash_attn_2_cuda as flash_attn_gpu
            from flash_attn import flash_attn_func
            q = torch.randn(1, 2, 2, 4, device="cuda")
            k = torch.randn(1, 2, 2, 4, device="cuda")
            v = torch.randn(1, 2, 2, 4, device="cuda")
            flash_attn_func(q, k, v, causal=True)
            return "flash"
        except Exception:
            pass

    if hasattr(F, "scaled_dot_product_attention"):
        return "sdpa"
    return "manual"


_ATTN_BACKEND = _select_attention_backend()
print(f"[LongContextRouter] attention backend: {_ATTN_BACKEND}")



class BaseEmbeddingModel(nn.Module):

    def __init__(
        self,
        name: str,
        max_chunk_length: int,
        overlap_tokens: int,
        device: str,
    ):
        super().__init__()
        self.name = name
        self.max_chunk_length = max_chunk_length
        self.overlap_tokens = overlap_tokens
        self.stride = max_chunk_length - overlap_tokens
        self.device = device
        self.tokenizer = None
        self.dim = None

    @torch.no_grad()
    def forward(
        self,
        texts: List[str],
        chunk_batch_size: int = 4,
        chunk_workers: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        if not texts:
            return (
                torch.empty(0, 0, self.dim, device=self.device),
                torch.empty(0, 0, 0, dtype=torch.bool, device=self.device),
                [],
            )

        encodings = self.tokenize_batch(texts)

        all_chunks = []
        chunk_counts = []
        if chunk_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=chunk_workers) as ex:
                chunk_lists = list(ex.map(
                    lambda enc: self.chunk(enc["input_ids"], enc["attention_mask"]),
                    encodings,
                ))
        else:
            chunk_lists = [
                self.chunk(enc["input_ids"], enc["attention_mask"])
                for enc in encodings
            ]

        for chunks in chunk_lists:
            all_chunks.extend(chunks)
            chunk_counts.append(len(chunks))

        batch = len(texts)
        max_n_chunk = max(chunk_counts) if chunk_counts else 0

        if not all_chunks:
            return (
                torch.empty(batch, 0, self.dim, device=self.device),
                torch.empty(batch, 0, 0, dtype=torch.bool, device=self.device),
                chunk_counts,
            )

        flat_input_ids = torch.cat([c["input_ids"] for c in all_chunks])
        flat_attention_mask = torch.cat([c["attention_mask"] for c in all_chunks])
        flat_chunk_embs = self.encode_batched(
            flat_input_ids, flat_attention_mask, batch_size=chunk_batch_size
        )

        chunk_embs = torch.nn.utils.rnn.pad_sequence(
            flat_chunk_embs.split(chunk_counts), batch_first=True
        )

        idx = torch.arange(max_n_chunk, device=self.device)
        causal = idx.unsqueeze(0) > idx.unsqueeze(1)
        counts_t = torch.tensor(chunk_counts, device=self.device)
        pad = torch.arange(max_n_chunk, device=self.device) >= counts_t.unsqueeze(1)
        attention_mask = causal.unsqueeze(0) | pad.unsqueeze(2) | pad.unsqueeze(1)

        return chunk_embs, attention_mask, chunk_counts

    def tokenize(self, text):
        return self.tokenizer(
            text,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )

    def tokenize_batch(self, texts: List[str]) -> List[Dict[str, torch.Tensor]]:
        enc = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_tensors=None,
        )
        return [
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
            for input_ids, attention_mask in zip(enc["input_ids"], enc["attention_mask"])
        ]

    def _pad_chunk(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_len = self.max_chunk_length - input_ids.shape[0]
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id
            input_ids = F.pad(input_ids, (0, pad_len), value=pad_id)
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
        return input_ids, attention_mask

    def chunk(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[dict]:
        seq_len = input_ids.shape[0]

        if seq_len <= self.max_chunk_length:
            chunk_input_ids, chunk_attention_mask = self._pad_chunk(
                input_ids, attention_mask
            )
            return [{
                "input_ids": chunk_input_ids.unsqueeze(0),
                "attention_mask": chunk_attention_mask.unsqueeze(0),
                "num_chunks": 1,
                "chunk_idx": 0,
            }]

        chunks = []
        num_chunks = 0
        for start_idx in range(0, seq_len, self.stride):
            end_idx = min(start_idx + self.max_chunk_length, seq_len)
            chunk_input_ids = input_ids[start_idx:end_idx]
            chunk_attention_mask = attention_mask[start_idx:end_idx]

            chunk_input_ids, chunk_attention_mask = self._pad_chunk(
                chunk_input_ids, chunk_attention_mask
            )

            chunks.append({
                "input_ids": chunk_input_ids.unsqueeze(0),
                "attention_mask": chunk_attention_mask.unsqueeze(0),
                "chunk_idx": num_chunks,
            })
            num_chunks += 1
            if end_idx >= seq_len:
                break

        for c in chunks:
            c["num_chunks"] = num_chunks
        return chunks

    def last_token_pool(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_lens = attention_mask.sum(dim=1) - 1
        return hidden_state[torch.arange(hidden_state.size(0), device=hidden_state.device), seq_lens]

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def encode_batched(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int = 4,
        offload_to_cpu: bool = True,
        empty_cache_between_batches: bool = True,
    ) -> torch.Tensor:
        N = input_ids.size(0)
        if N <= batch_size:
            try:
                return self.encode(input_ids, attention_mask)
            except torch.cuda.OutOfMemoryError:
                if not torch.cuda.is_available() or batch_size <= 1:
                    raise
                torch.cuda.empty_cache()
                return self.encode_batched(
                    input_ids, attention_mask,
                    batch_size=max(1, batch_size // 2),
                    offload_to_cpu=offload_to_cpu,
                    empty_cache_between_batches=empty_cache_between_batches,
                )

        all_embs = []
        current_bs = batch_size
        i = 0
        while i < N:
            current_bs = min(current_bs, N - i)
            end = i + current_bs
            try:
                emb = self.encode(input_ids[i:end], attention_mask[i:end])
            except torch.cuda.OutOfMemoryError:
                if current_bs <= 1:
                    raise
                torch.cuda.empty_cache()
                current_bs = max(1, current_bs // 2)
                print(
                    f"[{self.__class__.__name__}] CUDA OOM while encoding chunks {i}-{end}. "
                    f"Reducing encoder batch size to {current_bs} and retrying."
                )
                continue

            if offload_to_cpu:
                emb = emb.cpu()
            all_embs.append(emb)
            if empty_cache_between_batches and torch.cuda.is_available():
                torch.cuda.empty_cache()
            i = end

        if offload_to_cpu:
            return torch.cat(all_embs, dim=0).to(self.device)
        return torch.cat(all_embs, dim=0)


class TransformersEmbeddingModel(BaseEmbeddingModel):

    def __init__(
        self,
        name: str = "Qwen/Qwen3-Embedding-0.6B",
        max_chunk_length: int = 8192,
        overlap_tokens: int = 256,
        device: str = "cuda",
    ):
        super().__init__(name, max_chunk_length, overlap_tokens, device)
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("pip install transformers")

        self.tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        if torch.cuda.is_available() and _ATTN_BACKEND == "flash":
            attn_implementation = "flash_attention_2"
        elif torch.cuda.is_available():
            attn_implementation = "sdpa"
            print(
                "[TransformersEmbeddingModel] Flash Attention 2 is not available or incompatible "
                "with this PyTorch/CUDA build. Falling back to SDPA. To use flash-attn, "
                "install it with: pip install flash-attn --no-build-isolation"
            )
        else:
            attn_implementation = "eager"

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModel.from_pretrained(
            name,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            dtype=dtype,
            low_cpu_mem_usage=torch.cuda.is_available(),
            device_map="auto",
        )
        self.hidden_size = self.model.config.hidden_size
        self.dim = self.hidden_size
        self.to(device)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        seq_lens = attention_mask.sum(dim=1)
        max_len = int(seq_lens.max().item())
        if max_len < input_ids.size(1):
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.last_token_pool(outputs.last_hidden_state, attention_mask)
        return F.normalize(embeddings, p=2, dim=-1)


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):

    def __init__(
        self,
        name: str = "Qwen/Qwen3-Embedding-0.6B",
        max_chunk_length: int = 8192,
        overlap_tokens: int = 256,
        device: str = "cuda",
    ):
        super().__init__(name, max_chunk_length, overlap_tokens, device)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "SentenceTransformer backend requires: pip install sentence-transformers"
            )

        self.st = SentenceTransformer(name, trust_remote_code=True, device=device)
        self.tokenizer = self.st.tokenizer
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = 0
        self.dim = self.st.get_sentence_embedding_dimension()

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        texts = []
        for ids, mask in zip(input_ids, attention_mask):
            real_ids = ids[mask == 1].tolist()
            texts.append(self.tokenizer.decode(real_ids, skip_special_tokens=True))

        embs = self.st.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=len(texts),
            show_progress_bar=False,
            device=str(self.device),
        )
        return embs.to(self.device)


class VLLMEmbeddingModel(BaseEmbeddingModel):

    def __init__(
        self,
        name: str = "Qwen/Qwen3-Embedding-0.6B",
        max_chunk_length: int = 8192,
        overlap_tokens: int = 256,
        device: str = "cuda",
        gpu_memory_utilization: float = 0.85,
    ):
        super().__init__(name, max_chunk_length, overlap_tokens, device)
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("vLLM backend requires: pip install vllm")
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("pip install transformers")

        self.tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        _register_vllm_distributed_cleanup()
        self.llm = LLM(
            model=name,
            task="embed",
            trust_remote_code=True,
            dtype="auto",
            max_model_len=max_chunk_length,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        model_config = self.llm.llm_engine.model_config
        self.dim = getattr(model_config, "hidden_size", None)
        if self.dim is None:
            self.dim = getattr(getattr(model_config, "hf_config", None), "hidden_size", None)
        if self.dim is None:
            self.dim = getattr(getattr(model_config, "hf_text_config", None), "hidden_size", None)
        if self.dim is None:
            raise RuntimeError(
                "Cannot infer embedding dimension from vLLM model config. "
                "Please set the model's hidden_size manually."
            )

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        prompts = []
        for ids, mask in zip(input_ids, attention_mask):
            real_ids = ids[mask == 1].tolist()
            prompts.append({"prompt_token_ids": real_ids})

        outputs = self.llm.embed(prompts)
        embs = [_extract_vllm_embedding(out) for out in outputs]
        return torch.stack(embs).to(self.device)


def _extract_vllm_embedding(output) -> torch.Tensor:
    import numpy as np

    data = output
    if hasattr(data, "outputs"):
        data = data.outputs
        if isinstance(data, (list, tuple)) and len(data) > 0:
            data = data[0]

    emb = None
    for attr in ("embedding", "data", "embeddings"):
        emb = getattr(data, attr, None)
        if emb is not None:
            break

    if emb is None:
        raise ValueError(f"Cannot extract embedding from vLLM output: {output}")

    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)
    elif not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb)
    return emb.float()


EmbeddingModel = TransformersEmbeddingModel


_EMBEDDING_BACKEND_MAP = {
    "transformers": TransformersEmbeddingModel,
    "sentence_transformers": SentenceTransformerEmbeddingModel,
    "sentence-transformers": SentenceTransformerEmbeddingModel,
    "vllm": VLLMEmbeddingModel,
}


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[-2]
        device, dtype = x.device, x.dtype
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0)
        sin = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)
        return cos, sin


class GQASelfAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int = 16,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        dropout: float = 0.0,
        qk_norm: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.qk_norm = qk_norm
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.dropout_p = float(dropout)

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, dtype=torch.float32)
            self.k_norm = RMSNorm(self.head_dim, dtype=torch.float32)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=rope_base)

    def _manual_attention(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _, _, seqlen, _ = xq.shape
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is None:
            causal_mask = torch.triu(
                torch.ones(seqlen, seqlen, device=xq.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        else:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        return torch.matmul(attn, xv)

    def _flash_attention(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_func
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        dropout_p = self.dropout_p if self.training else 0.0
        out = flash_attn_func(
            xq,
            xk,
            xv,
            dropout_p=dropout_p,
            causal=causal,
        )
        return out.transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        cos, sin = self.rotary(xq, seq_len=seqlen)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)

        use_flash = (_ATTN_BACKEND == "flash") and (mask is None)
        dropout_p = self.dropout_p if self.training else 0.0

        if use_flash:
            out = self._flash_attention(xq, xk, xv, causal=True)
        elif _ATTN_BACKEND in ("flash", "sdpa"):
            if mask is None:
                out = F.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    dropout_p=dropout_p,
                    is_causal=True,
                )
            else:
                sdpa_mask = torch.zeros(
                    mask.shape,
                    dtype=xq.dtype,
                    device=xq.device,
                ).masked_fill(mask.to(xq.device), float("-inf"))
                out = F.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    attn_mask=sdpa_mask.unsqueeze(1),
                    dropout_p=dropout_p,
                )
        else:
            out = self._manual_attention(xq, xk, xv, mask)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int = 16,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        ffn_dim: int = 3072,
        dropout: float = 0.0,
        qk_norm: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attn = GQASelfAttention(
            dim, n_heads, n_kv_heads, head_dim, dropout,
            qk_norm=qk_norm, rope_base=rope_base,
        )
        self.ffn = SwiGLU(dim, ffn_dim, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(hidden_states, mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class ChunkAggregator(nn.Module):

    def __init__(
        self,
        dim: int = 1024,
        n_layers: int = 1,
        n_heads: int = 16,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        ffn_dim: int = 3072,
        dropout: float = 0.0,
        qk_norm: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(TransformerBlock(
                dim, n_heads, n_kv_heads, head_dim, ffn_dim, dropout,
                qk_norm=qk_norm, rope_base=rope_base,
            ))
        self.norm = RMSNorm(dim)

    def forward(
        self,
        chunk_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_counts: Optional[List[int]] = None,
    ) -> torch.Tensor:
        squeeze = chunk_embeddings.dim() == 2
        if squeeze:
            chunk_embeddings = chunk_embeddings.unsqueeze(0)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0)

        x = chunk_embeddings

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        x = self.norm(x)

        if chunk_counts is not None:
            output = torch.stack( [x[i, count - 1, :] for i, count in enumerate(chunk_counts)], dim=0 )
        else:
            output = x[:, -1, :]

        return output.squeeze(0) if squeeze else output


class LongTextEmbedder(nn.Module):

    def __init__(
        self,
        embedder_model_name: str,
        max_chunk_length: int = 8192,
        overlap_tokens: int = 1024,
        n_aggregator_layers: int = 1,
        n_heads: int = 16,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        ffn_dim: int = 3072,
        dropout: float = 0.0,
        qk_norm: bool = True,
        rope_base: float = 10000.0,
        device: str = "cuda",
        embedding_backend: str = "transformers",
        embedding_cache_path: Optional[str] = None,
    ):
        super().__init__()
        self.device = device
        self.embedding_cache = None
        if embedding_cache_path:
            self.embedding_cache = ChunkEmbeddingCache.load(embedding_cache_path)
            cache_config = self.embedding_cache.config
            expected_config = {
                "model_name": str(embedder_model_name),
                "max_chunk_length": int(max_chunk_length),
                "overlap_tokens": int(overlap_tokens),
            }
            for key, expected_value in expected_config.items():
                if cache_config.get(key) != expected_value:
                    raise ValueError(
                        f"embedding cache configuration mismatch for {key}: "
                        f"expected {expected_value!r}, got {cache_config.get(key)!r}"
                    )
            cached_backend = cache_config.get("backend")
            if cached_backend is not None and cached_backend != embedding_backend:
                raise ValueError(
                    "embedding cache configuration mismatch for backend: "
                    f"expected {embedding_backend!r}, got {cached_backend!r}"
                )
            self.embedding_model = None
            self.dim = int(cache_config["embedding_dim"])
        else:
            backend_cls = _EMBEDDING_BACKEND_MAP.get(embedding_backend)
            if backend_cls is None:
                raise ValueError(
                    f"Unknown embedding backend: {embedding_backend}. "
                    f"Choose from {list(_EMBEDDING_BACKEND_MAP.keys())}"
                )
            self.embedding_model = backend_cls(
                name=embedder_model_name,
                max_chunk_length=max_chunk_length,
                overlap_tokens=overlap_tokens,
                device=device,
            )
            self.dim = self.embedding_model.dim

        if n_aggregator_layers == 0:
            self.aggregator = nn.Identity()
        else:
            self.aggregator = ChunkAggregator(
                dim=self.dim,
                n_layers=n_aggregator_layers,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
                qk_norm=qk_norm,
                rope_base=rope_base,
            )
        self.to(device)

    def validate_cached_prompts(self, prompts: List[str]) -> None:
        if getattr(self, "embedding_cache", None) is None:
            return
        self.embedding_cache.validate_prompts(prompts)

    def aggregate_chunk_embeddings(
        self,
        chunk_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_counts: List[int],
    ) -> torch.Tensor:
        if chunk_embeddings.size(1) == 0:
            return torch.empty(chunk_embeddings.size(0), self.dim, device=self.device)

        chunk_embeddings = chunk_embeddings.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if isinstance(self.aggregator, nn.Identity):
            mask_float = (~attention_mask[:, :, 0]).float()
            summed = (chunk_embeddings * mask_float.unsqueeze(-1)).sum(dim=1)
            aggregated = summed / mask_float.sum(dim=1, keepdim=True).clamp_min(1)
        else:
            aggregated = self.aggregator(
                chunk_embeddings,
                attention_mask=attention_mask,
                chunk_counts=chunk_counts,
            )
        return F.normalize(aggregated, p=2, dim=-1)

    def forward(
        self,
        texts,
        chunk_batch_size: int = 4,
        chunk_workers: int = 0,
    ) -> torch.Tensor:
        if isinstance(texts, str): texts = [texts]

        if getattr(self, "embedding_cache", None) is not None:
            chunk_embs, attention_mask, chunk_counts = self.embedding_cache.get_batch(
                texts, device=self.device
            )
        else:
            if self.embedding_model is None:
                raise RuntimeError("raw embedding backend is not initialized")
            chunk_embs, attention_mask, chunk_counts = self.embedding_model(
                texts,
                chunk_batch_size=chunk_batch_size,
                chunk_workers=chunk_workers,
            )
        return self.aggregate_chunk_embeddings(chunk_embs, attention_mask, chunk_counts)
