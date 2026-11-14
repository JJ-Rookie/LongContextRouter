"""
LongContextRouter with Long Text Embedding (Qwen3-Embedding + Causal Transformer)

Replaces the original self.Q (nn.Embedding lookup) with dynamic long-text embedding
computed via: Raw Text -> Chunk -> Qwen3-Embed -> Causal Transformer Block
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
try:
    from .modules import LongTextEmbedder
except ImportError:
    # Allow running this file directly as a script.
    from modules import LongTextEmbedder

# Enable Python-level stack traces on segfaults / aborts (very helpful on WSL).
# import faulthandler

# faulthandler.enable()

# ============================================================================
# LongContextRouter
# Long Text Embedding with Qwen3-Embedding-0.6B + Causal Transformer Block Aggregation

# Pipeline:
#     N: batch size (number of prompts)
#     M: number of models
#     Long Text -> Tokenizer -> Split Chunks (with overlap) ->
#     Qwen3-Embedding -> [N x 1024] chunk embeddings ->
#     Causal Transformer Block -> Last Token Output as final representation [N x 1024]  --|
#                                                                                           ---> [N x M x 1024] Linear  --->  [N x M x 2]
#     Model Pool Embedding (P) -> [M x 1024] model embeddings                            --|
#
# Memory tuning:
#     If you hit CUDA OOM in the Qwen3 encoder, lower ``embedder_chunk_batch_size``
#     (default 4 for 8k chunks on a 24 GB GPU). The encoder also automatically
#     halves its per-forward chunk batch on ``torch.OutOfMemoryError`` down to 1.
#     Additionally, tokenization is now batched in C++, right-padding is trimmed
#     to the longest real chunk in each micro-batch, and ``chunk_workers`` allows
#     parallel chunking of very long batches.
#     For best efficiency install flash-attn: pip install flash-attn --no-build-isolation
#
# Backends:
#     The chunk encoder can be swapped via ``embedder_backend``:
#         - "transformers"  (default, original)
#         - "sentence_transformers"
#         - "vllm"
#     All backends share the same tokenization/chunking/masking logic.
# ============================================================================

class LongContextRouter(nn.Module):

    def __init__(
        self,

        # ===== LongTextEmbedder configs =====
        embedder_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        embedder_max_chunk_length_k: float = 8.,   # unit: k
        embedder_overlap_tokens_k: float = 1.,     # unit: k
        embedder_aggregator_n_layers: int = 1,  # if zero, no transformer aggregation, just mean pooling
        embedder_aggregator_dropout: float = 0.0,
        embedder_qk_norm: bool = True,
        embedder_rope_base: float = 10000.0,
        embedder_chunk_batch_size: int = 4,
        embedder_chunk_workers: int = 0,
        embedder_backend: str = "transformers",
        embedding_cache_path: Optional[str] = None,
        fusion_mode: str = "product",

        # Original configs
        alpha: float = 0.01,       # noise level for training
        num_LLM_models: int = 10,  # model pool size (number of models to select from)
        num_classes: int = 2,      # 0: fail, 1: success, one hot encoding for binary classification

        device: str = "cuda",
    ):
        super(LongContextRouter, self).__init__()
        assert embedder_max_chunk_length_k > 0 and embedder_max_chunk_length_k >= embedder_overlap_tokens_k
        if not 0.0 <= embedder_aggregator_dropout < 1.0:
            raise ValueError("embedder_aggregator_dropout must be in [0, 1)")
        if fusion_mode not in {"product", "sigmoid_p_mul_q", "p_mul_sigmoid_q"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

        embedder_max_chunk_length = int(embedder_max_chunk_length_k * 1024)  # convert to K length
        embedder_overlap_tokens = int(embedder_overlap_tokens_k * 1024)  # convert k

        self.embedder = LongTextEmbedder(
            embedder_model_name=embedder_model_name,
            max_chunk_length=embedder_max_chunk_length,
            overlap_tokens=embedder_overlap_tokens,
            n_aggregator_layers=embedder_aggregator_n_layers,
            dropout=embedder_aggregator_dropout,
            qk_norm=embedder_qk_norm,
            rope_base=embedder_rope_base,
            device=device,
            embedding_backend=embedder_backend,
            embedding_cache_path=embedding_cache_path,
        )
        self.embedder_chunk_batch_size = embedder_chunk_batch_size
        self.embedder_chunk_workers = embedder_chunk_workers
        self.fusion_mode = fusion_mode

        self.text_dim = self.embedder.dim              # embedding dimension (should match text_dim)

        # Model embedding network
        self.P = nn.Embedding(num_LLM_models, self.text_dim)
        self.alpha = alpha

        # =====  Long Text Embedder  =====
        #   dynamic embedding from raw text
        self.classifier = nn.Linear(self.text_dim, num_classes)
        self.to(device)

    def _classify_prompt_embeddings(self, model, q, test_mode=False):
        p = self.P(model)                                      # [Model Select Num, dim]
        if not test_mode:
            q = q + torch.randn_like(q) * self.alpha           # Adding noise to reduce overfitting (same as original)
        if self.fusion_mode == "sigmoid_p_mul_q":
            p = torch.sigmoid(p)
        elif self.fusion_mode == "p_mul_sigmoid_q":
            q = torch.sigmoid(q)
        sim_matrix = torch.einsum("ND,BD->BND", p, q)          # [Batch, dim] * [Model Select Num, dim] -> [Batch, Model Select Num, Dim]
        return self.classifier(sim_matrix)                     # [Batch, Model Select Num, 2]

    def forward(self, model, prompt, test_mode=False):
        q = self.embedder(
            prompt,
            chunk_batch_size=self.embedder_chunk_batch_size,
            chunk_workers=self.embedder_chunk_workers,
        )  # [Batch, dim], Batch is the number of prompts (1 for single prompt, >1 for batch)
        return self._classify_prompt_embeddings(model, q, test_mode=test_mode)

    def forward_from_chunk_embeddings(
        self,
        model,
        chunk_embeddings,
        attention_mask,
        chunk_counts,
        test_mode=False,
    ):
        q = self.embedder.aggregate_chunk_embeddings(
            chunk_embeddings,
            attention_mask=attention_mask,
            chunk_counts=chunk_counts,
        )
        return self._classify_prompt_embeddings(model, q, test_mode=test_mode)

    @torch.no_grad()
    def predict(self, model, prompt):
        logits = self.forward(model, prompt, test_mode=True)    # [Batch, Model Select Num, 2]
        success_margin = logits[:, :, 1] - logits[:, :, 0]
        return torch.argmax(success_margin, dim=1)              # [Batch],  the best model index for each prompt in the batch
