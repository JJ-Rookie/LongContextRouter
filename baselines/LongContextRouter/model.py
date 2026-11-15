import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
try:
    from .modules import LongTextEmbedder
except ImportError:
    from modules import LongTextEmbedder





class LongContextRouter(nn.Module):

    def __init__(
        self,

        embedder_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        embedder_max_chunk_length_k: float = 8.,
        embedder_overlap_tokens_k: float = 1.,
        embedder_aggregator_n_layers: int = 1,
        embedder_aggregator_dropout: float = 0.0,
        embedder_qk_norm: bool = True,
        embedder_rope_base: float = 10000.0,
        embedder_chunk_batch_size: int = 4,
        embedder_chunk_workers: int = 0,
        embedder_backend: str = "transformers",
        embedding_cache_path: Optional[str] = None,
        fusion_mode: str = "product",

        alpha: float = 0.01,
        num_LLM_models: int = 10,
        num_classes: int = 2,

        device: str = "cuda",
    ):
        super(LongContextRouter, self).__init__()
        assert embedder_max_chunk_length_k > 0 and embedder_max_chunk_length_k >= embedder_overlap_tokens_k
        if not 0.0 <= embedder_aggregator_dropout < 1.0:
            raise ValueError("embedder_aggregator_dropout must be in [0, 1)")
        if fusion_mode not in {"product", "sigmoid_p_mul_q", "p_mul_sigmoid_q"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

        embedder_max_chunk_length = int(embedder_max_chunk_length_k * 1024)
        embedder_overlap_tokens = int(embedder_overlap_tokens_k * 1024)

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

        self.text_dim = self.embedder.dim

        self.P = nn.Embedding(num_LLM_models, self.text_dim)
        self.alpha = alpha

        self.classifier = nn.Linear(self.text_dim, num_classes)
        self.to(device)

    def _classify_prompt_embeddings(self, model, q, test_mode=False):
        p = self.P(model)
        if not test_mode:
            q = q + torch.randn_like(q) * self.alpha
        if self.fusion_mode == "sigmoid_p_mul_q":
            p = torch.sigmoid(p)
        elif self.fusion_mode == "p_mul_sigmoid_q":
            q = torch.sigmoid(q)
        sim_matrix = torch.einsum("ND,BD->BND", p, q)
        return self.classifier(sim_matrix)

    def forward(self, model, prompt, test_mode=False):
        q = self.embedder(
            prompt,
            chunk_batch_size=self.embedder_chunk_batch_size,
            chunk_workers=self.embedder_chunk_workers,
        )
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
        logits = self.forward(model, prompt, test_mode=True)
        success_margin = logits[:, :, 1] - logits[:, :, 0]
        return torch.argmax(success_margin, dim=1)
