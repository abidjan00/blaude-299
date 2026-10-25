"""Blaude's decoder-only Transformer, built from basic PyTorch layers.

No pretrained weights or nn.Transformer wrapper are used. Inputs are token IDs
from Week 1; labels are already shifted by TokenDataset and must not shift again.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 4096
    context_length: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    pad_id: int = 0

    def __post_init__(self):
        sizes = (self.vocab_size, self.context_length, self.d_model,
                 self.n_heads, self.n_layers, self.d_ff)
        if any(not isinstance(value, int) or value < 1 for value in sizes):
            raise ValueError("Vocabulary, context, dimensions, heads, and layers must be positive integers")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 and 1 (excluding 1)")
        if not isinstance(self.pad_id, int) or not 0 <= self.pad_id < self.vocab_size:
            raise ValueError("pad_id must belong to the vocabulary")


class CausalSelfAttention(nn.Module):
    """Each position attends to itself and earlier real positions only."""

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.ones(config.context_length, config.context_length, dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, hidden, attention_mask):
        batch, length, width = hidden.shape
        # Split each token's vectors into query, key, and value, then into heads.
        qkv = self.qkv(hidden).reshape(batch, length, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = self.causal_mask[:length, :length][None, None, :, :]
        allowed = allowed & attention_mask[:, None, None, :]
        scores = scores.masked_fill(~allowed, float("-inf"))
        # The public model requires nonempty, right-padded rows. Every query has
        # at least the first real key, so softmax never sees an all-infinite row.
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        attended = (weights @ value).transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.output_dropout(self.projection(attended))


class TransformerBlock(nn.Module):
    """Pre-normalization, attention, and a feed-forward network with residuals."""

    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden, attention_mask):
        hidden = hidden + self.attention(self.attention_norm(hidden), attention_mask)
        hidden = hidden + self.feed_forward(self.feed_forward_norm(hidden))
        return hidden


class BlaudeModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.token_embedding = nn.Embedding(
            self.config.vocab_size, self.config.d_model, padding_idx=self.config.pad_id)
        self.position_embedding = nn.Embedding(self.config.context_length, self.config.d_model)
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(self.config) for _ in range(self.config.n_layers))
        self.final_norm = nn.LayerNorm(self.config.d_model)
        self.output = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.apply(self._initialize)
        with torch.no_grad():
            self.token_embedding.weight[self.config.pad_id].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, input_ids, attention_mask=None, labels=None):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be an int64 tensor shaped [batch, sequence]")
        batch, length = input_ids.shape
        if batch < 1 or not 1 <= length <= self.config.context_length:
            raise ValueError("Use a nonempty batch and a sequence within context_length")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("Input token ID is outside the vocabulary")
        if attention_mask is None:
            attention_mask = input_ids != self.config.pad_id
        if (attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool
                or attention_mask.device != input_ids.device):
            raise ValueError("attention_mask must be boolean, match input shape, and use the same device")
        if not torch.all(attention_mask[:, 0]):
            raise ValueError("Every sequence must start with a real token; use right padding only")
        if torch.any(~attention_mask[:, :-1] & attention_mask[:, 1:]):
            raise ValueError("attention_mask must contain real tokens followed by right padding")

        positions = torch.arange(length, device=input_ids.device)
        hidden = self.embedding_dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        logits = self.output(self.final_norm(hidden))
        # Padded predictions are unused and are set to zero for a clear contract.
        logits = logits.masked_fill(~attention_mask[:, :, None], 0.0)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape or labels.dtype != torch.long or labels.device != input_ids.device:
                raise ValueError("labels must be int64 and match input shape and device")
            effective_labels = labels.masked_fill(~attention_mask, -100)
            valid = effective_labels != -100
            if not torch.any(valid):
                raise ValueError("At least one real target is required to compute loss")
            if torch.any(effective_labels[valid] < 0) or torch.any(effective_labels[valid] >= self.config.vocab_size):
                raise ValueError("Target token ID is outside the vocabulary")
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size),
                                   effective_labels.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}
