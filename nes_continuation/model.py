"""Modelos do próximo token em PyTorch nativo: Transformer somente decodificador e LSTM.

As duas arquiteturas usam o mesmo vocabulário (os 1155 tokens do step01), o mesmo
compartilhamento de pesos entre embedding e saída e a mesma interface
(`model(tokens) -> logits`, `model.config.context`), então treino, avaliação e
geração são os mesmos. `build_model(config)` escolhe pela `config.arch`.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

# len(VOCAB) do step01: os 1155 tokens, inclusive os que nunca aparecem nos dados.
from .representation import VOCAB_SIZE


ARCHITECTURES = ("transformer", "lstm")


@dataclass(frozen=True)
class ModelConfig:
    context: int = 1024
    width: int = 256
    heads: int = 8
    layers: int = 6
    dropout: float = 0.1
    # Checkpoints antigos (sem este campo) são do Transformer.
    arch: str = "transformer"

    def __post_init__(self) -> None:
        if self.arch not in ARCHITECTURES:
            raise ValueError(f"arch deve ser uma de {ARCHITECTURES}")
        if min(self.context, self.width, self.heads, self.layers) < 1:
            raise ValueError("context, width, heads e layers devem ser positivos")
        if self.arch == "transformer" and self.width % self.heads:
            raise ValueError("width deve ser divisível por heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout deve estar em [0, 1)")


class CausalBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.dropout = config.dropout
        self.attention_norm = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.attention_out = nn.Linear(config.width, config.width)
        self.ff_norm = nn.LayerNorm(config.width)
        self.ff = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        qkv = self.qkv(self.attention_norm(hidden))
        qkv = qkv.reshape(batch, length, 3, self.heads, width // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind()
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        hidden = hidden + self.residual_dropout(self.attention_out(attended))
        return hidden + self.residual_dropout(self.ff(self.ff_norm(hidden)))


class NESTransformer(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.width)
        self.position_embedding = nn.Embedding(config.context, config.width)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(CausalBlock(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.width)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Produz logits (lote, posições, vocabulário) para o próximo token."""
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.config.context:
            raise ValueError("tokens deve ter forma (lote, posições) dentro do contexto")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        # Compartilhamento de pesos entre a projeção de saída e o embedding de entrada.
        return F.linear(self.final_norm(hidden), self.token_embedding.weight)


class NESLSTM(nn.Module):
    """LSTM empilhado sobre os mesmos tokens, janelas e saída compartilhada do Transformer.

    `context` é o comprimento da janela (BPTT truncado: o estado começa zerado em cada
    janela, como o Transformer, que também só vê a janela), para que os dois modelos
    vejam exatamente os mesmos dados. `width` é a dimensão do embedding e do estado;
    `heads` não é usado. O LSTM roda em FP32 mesmo dentro do autocast (BF16/FP16),
    porque o suporte do cuDNN a RNN em BF16 depende da versão; o resto segue o autocast.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.width)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(
            config.width, config.width, num_layers=config.layers, batch_first=True,
            dropout=config.dropout if config.layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(config.dropout)
        self.final_norm = nn.LayerNorm(config.width)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Produz logits (lote, posições, vocabulário) para o próximo token."""
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.config.context:
            raise ValueError("tokens deve ter forma (lote, posições) dentro do contexto")
        hidden = self.embedding_dropout(self.token_embedding(tokens))
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            hidden, _ = self.lstm(hidden.float())
        hidden = self.final_norm(self.output_dropout(hidden))
        # Mesmo compartilhamento de pesos do Transformer: saída = embedding transposto.
        return F.linear(hidden, self.token_embedding.weight)


def build_model(config: ModelConfig) -> nn.Module:
    """Instancia a arquitetura indicada em config.arch."""
    return {"transformer": NESTransformer, "lstm": NESLSTM}[config.arch](config)
