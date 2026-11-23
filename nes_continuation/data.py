"""Janelas para previsão do próximo token nas partições oficiais.

Lê os tokens já preparados pelo step01 (um .npz por música, chave "ids").
"""

from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .representation import PAD, TOKENS_DIR, load_ids, token_files


class NESWindows(Dataset):
    def __init__(
        self,
        root: str | Path = TOKENS_DIR,
        split: str = "train",
        context: int = 1024,
        stride: int | None = None,
        limit: int | None = None,
    ) -> None:
        if context < 1:
            raise ValueError("context deve ser positivo")
        if stride is None:
            stride = max(1, context // 2)
        if not 1 <= stride <= context:
            raise ValueError("stride deve estar entre 1 e context")
        if limit is not None and limit < 1:
            raise ValueError("limit deve ser positivo")

        paths = token_files(split, root)[:limit]
        self.context = context
        self.names: list[str] = []
        self.songs: list[np.ndarray] = []
        self.windows: list[tuple[int, int]] = []
        # Número de alvos (todos os tokens depois do BOS); com stride = context
        # cada alvo aparece em exatamente uma janela.
        self.target_count = 0
        for path in paths:
            tokens = load_ids(path).astype(np.uint16)
            if len(tokens) < 2:
                continue

            song_index = len(self.songs)
            self.names.append(path.stem)
            self.songs.append(tokens)
            target_count = len(tokens) - 1
            self.target_count += target_count
            for start in range(0, target_count, stride):
                self.windows.append((song_index, start))
                if start + context >= target_count:
                    break

        if not self.windows:
            raise ValueError(f"Nenhuma música encontrada em {Path(root) / split}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        song_index, start = self.windows[index]
        tokens = self.songs[song_index][start:start + self.context + 1]
        return torch.from_numpy(tokens.astype(np.int64))


def collate_windows(windows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Retorna entradas e alvos deslocados em uma posição, com PAD à direita."""
    padded = pad_sequence(windows, batch_first=True, padding_value=PAD)
    return padded[:, :-1].contiguous(), padded[:, 1:].contiguous()
