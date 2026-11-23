"""Janelas para previsão do próximo token nas partições oficiais."""

from array import array
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .tokenizer import EVENTS, PAD, encode_midi


class NESWindows(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        context: int = 1024,
        stride: int | None = None,
        limit: int | None = None,
    ) -> None:
        if split not in {"train", "valid", "test"}:
            raise ValueError("Use uma partição oficial: train, valid ou test")
        if context < 1:
            raise ValueError("context deve ser positivo")
        if stride is None:
            stride = max(1, context // 2)
        if not 1 <= stride <= context:
            raise ValueError("stride deve estar entre 1 e context")
        if limit is not None and limit < 1:
            raise ValueError("limit deve ser positivo")

        folder = Path(root) / split
        paths = sorted(path for path in folder.iterdir() if path.suffix.lower() in {".mid", ".midi"})
        self.context = context
        self.songs: list[array] = []
        self.windows: list[tuple[int, int]] = []
        self.skipped_silent = 0
        for path in paths[:limit]:
            tokens = encode_midi(path)
            if not any(EVENTS[token][0] == "ON" for token in tokens):
                self.skipped_silent += 1
                continue

            song_index = len(self.songs)
            self.songs.append(array("H", tokens))
            target_count = len(tokens) - 1
            for start in range(0, target_count, stride):
                self.windows.append((song_index, start))
                if start + context >= target_count:
                    break

        if not self.windows:
            raise ValueError(f"Nenhuma música com notas encontrada em {folder}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        song_index, start = self.windows[index]
        tokens = self.songs[song_index][start:start + self.context + 1]
        return torch.tensor(tokens, dtype=torch.long)


def collate_windows(windows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Retorna entradas e alvos deslocados em uma posição, com PAD à direita."""
    padded = pad_sequence(windows, batch_first=True, padding_value=PAD)
    return padded[:, :-1].contiguous(), padded[:, 1:].contiguous()
