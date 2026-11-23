"""Continuação autoregressiva condicionada por um prefixo NES-MDB."""

import time
from pathlib import Path

import torch

from .audio import render_wav
from .tokenizer import BOS, EOS, EVENTS, FRAME_RATE, PAD, VOICES, decode_midi, encode_prompt
from .training import load_checkpoint, restore_model


class MusicalState:
    def __init__(self) -> None:
        self.frame = 0
        self.active = [False] * len(VOICES)
        self.last_voice = -1

    def apply(self, token: int) -> None:
        kind, voice, value = EVENTS[token]
        if kind == "SHIFT":
            self.frame += value
            self.last_voice = -1
        elif kind in ("ON", "OFF"):
            self.active[voice] = kind == "ON"
            self.last_voice = voice

    def allowed(self, token: int, elapsed: int, minimum: int, remaining: int) -> bool:
        kind, voice, value = EVENTS[token]
        if token in (BOS, PAD):
            return False
        if token == EOS:
            return elapsed >= minimum
        if kind == "SHIFT":
            return value <= remaining
        return voice > self.last_voice and (kind == "ON" or self.active[voice])


@torch.inference_mode()
def sample(model, prefix: list[int], args, device: torch.device) -> tuple[list[int], int, str]:
    model.eval()
    generator = torch.Generator().manual_seed(args.seed)
    sequence = prefix.copy()
    state = MusicalState()
    for token in prefix:
        state.apply(token)
    boundary = state.frame
    minimum = round(args.min_seconds * FRAME_RATE)
    maximum = round(args.max_seconds * FRAME_RATE)
    elapsed = 0
    reason = "limite de tokens"
    for _ in range(args.max_new_tokens):
        inputs = torch.tensor([sequence[-model.config.context:]], dtype=torch.long, device=device)
        logits = model(inputs)[0, -1].float().cpu()
        mask = torch.tensor([
            state.allowed(token, elapsed, minimum, maximum - elapsed)
            for token in range(len(EVENTS))
        ], dtype=torch.bool)
        logits = logits.masked_fill(~mask, -torch.inf)
        if args.top_k:
            values, ids = logits.topk(min(args.top_k, len(EVENTS)))
        else:
            values = logits
            ids = None
        weights = ((values - values.max()) / args.temperature).softmax(dim=-1)
        choice = torch.multinomial(weights, 1, generator=generator)
        token = int(choice if ids is None else ids[choice])
        sequence.append(token)
        if token == EOS:
            reason = "EOS"
            break
        state.apply(token)
        elapsed = state.frame - boundary
        if elapsed >= maximum:
            reason = "limite de duração"
            break
    return sequence, elapsed, reason


def generate(args, device: torch.device) -> None:
    if round(args.max_seconds * FRAME_RATE) < 1 or args.min_seconds > args.max_seconds:
        raise ValueError("Use 0 <= --min-seconds <= --max-seconds e máximo de pelo menos um quadro.")
    output: Path = args.output
    if output.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError("--output precisa terminar em .mid ou .midi")
    if output.resolve() in {args.prompt.resolve(), args.checkpoint.resolve()}:
        raise ValueError("Escolha uma saída diferente do prompt e do checkpoint.")
    wav_path = output.with_suffix(".wav")
    if args.wav and wav_path.resolve() in {args.prompt.resolve(), args.checkpoint.resolve()}:
        raise ValueError("O caminho do WAV coincide com uma entrada.")
    prefix = encode_prompt(args.prompt, args.prompt_seconds)
    checkpoint = load_checkpoint(args.checkpoint)
    model = restore_model(checkpoint, device)
    print(f"{device} | checkpoint da época {checkpoint['epoch']} | {len(prefix)} tokens no prefixo", flush=True)
    start = time.monotonic()
    sequence, elapsed, reason = sample(model, prefix, args, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    decode_midi(sequence, output)
    suffix_count = len(sequence) - len(prefix)
    print(
        f"{suffix_count} tokens novos | +{elapsed / FRAME_RATE:.2f} s musicais | "
        f"{reason} | geração em {time.monotonic() - start:.1f} s",
        flush=True,
    )
    print(f"MIDI: {output}")
    if args.wav:
        render_wav(output, wav_path)
        print(f"WAV: {wav_path}")
