"""
sampling

Continuação autoregressiva condicionada por um prefixo de tokens do NES-MDB.
"""

### Imports
import time
from pathlib import Path

import numpy as np
import torch

from .audio import render_wav
from .representation import (
    CHANNELS, EOS, IS_ON, OFF_IDS, ON_IDS, SHIFT_IDS, SHIFT_MAX, TIME_RESOLUTION_HZ, VOCAB_SIZE,
    NoteState, close_open_notes, cut_prompt, load_ids, seconds_to_steps, write_midi,
)
from .training import load_checkpoint, restore_model


### Máscara de tokens válidos
def allowed_tokens(state: NoteState, elapsed: int, minimum: int, maximum: int) -> np.ndarray:
    """Retorna a máscara dos tokens válidos no estado atual.

    Proíbe PAD e BOS, limita TIME_SHIFT ao tempo restante e só permite NOTE_ON
    e NOTE_OFF compatíveis com as notas abertas. EOS fica bloqueado até `minimum`.
    """
    mask = np.zeros(VOCAB_SIZE, dtype=bool)
    mask[SHIFT_IDS[:max(0, min(maximum - elapsed, SHIFT_MAX))]] = True
    mask[EOS] = elapsed >= minimum
    for voice in range(len(CHANNELS)):
        if not state.sounding(voice):
            mask[ON_IDS[voice]] = True
    for (voice, pitch), starts in state.open.items():
        if starts[0] < state.step:
            mask[OFF_IDS[voice, pitch]] = True
    return mask

### Amostragem
@torch.inference_mode()
def sample(
    model,
    prefix,
    device: torch.device,
    *,
    seed: int = 42,
    temperature: float = 1.0,
    top_k: int = 32,
    min_seconds: float = 2.0,
    max_seconds: float = 12.0,
    max_new_tokens: int = 0,
) -> dict:
    """Continua `prefix` até EOS ou até atingir um limite configurado.

    Retorna prompt, tokens amostrados, fechamento de notas e EOS. A faixa indicada
    por `generated_tokens` não inclui EOS.
    """
    minimum = seconds_to_steps(min_seconds)
    maximum = seconds_to_steps(max_seconds)
    if maximum < 1 or minimum > maximum:
        raise ValueError("Use 0 <= min_seconds <= max_seconds e máximo de pelo menos um passo.")

    model.eval()
    generator = torch.Generator().manual_seed(seed)
    sequence = [int(token) for token in prefix]
    prompt_length = len(sequence)
    state = NoteState(sequence)
    boundary = state.step
    elapsed = 0
    reason = "limite de tokens"
    while not max_new_tokens or len(sequence) - prompt_length < max_new_tokens:
        inputs = torch.tensor([sequence[-model.config.context:]], dtype=torch.long, device=device)
        logits = model(inputs)[0, -1].float().cpu()
        mask = torch.from_numpy(allowed_tokens(state, elapsed, minimum, maximum))
        logits = logits.masked_fill(~mask, -torch.inf)
        if top_k:
            values, ids = logits.topk(min(top_k, VOCAB_SIZE))
        else:
            values, ids = logits, None
        weights = ((values - values.max()) / temperature).softmax(dim=-1)
        choice = torch.multinomial(weights, 1, generator=generator)
        token = int(choice if ids is None else ids[choice])
        if token == EOS:
            reason = "EOS"
            break
        sequence.append(token)
        state.apply(token)
        elapsed = state.step - boundary
        if elapsed >= maximum:
            reason = "limite de duração"
            break

    generated = len(sequence) - prompt_length
    return {
        "ids": close_open_notes(sequence),
        "prompt_length": prompt_length,
        "generated_tokens": generated,
        "generated_steps": elapsed,
        "stop_reason": reason,
    }

### Salvamento
def save_generation(result: dict, midi_path: Path, wav: bool = False) -> None:
    """Salva <nome>.npz (ids + prompt_length), <nome>.mid e, se pedido, <nome>.wav."""
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        midi_path.with_suffix(".npz"),
        ids=np.asarray(result["ids"], dtype=np.uint16),
        prompt_length=result["prompt_length"],
    )
    write_midi(result["ids"], midi_path)
    if wav:
        render_wav(midi_path, midi_path.with_suffix(".wav"))

### Geração pela CLI
def generate(args, device: torch.device) -> None:
    output: Path = args.output
    if output.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError("--output precisa terminar em .mid ou .midi")
    inputs = {args.prompt.resolve(), args.checkpoint.resolve()}
    if any(output.with_suffix(suffix).resolve() in inputs for suffix in (output.suffix, ".npz", ".wav")):
        raise ValueError("Escolha uma saída diferente do prompt e do checkpoint.")

    ids = load_ids(args.prompt)
    if args.prompt_seconds is None:
        prefix = ids[:-1] if ids[-1] == EOS else ids
    else:
        prefix = cut_prompt(ids, args.prompt_seconds)
    if not IS_ON[prefix].any():
        raise ValueError("O trecho inicial deve conter pelo menos um NOTE_ON")

    checkpoint = load_checkpoint(args.checkpoint)
    model = restore_model(checkpoint, device)
    print(f"{device} | checkpoint da época {checkpoint['epoch']} | {len(prefix)} tokens no prefixo", flush=True)
    start = time.monotonic()
    result = sample(
        model, prefix, device,
        seed=args.seed, temperature=args.temperature, top_k=args.top_k,
        min_seconds=args.min_seconds, max_seconds=args.max_seconds, max_new_tokens=args.max_new_tokens,
    )
    save_generation(result, output, wav=args.wav)
    print(
        f"{result['generated_tokens']} tokens novos | +{result['generated_steps'] / TIME_RESOLUTION_HZ:.2f} s musicais | "
        f"{result['stop_reason']} | geração em {time.monotonic() - start:.1f} s",
        flush=True,
    )
    print(f"MIDI: {output}")
    if args.wav:
        print(f"WAV: {output.with_suffix('.wav')}")
