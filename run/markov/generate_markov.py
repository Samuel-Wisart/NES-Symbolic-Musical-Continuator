"""
generate_markov

Gera continuações com Markov de ordem 1 usando o protocolo compartilhado.
"""

### Imports
from pathlib import Path
import json
import sys

import numpy as np

### Caminhos e imports locais
ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from nes_continuation.representation import (
    SHIFT_IDS,
    TIME_RESOLUTION_HZ,
    VOCAB_SIZE,
    NoteState,
    close_open_notes,
    cut_prompt,
    load_ids,
    seconds_to_steps,
)
from nes_continuation.sampling import allowed_tokens, save_generation
import protocol

### Arquivos de entrada e saída
PROCESSED = ROOT / "data" / "processed" / "nesmdb"
MATRIX_PATH = PROCESSED / "markov_order1.npz"
OUTPUT_DIR = protocol.GENERATED_DIR / "markov"


### Distribuição de amostragem
def sample_distribution(probabilities, mask, temperature, top_k, rng):
    """Amostra somente entre tokens válidos com probabilidade positiva."""
    if temperature <= 0:
        raise ValueError("temperature deve ser > 0")

    valid_ids = np.flatnonzero(mask & (probabilities > 0))
    if not len(valid_ids):
        return None

    weights = probabilities[valid_ids].astype(np.float64)
    if top_k and len(valid_ids) > top_k:
        selected = np.argpartition(weights, -top_k)[-top_k:]
        valid_ids = valid_ids[selected]
        weights = weights[selected]

    with np.errstate(divide="ignore"):
        logits = np.log(weights) / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return int(rng.choice(valid_ids, p=weights))

### Geração
def sample_markov(
    probs,
    global_probs,
    prefix,
    *,
    seed,
    temperature,
    top_k,
    min_seconds,
    max_seconds,
):
    """Continua `prefix` com P(x_t | x_{t-1}) até o limite temporal do protocolo."""
    minimum = seconds_to_steps(min_seconds)
    maximum = seconds_to_steps(max_seconds)
    if maximum < 1 or minimum > maximum:
        raise ValueError("Use 0 <= min_seconds <= max_seconds e máximo de pelo menos um passo.")

    sequence = [int(token) for token in prefix]
    if not sequence:
        raise ValueError("O prompt não pode ser vazio.")

    prompt_length = len(sequence)
    state = NoteState(sequence)
    boundary = state.step
    elapsed = 0
    rng = np.random.default_rng(seed)

    while elapsed < maximum:
        mask = allowed_tokens(state, elapsed, minimum, maximum)
        token = sample_distribution(probs[sequence[-1]], mask, temperature, top_k, rng)

        if token is None:
            token = sample_distribution(global_probs, mask, temperature, top_k, rng)

        if token is None:
            valid_shifts = SHIFT_IDS[mask[SHIFT_IDS]]
            if not len(valid_shifts):
                raise RuntimeError("Nenhum token válido disponível para continuar a geração.")
            token = int(valid_shifts[-1])

        sequence.append(token)
        state.apply(token)
        elapsed = state.step - boundary

    generated_tokens = len(sequence) - prompt_length
    return {
        "ids": close_open_notes(sequence),
        "prompt_length": prompt_length,
        "generated_tokens": generated_tokens,
        "generated_steps": elapsed,
        "stop_reason": "limite de duração",
    }

### Protocolo
def settings():
    names = (
        "PROMPT_SECONDS",
        "GEN_SECONDS",
        "MIN_GEN_SECONDS",
        "N_PROMPTS",
        "SEED",
        "TEMPERATURE",
        "TOP_K",
    )
    return {name.lower(): getattr(protocol, name) for name in names}

### Execução
def main():
    data = np.load(MATRIX_PATH)
    probs = data["probs"]
    global_probs = data["global_probs"]

    if int(data["vocab_size"]) != VOCAB_SIZE:
        raise ValueError(
            f"Modelo treinado com {int(data['vocab_size'])} tokens, "
            f"mas a representação atual possui {VOCAB_SIZE}."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for index, path in enumerate(protocol.prompt_paths()):
        ids = load_ids(path)
        prompt = cut_prompt(ids, protocol.PROMPT_SECONDS)
        seed = protocol.generation_seed(index)
        result = sample_markov(
            probs,
            global_probs,
            prompt,
            seed=seed,
            temperature=protocol.TEMPERATURE,
            top_k=protocol.TOP_K,
            min_seconds=protocol.MIN_GEN_SECONDS,
            max_seconds=protocol.GEN_SECONDS,
        )
        wav = index < protocol.N_WAV
        save_generation(result, OUTPUT_DIR / f"{path.stem}.mid", wav=wav)
        rows.append({
            "name": path.stem,
            "seed": seed,
            "prompt_tokens": result["prompt_length"],
            "generated_tokens": result["generated_tokens"],
            "generated_seconds": result["generated_steps"] / TIME_RESOLUTION_HZ,
            "stop_reason": result["stop_reason"],
            "wav": wav,
        })
        print(
            f"[{index + 1}/{protocol.N_PROMPTS}] {path.stem}: "
            f"{result['generated_tokens']} tokens | "
            f"{result['generated_steps'] / TIME_RESOLUTION_HZ:.2f} s"
        )

    summary = {"model": "markov", "order": 1, "protocol": settings(), "songs": rows}
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nResultados: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
