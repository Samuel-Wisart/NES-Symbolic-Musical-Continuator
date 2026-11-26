"""
test04_markov_tests

Valida a integração do Markov de ordem 1 com o protocolo compartilhado.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "run" / "markov"))

import protocol
from generate_markov import MATRIX_PATH, sample_markov
from nes_continuation.representation import NoteState, cut_prompt, load_ids, seconds_to_steps
from nes_continuation.sampling import allowed_tokens


def load_model():
    data = np.load(MATRIX_PATH)
    return data["probs"], data["global_probs"]

def generate(index, probs, global_probs):
    path = protocol.prompt_paths()[index]
    ids = load_ids(path)
    prompt = cut_prompt(ids, protocol.PROMPT_SECONDS)
    result = sample_markov(
        probs,
        global_probs,
        prompt,
        seed=protocol.generation_seed(index),
        temperature=protocol.TEMPERATURE,
        top_k=protocol.TOP_K,
        min_seconds=protocol.MIN_GEN_SECONDS,
        max_seconds=protocol.GEN_SECONDS,
    )
    return prompt, result

def test_generation():
    probs, global_probs = load_model()
    minimum = seconds_to_steps(protocol.MIN_GEN_SECONDS)
    maximum = seconds_to_steps(protocol.GEN_SECONDS)

    for index in range(3):
        prompt, result = generate(index, probs, global_probs)
        assert np.array_equal(result["ids"][:result["prompt_length"]], prompt)
        assert result["generated_steps"] == maximum

        state = NoteState(prompt)
        boundary = state.step
        sampled = result["ids"][
            result["prompt_length"]:result["prompt_length"] + result["generated_tokens"]
        ]
        for token in sampled:
            elapsed = state.step - boundary
            assert allowed_tokens(state, elapsed, minimum, maximum)[token]
            state.apply(token)

        assert not NoteState(result["ids"]).open

def test_reproducibility():
    probs, global_probs = load_model()
    prompt, first = generate(0, probs, global_probs)
    second = sample_markov(
        probs,
        global_probs,
        prompt,
        seed=protocol.generation_seed(0),
        temperature=protocol.TEMPERATURE,
        top_k=protocol.TOP_K,
        min_seconds=protocol.MIN_GEN_SECONDS,
        max_seconds=protocol.GEN_SECONDS,
    )
    assert np.array_equal(first["ids"], second["ids"])

def main():
    tests = [test_generation, test_reproducibility]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    print(f"\n{len(tests)} testes passaram. Markov integrado ao protocolo.")


if __name__ == "__main__":
    main()
