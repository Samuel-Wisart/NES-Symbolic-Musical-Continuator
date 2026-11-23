"""
test01_dataset_tests

Testes da representação MIDI-Like usada no projeto.
"""

### Imports
from step01_prepare_dataset import CHANNELS, SHIFT_MAX, TIME_RESOLUTION_HZ, TO_ID, VOCAB, decode, encode

### Constants
# Tokens auxiliares para treinamento autoregressivo.
SPECIAL_TOKENS = ("PAD", "BOS", "EOS")

# O padrão MIDI permite pitches de 0 a 127.
MIDI_PITCH_COUNT = 128

# Tokens especiais + TIME_SHIFTs + NOTE_ON/OFF para cada pitch e canal.
EXPECTED_VOCAB_SIZE = len(SPECIAL_TOKENS) + SHIFT_MAX + len(CHANNELS) * MIDI_PITCH_COUNT * 2

### Funções auxiliares
def to_tokens(notes):
    """Codifica notas e devolve os tokens como texto."""
    return [VOCAB[int(token_id)] for token_id in encode(notes)]

### Testes
def test_vocab_contract():
    """Confere tamanho, unicidade e tokens especiais do vocabulário."""
    assert len(VOCAB) == EXPECTED_VOCAB_SIZE
    assert len(VOCAB) == len(set(VOCAB))
    assert TO_ID["PAD"] == 0
    assert TO_ID["BOS"] == 1
    assert TO_ID["EOS"] == 2

def test_round_trip():
    """Confere se encode -> decode preserva notas que não sofrem quantização."""
    notes = [
        ("p1", 60, 0.0, 0.5),
        ("p2", 62, 0.25, 0.75),
        ("tr", 48, 1.0, 1.5),
        ("no", 40, 1.5, 1.75),
    ]
    assert decode(encode(notes)) == sorted(notes)

def test_quantization():
    """Confere a quantização e a duração mínima de uma nota."""
    off_grid = decode(encode([("p1", 60, 0.1, 0.5)]))
    assert off_grid == [("p1", 60, 6 / TIME_RESOLUTION_HZ, 0.5)]

    very_short = decode(encode([("p1", 60, 0.0, 0.001)]))
    assert very_short == [("p1", 60, 0.0, 1 / TIME_RESOLUTION_HZ)]

def test_long_time_shift():
    """Confere se intervalos maiores que SHIFT_MAX são divididos corretamente."""
    notes = [
        ("p1", 60, 0.0, 0.125),
        ("p1", 62, 5.0, 5.125),
    ]

    tokens = to_tokens(notes)
    shifts = [int(token.rsplit("_", 1)[1]) for token in tokens if token.startswith("TIME_SHIFT_")]

    assert max(shifts) <= SHIFT_MAX
    assert sum(shifts) == round(5.125 * TIME_RESOLUTION_HZ)
    assert decode(encode(notes)) == sorted(notes)

def main():
    tests = [test_vocab_contract, test_round_trip, test_quantization, test_long_time_shift]

    for test in tests:
        test()
        print(f"[ok] {test.__name__}")

    print(f"\n{len(tests)} testes passaram. Dataset ok.")

if __name__ == "__main__":
    main()