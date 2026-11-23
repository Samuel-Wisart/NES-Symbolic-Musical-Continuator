"""
test02_transformer_tests

Testes da adaptação do Transformer à representação do step01.
"""

### Imports
from collections import Counter
from itertools import product
import random
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mido
import numpy as np
import torch

from nes_continuation.data import NESWindows, ScoringWindows, collate_scoring, collate_windows
from nes_continuation.model import ModelConfig, NESTransformer
from nes_continuation.representation import (
    BOS, CHANNELS, EOS, MIDI_DIR, PAD, SHIFT_STEPS, TIME_RESOLUTION_HZ, TO_ID, TOKENS_DIR, VOCAB,
    NoteState, close_open_notes, cut_prompt, decode, duration_steps, encode, load_ids, read_notes,
    reference_continuation, split_at_seconds, token_files, write_midi,
)
from nes_continuation.sampling import allowed_tokens, sample
from step01_prepare_dataset import quantize_interval
import protocol

### Constantes
# Poucas músicas por teste, para a suíte rodar em segundos.
SONGS_PER_SPLIT = 3
PROMPT_SECONDS = (4, 2.5, 1 / TIME_RESOLUTION_HZ)

# Canal MIDI esperado para cada voz (o audio.py depende disso).
EXPECTED_MIDI_CHANNEL = {"p1": 0, "p2": 1, "tr": 2, "no": 9}

### Funções auxiliares
def first_songs(split, count=SONGS_PER_SPLIT):
    """Primeiros .npz de uma partição."""
    return token_files(split)[:count]

def long_test_songs(min_seconds, count=SONGS_PER_SPLIT):
    """Primeiras músicas de teste com pelo menos `min_seconds` segundos."""
    songs = [path for path in token_files("test") if duration_steps(load_ids(path)) >= min_seconds * TIME_RESOLUTION_HZ]
    return songs[:count]

def token_name(token):
    return VOCAB[int(token)]

def tiny_model(context=64):
    torch.manual_seed(0)
    return NESTransformer(ModelConfig(context=context, width=32, heads=4, layers=2, dropout=0.0))

class AdversarialModel(torch.nn.Module):
    """Modelo falso que prefere justamente PAD, BOS, EOS e notas, para forçar a máscara."""

    def __init__(self, context=64):
        super().__init__()
        self.config = ModelConfig(context=context, width=4, heads=1, layers=1)
        # PAD, BOS e EOS muito acima das notas, e notas muito acima dos TIME_SHIFT.
        self.bias = torch.tensor([
            0.0 if token.startswith("TIME_SHIFT_") else 6.0 if token.startswith("NOTE_") else 20.0
            for token in VOCAB
        ])
        self.generator = torch.Generator().manual_seed(0)

    def forward(self, tokens):
        noise = torch.randn(*tokens.shape, len(VOCAB), generator=self.generator)
        return noise + self.bias

### Testes
def test_npz_matches_encode():
    """Confere se o .npz é igual a encode(read_notes(mid)) do MIDI original."""
    for split in ("train", "valid", "test"):
        for path in first_songs(split):
            midi = MIDI_DIR / split / f"{path.stem}.mid"
            expected = encode(read_notes(midi)).astype(np.int64)
            assert np.array_equal(load_ids(path), expected), path.name

def test_prompt_cut_preserves_seconds():
    """Confere se o prompt dura exatamente N segundos, sem EOS e com a música como prefixo."""
    for path in long_test_songs(5):
        ids = load_ids(path)

        for seconds in PROMPT_SECONDS:
            prompt = cut_prompt(ids, seconds)
            steps = round(seconds * TIME_RESOLUTION_HZ)

            assert duration_steps(prompt) == steps, (path.name, seconds)
            assert prompt[0] == BOS and EOS not in prompt.tolist()
            assert token_name(prompt[-1]).startswith("TIME_SHIFT_")
            # Só o último TIME_SHIFT pode ter sido dividido.
            assert np.array_equal(prompt[:-1], ids[:len(prompt) - 1])
            assert SHIFT_STEPS[prompt[-1]] <= SHIFT_STEPS[ids[len(prompt) - 1]]

            # As notas abertas no prompt são as que começam antes do corte e terminam
            # nele ou depois (um NOTE_OFF no passo do corte fica para a continuação).
            crossing = [
                (CHANNELS.index(channel), pitch)
                for channel, pitch, start, end in read_notes(MIDI_DIR / "test" / f"{path.stem}.mid")
                for start_step, end_step in [quantize_interval(start, end)]
                if start_step < steps <= end_step
            ]
            assert sorted((voice, pitch) for voice, pitch, _ in NoteState(prompt).open_notes()) == sorted(crossing)

def test_prompt_cut_splits_time_shift():
    """Confere a divisão de um TIME_SHIFT longo no ponto de corte."""
    ids = encode([("p1", 60, 0.0, 0.5), ("p1", 62, 5.0, 5.5)])
    prompt = cut_prompt(ids, 1.0)

    assert [token_name(token) for token in prompt] == ["BOS", "NOTE_ON_p1_60", "TIME_SHIFT_32", "NOTE_OFF_p1_60", "TIME_SHIFT_32"]

    # Corte no meio da nota: ela continua aberta no prompt.
    prompt = cut_prompt(ids, 0.25)
    assert [token_name(token) for token in prompt] == ["BOS", "NOTE_ON_p1_60", "TIME_SHIFT_16"]
    assert NoteState(prompt).open_notes() == [(0, 60, 0)]

    # Eventos exatamente no passo do corte ficam para a continuação.
    prompt = cut_prompt(ids, 0.5)
    assert [token_name(token) for token in prompt] == ["BOS", "NOTE_ON_p1_60", "TIME_SHIFT_32"]

def test_split_and_reference():
    """Confere se início + resto preserva as notas e se a referência começa pelo mesmo prompt."""
    for path in long_test_songs(5):
        ids = load_ids(path)
        head, tail = split_at_seconds(ids, 4)

        assert np.array_equal(head, cut_prompt(ids, 4))
        assert decode(np.concatenate([head, tail])) == decode(ids)
        assert duration_steps(head) + duration_steps(tail) == duration_steps(ids)

        reference, prompt_length = reference_continuation(ids, 4, 12)
        continuation = duration_steps(reference[prompt_length:])

        assert np.array_equal(reference[:prompt_length], head)
        assert continuation <= 12 * TIME_RESOLUTION_HZ + 1
        assert continuation >= min(12 * TIME_RESOLUTION_HZ, duration_steps(tail))
        assert reference[-1] == EOS and not NoteState(reference).open

def test_scoring_windows():
    """Confere se as janelas de avaliação pontuam cada alvo uma vez, com o histórico mínimo."""
    context, stride = 64, 32
    dataset = ScoringWindows(TOKENS_DIR, "test", context=context, stride=stride, limit=8)
    scored = Counter()

    for index in range(len(dataset)):
        song, start, skip = dataset.windows[index]
        _, targets = collate_scoring([dataset[index]])

        for offset in np.flatnonzero(targets[0].numpy() != PAD):
            assert offset >= skip
            # Alvo `start + offset`: histórico de offset + 1 tokens nesta janela.
            assert start == 0 or offset + 1 > context - stride
            scored[song, start + offset] += 1

    assert sum(scored.values()) == dataset.target_count
    assert set(scored.values()) == {1}

def test_protocol_prompt_list():
    """Confere a lista fixa de prompts e a contagem de tokens do teste no protocol.py."""
    files = token_files("test")
    durations = {path.stem: duration_steps(load_ids(path)) for path in files}
    eligible = sorted(name for name, steps in durations.items() if steps >= (protocol.PROMPT_SECONDS + 2) * TIME_RESOLUTION_HZ)

    assert len(files) == protocol.TEST_SONGS
    assert sum(len(load_ids(path)) - 1 for path in files) == protocol.TEST_TARGET_TOKENS
    assert list(protocol.PROMPT_SONGS) == random.Random(protocol.SEED).sample(eligible, protocol.N_PROMPTS)
    assert all(path.exists() for path in protocol.prompt_paths())

def test_close_open_notes():
    """Confere se as notas abertas são fechadas sem criar notas de duração zero."""
    prompt = cut_prompt(encode([("p1", 60, 0.0, 0.5), ("tr", 40, 0.0, 1.0)]), 0.25)
    closed = close_open_notes(prompt)

    assert [token_name(token) for token in closed[len(prompt):]] == ["NOTE_OFF_p1_60", "NOTE_OFF_tr_40", "EOS"]
    assert not NoteState(closed).open

    # Nota aberta no último passo: avança 1 passo antes de fechar.
    closed = close_open_notes([BOS, TO_ID["TIME_SHIFT_4"], TO_ID["NOTE_ON_no_5"]])
    assert [token_name(token) for token in closed[3:]] == ["TIME_SHIFT_1", "NOTE_OFF_no_5", "EOS"]

def test_midi_round_trip():
    """Confere tokens -> decode -> MIDI -> read_notes -> encode e os canais do MIDI."""
    with tempfile.TemporaryDirectory() as folder:
        for index, path in enumerate(first_songs("test", 6)):
            ids = load_ids(path)
            midi_path = Path(folder) / f"{index}.mid"
            write_midi(ids, midi_path)

            assert np.array_equal(encode(read_notes(midi_path)), ids), path.name

            midi = mido.MidiFile(midi_path)
            for track in midi.tracks[1:]:
                channels = {message.channel for message in track if message.type in ("note_on", "note_off")}
                assert channels <= {EXPECTED_MIDI_CHANNEL[track.name]}, (track.name, channels)

def test_windows_cover_targets():
    """Confere se, com stride = contexto, cada token depois do BOS é alvo uma única vez."""
    context = 64
    dataset = NESWindows(TOKENS_DIR, "test", context=context, stride=context, limit=8)
    inputs, targets = collate_windows([dataset[index] for index in range(len(dataset))])

    assert dataset.target_count == sum(len(song) - 1 for song in dataset.songs)
    assert int(targets.ne(PAD).sum()) == dataset.target_count
    assert inputs.shape[1] <= context

def test_model_forward():
    """Confere o forward com o vocabulário inteiro de 1155 tokens."""
    model = tiny_model()
    dataset = NESWindows(TOKENS_DIR, "valid", context=64, limit=2)
    inputs, targets = collate_windows([dataset[0], dataset[len(dataset) - 1]])
    logits = model(inputs)

    assert len(VOCAB) == 1155
    assert model.token_embedding.num_embeddings == len(VOCAB)
    assert logits.shape == (*inputs.shape, len(VOCAB))
    assert torch.isfinite(logits).all()

def test_sampled_tokens_obey_mask():
    """Gera com modelos aleatórios e confere cada token contra as regras e a máscara."""
    min_seconds, max_seconds = 0.5, 3.0
    minimum, maximum = round(min_seconds * TIME_RESOLUTION_HZ), round(max_seconds * TIME_RESOLUTION_HZ)
    models = (tiny_model(), AdversarialModel())
    checked = Counter()

    for path, model, seed in product(long_test_songs(4, 2), models, range(3)):
        prompt = cut_prompt(load_ids(path), 2)
        result = sample(
            model, prompt, torch.device("cpu"),
            seed=seed, top_k=0, temperature=1.5, min_seconds=min_seconds, max_seconds=max_seconds,
        )
        ids = result["ids"]
        start = result["prompt_length"]
        generated = ids[start:start + result["generated_tokens"]]
        assert np.array_equal(ids[:len(prompt)], prompt)

        state = NoteState(prompt)
        boundary = state.step

        # O EOS amostrado não fica na fatia `generated`: entra como último token conferido.
        sampled = list(generated) + ([EOS] if result["stop_reason"] == "EOS" else [])

        for token in sampled:
            name = token_name(token)
            elapsed = state.step - boundary
            assert allowed_tokens(state, elapsed, minimum, maximum)[token], name

            # As mesmas regras, verificadas pelo significado de cada token.
            assert token not in (PAD, BOS), name

            if name == "EOS":
                assert elapsed >= minimum
            elif name.startswith("TIME_SHIFT_"):
                assert elapsed + int(name.rsplit("_", 1)[1]) <= maximum
            else:
                _, kind, channel, pitch = name.split("_")
                voice = CHANNELS.index(channel)

                if kind == "ON":
                    assert not any(key[0] == voice for key in state.open), name
                else:
                    assert state.open.get((voice, int(pitch))), name
                    assert state.open[(voice, int(pitch))][0] < state.step, name

            state.apply(token)
            checked[name.split("_")[0] if not name.startswith("NOTE_") else "_".join(name.split("_")[:2])] += 1

        # Ao final, só vêm os NOTE_OFF de fechamento (com no máximo um TIME_SHIFT_1),
        # nenhuma nota fica aberta e a sequência termina com EOS.
        closing = [token_name(token) for token in ids[start + len(generated):-1]]
        assert all(name.startswith("NOTE_OFF_") for name in closing if name != "TIME_SHIFT_1")
        assert closing.count("TIME_SHIFT_1") <= 1
        assert ids[-1] == EOS and EOS not in ids[:-1].tolist()
        assert not NoteState(ids).open
        assert duration_steps(generated) <= maximum

    # O teste só vale se as regras foram de fato exercitadas.
    assert min(checked[kind] for kind in ("TIME", "NOTE_ON", "NOTE_OFF", "EOS")) > 0, checked

def main():
    tests = [
        test_npz_matches_encode,
        test_prompt_cut_preserves_seconds,
        test_prompt_cut_splits_time_shift,
        test_split_and_reference,
        test_close_open_notes,
        test_midi_round_trip,
        test_windows_cover_targets,
        test_scoring_windows,
        test_protocol_prompt_list,
        test_model_forward,
        test_sampled_tokens_obey_mask,
    ]

    for test in tests:
        test()
        print(f"[ok] {test.__name__}")

    print(f"\n{len(tests)} testes passaram. Transformer ok.")

if __name__ == "__main__":
    main()
