"""
representation

Ponte entre o pacote do Transformer e a representação oficial do grupo.

Tudo o que define os tokens (VOCAB, TO_ID, encode, decode, read_notes) vem de
scripts/step01_prepare_dataset.py. Aqui ficam apenas utilitários construídos em
cima dela: leitura dos .npz, corte do prompt, acompanhamento das notas abertas
e exportação para MIDI.
"""

### Imports
from collections import deque
from pathlib import Path
import sys

import numpy as np
import pretty_midi

### Caminhos
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
TOKENS_DIR = ROOT / "data" / "processed" / "nesmdb" / "tokens"
MIDI_DIR = ROOT / "data" / "raw" / "nesmdb_midi"

# O step01 é um script, não um pacote: entra no sys.path para ser importado.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from step01_prepare_dataset import (  # noqa: E402
    CHANNELS, SHIFT_MAX, SPLITS, TIME_RESOLUTION_HZ, TO_ID, VOCAB, decode, encode, read_notes,
)

### Tokens especiais
PAD = TO_ID["PAD"]
BOS = TO_ID["BOS"]
EOS = TO_ID["EOS"]
VOCAB_SIZE = len(VOCAB)

### Tabelas derivadas do VOCAB
# Não é um segundo vocabulário: são índices calculados a partir do VOCAB do step01
# para consultar tipos de token sem fazer parsing de texto durante a geração.

# SHIFT_IDS[s - 1] é o id de TIME_SHIFT_s.
SHIFT_IDS = np.array([TO_ID[f"TIME_SHIFT_{shift}"] for shift in range(1, SHIFT_MAX + 1)])

# ON_IDS[voz, pitch] e OFF_IDS[voz, pitch], com as vozes na ordem de CHANNELS.
ON_IDS = np.array([[TO_ID[f"NOTE_ON_{channel}_{pitch}"] for pitch in range(128)] for channel in CHANNELS])
OFF_IDS = np.array([[TO_ID[f"NOTE_OFF_{channel}_{pitch}"] for pitch in range(128)] for channel in CHANNELS])

# Para cada id: passos do TIME_SHIFT (0 se não for), voz e pitch da nota (-1 se não for).
SHIFT_STEPS = np.zeros(VOCAB_SIZE, dtype=np.int64)
SHIFT_STEPS[SHIFT_IDS] = np.arange(1, SHIFT_MAX + 1)

NOTE_VOICE = np.full(VOCAB_SIZE, -1, dtype=np.int64)
NOTE_PITCH = np.full(VOCAB_SIZE, -1, dtype=np.int64)
IS_ON = np.zeros(VOCAB_SIZE, dtype=bool)
IS_OFF = np.zeros(VOCAB_SIZE, dtype=bool)

for voice in range(len(CHANNELS)):
    for ids, flags in ((ON_IDS, IS_ON), (OFF_IDS, IS_OFF)):
        NOTE_VOICE[ids[voice]] = voice
        NOTE_PITCH[ids[voice]] = np.arange(128)
        flags[ids[voice]] = True

### Leitura dos dados
def token_files(split, root=TOKENS_DIR):
    """Lista os .npz de uma partição oficial, em ordem alfabética."""
    if split not in SPLITS:
        raise ValueError(f"Use uma partição oficial: {', '.join(SPLITS)}")

    return sorted((Path(root) / split).glob("*.npz"))

def load_ids(path):
    """Carrega os ids de uma música (.npz do step01 ou MIDI original)."""
    path = Path(path)

    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            return data["ids"].astype(np.int64)

    if path.suffix.lower() in (".mid", ".midi"):
        return encode(read_notes(path)).astype(np.int64)

    raise ValueError(f"Formato não suportado: {path}")

### Tempo
def seconds_to_steps(seconds):
    """Converte segundos para passos de TIME_RESOLUTION_HZ."""
    return round(seconds * TIME_RESOLUTION_HZ)

def duration_steps(ids):
    """Soma dos TIME_SHIFT de uma sequência, em passos."""
    return int(SHIFT_STEPS[np.asarray(ids, dtype=np.int64)].sum())

### Notas abertas
class NoteState:
    """Acompanha o tempo e as notas abertas ao percorrer os tokens em ordem.

    Guarda, para cada (voz, pitch), os passos de início das notas abertas, na
    mesma ordem FIFO que o decode() usa para parear NOTE_ON e NOTE_OFF. Mais de
    uma nota aberta na mesma voz acontece nos dados (notas muito curtas que o
    step01 estica para 1 passo), então isso não é tratado como erro.
    """

    def __init__(self, ids=()):
        self.step = 0
        self.open = {}
        self.extend(ids)

    def apply(self, token):
        token = int(token)

        if SHIFT_STEPS[token]:
            self.step += int(SHIFT_STEPS[token])
        elif IS_ON[token]:
            key = (int(NOTE_VOICE[token]), int(NOTE_PITCH[token]))
            self.open.setdefault(key, deque()).append(self.step)
        elif IS_OFF[token]:
            key = (int(NOTE_VOICE[token]), int(NOTE_PITCH[token]))
            starts = self.open.get(key)

            if starts:
                starts.popleft()
                if not starts:
                    del self.open[key]

    def extend(self, ids):
        for token in ids:
            self.apply(token)

    def sounding(self, voice):
        """Indica se a voz tem alguma nota aberta."""
        return any(key[0] == voice for key in self.open)

    def open_notes(self):
        """Lista (voz, pitch, início) de cada nota aberta, na ordem do decode()."""
        return [(voice, pitch, start) for (voice, pitch), starts in sorted(self.open.items()) for start in starts]

### Prompt e fechamento
def cut_prompt(ids, seconds):
    """Primeiros `seconds` segundos da música, sem EOS e com as notas abertas.

    O corte cai exatamente em seconds * TIME_RESOLUTION_HZ passos: se um
    TIME_SHIFT atravessa esse ponto, ele é dividido e só a primeira parte entra.
    Eventos no próprio passo do corte ficam para a continuação. A soma dos
    TIME_SHIFT do prompt é, portanto, igual à duração pedida.
    """
    ids = [int(token) for token in ids]
    stop = seconds_to_steps(seconds)

    if stop < 1:
        raise ValueError("O prompt precisa ter pelo menos um passo")

    prompt = []
    step = 0

    for token in ids:
        if token == EOS:
            break

        shift = int(SHIFT_STEPS[token])

        if shift and step + shift >= stop:
            prompt.append(int(SHIFT_IDS[stop - step - 1]))
            return np.asarray(prompt, dtype=np.int64)

        prompt.append(token)
        step += shift

    raise ValueError(f"A música tem {step / TIME_RESOLUTION_HZ:.2f} s, menos que os {seconds} s do prompt")

def close_open_notes(ids):
    """Fecha as notas abertas no fim da sequência e termina com EOS.

    Se alguma nota aberta começou no último passo, avança 1 passo antes de fechar,
    para não criar uma nota de duração zero (que o decode() descartaria).
    """
    ids = [int(token) for token in ids if int(token) != EOS]
    state = NoteState(ids)
    open_notes = state.open_notes()

    if any(start == state.step for *_, start in open_notes):
        ids.append(int(SHIFT_IDS[0]))

    ids.extend(int(OFF_IDS[voice, pitch]) for voice, pitch, _ in open_notes)
    ids.append(EOS)
    return np.asarray(ids, dtype=np.int64)

### MIDI
# Programas General MIDI só para a escuta em players comuns. O canal é o que importa:
# p1, p2 e tr ficam nos canais 0, 1 e 2 e o ruído no canal 9 (bateria), que é como
# o audio.py identifica as vozes. O pretty_midi atribui os canais pela ordem dos
# instrumentos, por isso CHANNELS precisa manter a ordem p1, p2, tr, no.
MIDI_PROGRAMS = {"p1": 80, "p2": 81, "tr": 38, "no": 0}
MIDI_VELOCITY = 100

def write_midi(ids, path):
    """Converte tokens -> decode() -> notas -> MIDI, com uma trilha por voz."""
    midi = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120)
    instruments = {}

    for channel in CHANNELS:
        instrument = pretty_midi.Instrument(MIDI_PROGRAMS[channel], is_drum=channel == "no", name=channel)
        instruments[channel] = instrument
        midi.instruments.append(instrument)

    for channel, pitch, start, end in decode(ids):
        instruments[channel].notes.append(pretty_midi.Note(MIDI_VELOCITY, pitch, start, end))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
