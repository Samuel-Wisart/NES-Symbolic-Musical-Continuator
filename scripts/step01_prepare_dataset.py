"""
step01_prepare_dataset

Prepara o NES-MDB e converte cada música MIDI para tokens MIDI-Like.
"""

### Imports
from collections import defaultdict, deque
from pathlib import Path
import csv, json, shutil, tarfile
import numpy as np
import pretty_midi

### Caminhos
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed" / "nesmdb"
ARCHIVE = RAW_DIR / "nesmdb_midi.tar.gz"
MIDI_DIR = RAW_DIR / "nesmdb_midi"

### Dataset
SPLITS = ("train", "valid", "test")

# Quatro vozes do NES-MDB: dois pulsos, triângulo e ruído.
CHANNELS = ("p1", "p2", "tr", "no")
CHANNEL_ORDER = {channel: index for index, channel in enumerate(CHANNELS)}

# Recomendado pelos autores caso algum MIDI falhe ao carregar.
# No README do dataset eles informam que:
# If loading MIDI file fails, try pretty_midi.pretty_midi.MAX_TICK = 1e10
pretty_midi.pretty_midi.MAX_TICK = 1e10

### Representação MIDI-Like
TIME_RESOLUTION_HZ = 64
SHIFT_MAX = 128

### Vocabulário
# PAD auxilia batches. BOS e EOS delimitam cada música.
VOCAB = (
    ["PAD", "BOS", "EOS"]
    + [f"TIME_SHIFT_{shift}" for shift in range(1, SHIFT_MAX + 1)]
    + [
        f"{event}_{channel}_{pitch}"
        for event in ("NOTE_ON", "NOTE_OFF")
        for channel in CHANNELS
        for pitch in range(128)
    ]
)

TO_ID = {token: index for index, token in enumerate(VOCAB)}

### Preparação do dataset
def extract_dataset():
    """Extrai o NES-MDB em data/raw."""
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {ARCHIVE}")

    shutil.rmtree(MIDI_DIR, ignore_errors=True)

    with tarfile.open(ARCHIVE, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".mid"):
                continue

            split = next((split for split in SPLITS if split in Path(member.name).parts), None)
            if split is None:
                continue

            destination = MIDI_DIR / split / Path(member.name).name
            destination.parent.mkdir(parents=True, exist_ok=True)

            with tar.extractfile(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)

def get_midi_files():
    """Lista os MIDI de treino, validação e teste, ja splitados previamente."""
    return {split: sorted((MIDI_DIR / split).glob("*.mid")) for split in SPLITS}


### Leitura e representação
def read_notes(path):
    """Lê um MIDI como (canal, pitch, início, fim)."""
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = []

    for instrument in midi.instruments:
        channel = instrument.name.lower().strip()

        if channel not in CHANNEL_ORDER:
            raise ValueError(f"Canal não reconhecido: {instrument.name!r}")

        for note in instrument.notes:
            if note.end > note.start:
                notes.append((channel, note.pitch, note.start, note.end))

    return notes

def quantize_interval(start, end, rate_hz=TIME_RESOLUTION_HZ):
    """Converte tempos em segundos para passos discretos."""
    start_step = max(0, round(start * rate_hz))
    end_step = max(start_step + 1, round(end * rate_hz))
    return start_step, end_step

def encode(notes, rate_hz=TIME_RESOLUTION_HZ):
    """Converte notas para uma sequência de IDs de tokens."""
    events = []

    for channel, pitch, start, end in notes:
        start_step, end_step = quantize_interval(start, end, rate_hz)
        channel_order = CHANNEL_ORDER[channel]

        events.append((start_step, channel_order, 1, pitch, TO_ID[f"NOTE_ON_{channel}_{pitch}"]))
        events.append((end_step, channel_order, 0, pitch, TO_ID[f"NOTE_OFF_{channel}_{pitch}"]))

    events.sort()

    ids = [TO_ID["BOS"]]
    current_step = 0

    for step, *_, token_id in events:
        while current_step < step:
            shift = min(step - current_step, SHIFT_MAX)
            ids.append(TO_ID[f"TIME_SHIFT_{shift}"])
            current_step += shift

        ids.append(token_id)

    ids.append(TO_ID["EOS"])
    return np.asarray(ids, dtype=np.uint16)


def decode(ids, rate_hz=TIME_RESOLUTION_HZ):
    """Converte uma sequência de tokens novamente para notas."""
    current_step = 0
    open_notes = defaultdict(deque)
    notes = []

    for token_id in ids:
        token = VOCAB[int(token_id)]

        if token.startswith("TIME_SHIFT_"):
            current_step += int(token.rsplit("_", 1)[1])
            continue

        if not token.startswith("NOTE_"):
            continue

        _, state, channel, pitch = token.split("_")
        pitch = int(pitch)
        key = (channel, pitch)

        if state == "ON":
            open_notes[key].append(current_step)
        elif open_notes[key]:
            start_step = open_notes[key].popleft()
            notes.append((channel, pitch, start_step / rate_hz, current_step / rate_hz))

    return sorted(note for note in notes if note[3] > note[2])

### Tokenização
def tokenize_dataset(files):
    """Tokeniza cada música e salva os dados processados."""
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True)

    rows = []

    for split, paths in files.items():
        output_dir = OUTPUT_DIR / "tokens" / split
        output_dir.mkdir(parents=True)

        print(f"Tokenizando {split}: {len(paths)} MIDI")

        for path in paths:
            notes = read_notes(path)

            if not notes:
                continue

            ids = encode(notes)
            np.savez_compressed(output_dir / f"{path.stem}.npz", ids=ids)

            rows.append({
                "split": split,
                "file": path.name,
                "notes": len(notes),
                "tokens": len(ids),
            })

    with (OUTPUT_DIR / "token_index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("split", "file", "notes", "tokens"))
        writer.writeheader()
        writer.writerows(rows)

    (OUTPUT_DIR / "vocab.json").write_text(json.dumps(VOCAB, indent=2), encoding="utf-8")

### Execução
def main():
    extract_dataset()
    files = get_midi_files()
    tokenize_dataset(files)

    print("\nDataset preparado.")
    print("Saída:", OUTPUT_DIR)

if __name__ == "__main__":
    main()