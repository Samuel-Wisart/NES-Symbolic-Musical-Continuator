"""
step02_analyze_dataset

Analisa estatísticas básicas do dataset e da representação.
"""

### Imports
from pathlib import Path
from statistics import mean, median
import csv, json

### Caminhos
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "nesmdb"
TOKEN_INDEX = DATA_DIR / "token_index.csv"
VOCAB_FILE = DATA_DIR / "vocab.json"

### Dataset
SPLITS = ("train", "valid", "test")

def load_data():
    """Carrega o índice das músicas e o vocabulário."""
    with TOKEN_INDEX.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    with VOCAB_FILE.open(encoding="utf-8") as file:
        vocab = json.load(file)

    return rows, vocab

### Estatísticas
def print_dataset_stats(rows):
    """Analisa quantidade de músicas e notas."""
    print("Dataset")

    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        songs = len(split_rows)
        notes = sum(int(row["notes"]) for row in split_rows)

        print(f"{split}: {songs} músicas | {notes} notas")

    total_notes = sum(int(row["notes"]) for row in rows)
    print(f"total: {len(rows)} músicas | {total_notes} notas")

def print_representation_stats(rows, vocab):
    """Analisa tamanho do vocabulário e comprimento das sequências."""
    print("\nRepresentação")
    print(f"vocabulário: {len(vocab)} tokens")

    print("\nComprimento das sequências")

    for split in SPLITS:
        lengths = [int(row["tokens"]) for row in rows if row["split"] == split]
        print(f"{split}: média {mean(lengths):.1f} | mediana {median(lengths):.1f}")

    lengths = [int(row["tokens"]) for row in rows]
    print(f"geral: média {mean(lengths):.1f} | mediana {median(lengths):.1f}")

def main():
    rows, vocab = load_data()
    print_dataset_stats(rows)
    print_representation_stats(rows, vocab)

if __name__ == "__main__":
    main()