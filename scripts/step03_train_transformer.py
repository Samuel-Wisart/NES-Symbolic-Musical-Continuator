"""
step03_train_transformer

Treina o Transformer decoder-only sobre os tokens do step01.

Usa a configuração da Luiza (contexto 1024, 6 camadas, largura 256, 8 cabeças,
dropout 0.1, AdamW 3e-4, lote 8) e salva best.pt, last.pt e metrics.json em
runs/transformer/. Qualquer argumento de `python -m nes_continuation train`
pode ser passado e substitui o padrão, por exemplo:

    python scripts/step03_train_transformer.py --epochs 20 --batch-size 32
    python scripts/step03_train_transformer.py --epochs 20 --batch-size 32 --resume
    python scripts/step03_train_transformer.py --benchmark 30 --batch-size 32
"""

### Imports
from pathlib import Path
import sys

### Caminhos
ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "transformer"

sys.path.insert(0, str(ROOT))

from nes_continuation.__main__ import main as cli  # noqa: E402

### Configuração
# Os argumentos da linha de comando vêm depois e, no argparse, o último vence.
DEFAULTS = [
    "--run-dir", str(RUN_DIR),
    "--context", "1024",
    "--layers", "6",
    "--width", "256",
    "--heads", "8",
    "--dropout", "0.1",
    "--lr", "3e-4",
    "--batch-size", "8",
    "--seed", "42",
]

### Execução
def main():
    sys.argv = [sys.argv[0], "train", *DEFAULTS, *sys.argv[1:]]
    cli()

if __name__ == "__main__":
    main()
