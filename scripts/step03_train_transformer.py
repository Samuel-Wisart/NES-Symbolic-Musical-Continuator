"""
step03_train_transformer

Treina o Transformer decoder-only sobre os tokens do step01.

Treina por padrão por 100 épocas com contexto 1024, 6 camadas, largura 256 e
8 cabeças. Salva o melhor checkpoint, o último estado e as métricas em
runs/transformer/. Argumentos da CLI podem substituir os padrões:

    python scripts/step03_train_transformer.py
    python scripts/step03_train_transformer.py --epochs 20
    python scripts/step03_train_transformer.py --benchmark 30 --batch-size 32

O mesmo comando inicia ou retoma o treino, inclusive no meio de uma época. Use
--do-zero ou outro --run-dir para começar novamente.
"""

### Imports
from pathlib import Path
import sys

### Caminhos
ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "transformer"

sys.path.insert(0, str(ROOT))

from nes_continuation.__main__ import main as cli

### Configuração
DEFAULTS = [
    "--run-dir", str(RUN_DIR),
    "--epochs", "100",
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
