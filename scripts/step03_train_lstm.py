"""
step03_train_lstm

Treina o LSTM com os mesmos dados e parâmetros de otimização do Transformer.
Usa duas camadas de largura 512 e salva checkpoints em runs/lstm/.

    python scripts/step03_train_lstm.py
    python scripts/step03_train_lstm.py --benchmark 30 --batch-size 32

O mesmo comando começa do zero ou continua do last.pt. Para recomeçar, use
--do-zero ou outra --run-dir.
"""

### Imports
from pathlib import Path
import sys

### Caminhos
ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "lstm"

sys.path.insert(0, str(ROOT))

from nes_continuation.__main__ import main as cli

### Configuração
# Argumentos adicionais substituem os padrões definidos pela CLI.
DEFAULTS = [
    "--run-dir", str(RUN_DIR),
    "--epochs", "79",
    "--arch", "lstm",
    "--context", "1024",
    "--layers", "2",
    "--width", "512",
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
