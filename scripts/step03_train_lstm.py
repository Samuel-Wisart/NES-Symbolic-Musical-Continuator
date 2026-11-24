"""
step03_train_lstm

Treina o LSTM com o mesmo treino do Transformer (step03_train_transformer), para
comparar as duas arquiteturas: mesmos tokens do step01, mesmas janelas (contexto
1024, passo 512), mesmo otimizador (AdamW 3e-4, weight decay 0.01, clip 1.0),
lote 8, semente 42 e mesma seleção do best.pt pela loss de validação. Muda só o
modelo: 2 camadas LSTM de largura 512 (~4,8 M parâmetros, perto dos ~5,3 M do
Transformer), saída compartilhada com o embedding. Salva em runs/lstm/.

Padrão de 79 épocas: no treino entregue, a validação do LSTM foi a melhor na
época 12 e só piorou depois (overfitting), então o treino foi encerrado na 79.

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

from nes_continuation.__main__ import main as cli  # noqa: E402

### Configuração
# Os argumentos da linha de comando vêm depois e, no argparse, o último vence.
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
