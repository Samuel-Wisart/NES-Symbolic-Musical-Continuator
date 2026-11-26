"""
__main__

Comandos para treinar, avaliar e gerar continuações musicais.
"""

### Tratamento de sinais
# Instala os tratadores de sinal antes dos imports pesados.
from nes_continuation.stop import install_handlers

install_handlers()

### Imports
import argparse
import math
from pathlib import Path

import torch

from nes_continuation.model import ARCHITECTURES, ModelConfig
from nes_continuation.representation import TOKENS_DIR
from nes_continuation.sampling import generate
from nes_continuation.training import evaluate, train

### Argumentos
def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("use um inteiro positivo")
    return number

def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("use um inteiro não negativo")
    return number

def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("use um número finito não negativo")
    return number

def positive_float(value: str) -> float:
    number = nonnegative_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("use um número positivo")
    return number

### Execução
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    common.add_argument("--threads", type=positive_int, help="threads PyTorch (padrão: até 4)")
    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("--data", type=Path, default=TOKENS_DIR, help="pasta com os .npz do step01 por partição")
    data.add_argument("--batch-size", type=positive_int, default=8)

    fit = commands.add_parser("train", parents=[common, data], help="treina e salva checkpoints")
    fit.set_defaults(handler=train)
    fit.add_argument("--run-dir", type=Path, default=Path("runs/nes"))
    fit.add_argument("--epochs", type=positive_int, default=10, help="número total de épocas")
    fit.add_argument(
        "--resume", action="store_true",
        help="obsoleto e ignorado: a retomada do last.pt de --run-dir é automática",
    )
    fit.add_argument("--do-zero", action="store_true", help="apaga o estado salvo em --run-dir e recomeça")
    fit.add_argument(
        "--checkpoint-minutes", type=positive_float, default=15.0,
        help="minutos entre checkpoints dentro da época (além do fim de cada época)",
    )
    fit.add_argument("--lr", type=positive_float, default=3e-4)
    fit.add_argument("--seed", type=nonnegative_int, default=42)
    fit.add_argument("--log-every", type=positive_int, default=50)
    fit.add_argument(
        "--benchmark", type=nonnegative_int, default=0,
        help="mede N lotes, estima o tempo por época e sai sem salvar nada",
    )
    fit.add_argument("--train-limit", type=positive_int, help="máximo de arquivos do treino")
    fit.add_argument("--valid-limit", type=positive_int, help="máximo de arquivos da validação")
    config = ModelConfig()
    fit.add_argument("--arch", choices=ARCHITECTURES, default=config.arch, help="arquitetura do modelo")
    fit.add_argument("--context", type=positive_int, default=config.context)
    fit.add_argument("--width", type=positive_int, default=config.width)
    fit.add_argument("--heads", type=positive_int, default=config.heads)
    fit.add_argument("--layers", type=positive_int, default=config.layers)
    fit.add_argument("--dropout", type=nonnegative_float, default=config.dropout)

    test = commands.add_parser("evaluate", parents=[common, data], help="mede predição do próximo token")
    test.set_defaults(handler=evaluate)
    test.add_argument("--checkpoint", type=Path, required=True)
    test.add_argument("--split", choices=("valid", "test"), default="test")
    test.add_argument("--limit", type=positive_int, help="máximo de arquivos da partição selecionada")

    gen = commands.add_parser("generate", parents=[common], help="gera MIDI a partir de um prompt")
    gen.set_defaults(handler=generate)
    gen.add_argument("--checkpoint", type=Path, required=True)
    gen.add_argument("--prompt", type=Path, required=True, help=".npz de tokens do step01 ou MIDI original")
    gen.add_argument("--prompt-seconds", type=positive_float, help="usa apenas os primeiros segundos da música")
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--wav", action="store_true", help="gera também um WAV ao lado do MIDI")
    gen.add_argument("--temperature", type=positive_float, default=1.0)
    gen.add_argument("--top-k", type=nonnegative_int, default=32, help="0 desativa top-k")
    gen.add_argument("--seed", type=nonnegative_int, default=42)
    gen.add_argument("--min-seconds", type=nonnegative_float, default=2.0, help="aceita EOS após este tempo de continuação")
    gen.add_argument("--max-seconds", type=positive_float, default=12.0, help="duração musical máxima da continuação")
    gen.add_argument("--max-new-tokens", type=nonnegative_int, default=0, help="0 = sem limite (a duração já limita)")

    args = parser.parse_args()
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    device = args.device
    if device == "auto":
        device = "cuda" if cuda else "mps" if mps else "cpu"
    if (device == "cuda" and not cuda) or (device == "mps" and not mps):
        parser.error(f"{device} indisponível nesta instalação do PyTorch")
    torch.set_num_threads(args.threads or min(4, torch.get_num_threads()))
# Ativa TF32 nas GPUs compatíveis para acelerar operações FP32.
    torch.set_float32_matmul_precision("high")
    try:
        args.handler(args, torch.device(device))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
