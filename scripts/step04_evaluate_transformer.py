"""
step04_evaluate_transformer

Avalia o Transformer no conjunto de teste: loss, perplexidade e acurácia.

Perplexidade = exp(média da entropia cruzada, em log natural, de todos os
tokens depois do BOS em cada música de teste), como no scripts/protocol.py.
Cada token é pontuado uma única vez. A medida principal usa janelas
deslizantes (passo = contexto / 2), para que todo token tenha pelo menos meio
contexto de histórico; a medida com janelas disjuntas (passo = contexto, igual
à validação durante o treino) também é salva.

Salva results/transformer/test_metrics.json e copia o histórico de treino e as
informações da execução para results/transformer/.
"""

### Imports
from pathlib import Path
import argparse, json, shutil, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nes_continuation.data import ScoringWindows, collate_scoring  # noqa: E402
from nes_continuation.training import (  # noqa: E402
    device_name, load_checkpoint, portable_path, restore_model, score, show_metrics,
)
from protocol import TEST_SONGS, TEST_TARGET_TOKENS  # noqa: E402

### Caminhos
RUN_DIR = ROOT / "runs" / "transformer"
RESULTS_DIR = ROOT / "results" / "transformer"

### Avaliação
def evaluate(model, split, stride, batch_size, device, limit=None):
    """Pontua todos os tokens depois do BOS de uma partição, com janelas de passo `stride`."""
    dataset = ScoringWindows(split=split, context=model.config.context, stride=stride, limit=limit)
    batches = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_scoring, pin_memory=device.type == "cuda")

    start = time.monotonic()
    metrics = score(model, batches, device)
    metrics.update(songs=len(dataset.songs), windows=len(dataset), stride=stride, seconds=time.monotonic() - start)

    # Confere que cada token depois do BOS foi pontuado exatamente uma vez.
    assert metrics["tokens"] == dataset.target_count, (metrics["tokens"], dataset.target_count)
    return metrics

def copy_training_files(run_dir):
    """Copia o histórico de treino e as informações da execução (pequenos JSON) para results/."""
    for source, target in (("metrics.json", "train_history.json"), ("run_info.json", "run_info.json")):
        if (run_dir / source).exists():
            shutil.copyfile(run_dir / source, RESULTS_DIR / target)

### Execução
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=RUN_DIR / "best.pt")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--limit", type=int, help="máximo de músicas (só para testar o script)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "test_metrics.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    checkpoint = load_checkpoint(args.checkpoint)
    model = restore_model(checkpoint, device)
    context = model.config.context
    print(f"{device_name(device)} | checkpoint da época {checkpoint['epoch']} | contexto {context}", flush=True)

    sliding = evaluate(model, args.split, max(1, context // 2), args.batch_size, device, args.limit)
    show_metrics(f"{args.split} (janelas deslizantes)", sliding)
    disjoint = evaluate(model, args.split, context, args.batch_size, device, args.limit)
    show_metrics(f"{args.split} (janelas disjuntas)", disjoint)

    full_test = args.split == "test" and args.limit is None
    if full_test:
        assert sliding["songs"] == TEST_SONGS and sliding["tokens"] == TEST_TARGET_TOKENS, sliding

    results = {
        "split": args.split,
        "checkpoint": portable_path(args.checkpoint),
        "epoch": checkpoint["epoch"],
        "best_valid_loss": checkpoint["best_loss"],
        "device": device_name(device),
        "definition": "exp(média da entropia cruzada, log natural, de todos os tokens depois do BOS)",
        "tokens": sliding["tokens"],
        "loss": sliding["loss"],
        "perplexity": sliding["perplexity"],
        "accuracy": sliding["accuracy"],
        "sliding_windows": sliding,
        "disjoint_windows": disjoint,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Métricas: {args.output}")

    if full_test and args.output.parent == RESULTS_DIR:
        copy_training_files(args.checkpoint.parent)

if __name__ == "__main__":
    main()
