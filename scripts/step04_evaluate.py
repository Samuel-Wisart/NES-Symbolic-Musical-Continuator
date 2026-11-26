"""
step04_evaluate

Avalia um modelo (Transformer ou LSTM) no conjunto de teste: loss, perplexidade
e acurácia.

    python scripts/step04_evaluate.py --model transformer
    python scripts/step04_evaluate.py --model lstm

Perplexidade = exp(média da entropia cruzada, em log natural, de todos os
tokens depois do BOS em cada música de teste), como no scripts/protocol.py.
Cada token é pontuado uma única vez. A medida principal usa janelas
deslizantes (passo = contexto / 2), para que todo token tenha pelo menos meio
contexto de histórico. A medida com janelas disjuntas (passo = contexto, igual
à validação durante o treino) também é salva.

Usa runs/<modelo>/best.pt (se o modelo foi treinado nesta máquina) ou o modelo
entregue em resultados/<modelo>/modelo.pt. Salva as métricas em
resultados/<modelo>/test_metrics.json. Se usar best.pt, também copia o histórico
e as informações da execução e exporta os pesos sem o estado do otimizador.
"""

### Imports
from pathlib import Path
import argparse, json, shutil, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from nes_continuation.data import ScoringWindows, collate_scoring
from nes_continuation.training import (
    device_name, export_model, load_checkpoint, portable_path, restore_model, score, show_metrics,
)
from protocol import MODELS, RESULTS_DIR, TEST_SONGS, TEST_TARGET_TOKENS, checkpoint_path

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

def copy_training_files(run_dir, results_dir):
    """Copia o histórico de treino e as informações da execução (pequenos JSON) para resultados/."""
    for source, target in (("metrics.json", "train_history.json"), ("run_info.json", "run_info.json")):
        if (run_dir / source).exists():
            shutil.copyfile(run_dir / source, results_dir / target)

### Execução
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True, help="modelo a avaliar")
    parser.add_argument("--checkpoint", type=Path, help="padrão: runs/<modelo>/best.pt ou resultados/<modelo>/modelo.pt")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--limit", type=int, help="máximo de músicas (só para testar o script)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results_dir = RESULTS_DIR / args.model
    args.checkpoint = args.checkpoint or checkpoint_path(args.model)
    args.output = args.output or results_dir / "test_metrics.json"

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

    trained = args.checkpoint.resolve() != (results_dir / "modelo.pt").resolve()
    if full_test and args.output.parent == results_dir and trained:
        copy_training_files(args.checkpoint.parent, results_dir)
        export_model(checkpoint, results_dir / "modelo.pt")
        print(f"Modelo exportado: {results_dir / 'modelo.pt'}")

if __name__ == "__main__":
    main()
