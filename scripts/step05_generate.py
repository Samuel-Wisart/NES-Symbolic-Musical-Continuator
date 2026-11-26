"""
step05_generate

Gera as continuações de um modelo (Transformer ou LSTM) seguindo o
scripts/protocol.py.

    python scripts/step05_generate.py --model transformer
    python scripts/step05_generate.py --model lstm

Usa runs/<modelo>/best.pt ou o modelo exportado em resultados/<modelo>/modelo.pt.
Salva os dados da geração, o MIDI e a referência real em resultados/amostras/.
Também cria WAVs chiptune para as primeiras N_WAV amostras.
"""

### Imports
from pathlib import Path
import argparse, json, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from nes_continuation.representation import (
    TIME_RESOLUTION_HZ, cut_prompt, duration_steps, load_ids, reference_continuation,
)
from nes_continuation.sampling import sample, save_generation
from nes_continuation.training import device_name, load_checkpoint, portable_path, restore_model
import protocol

### Funções auxiliares
def settings():
    """Parâmetros do protocolo registrados no summary.json."""
    names = ("PROMPT_SECONDS", "GEN_SECONDS", "MIN_GEN_SECONDS", "N_PROMPTS", "SEED", "TEMPERATURE", "TOP_K")
    return {name.lower(): getattr(protocol, name) for name in names}

def write_summary(folder, summary):
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

### Execução
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=protocol.MODELS, required=True, help="modelo que gera as continuações")
    parser.add_argument("--checkpoint", type=Path, help="padrão: runs/<modelo>/best.pt ou resultados/<modelo>/modelo.pt")
    parser.add_argument("--output-dir", type=Path, default=protocol.GENERATED_DIR)
    parser.add_argument("--limit", type=int, help="só as primeiras músicas (para testar o script)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint or protocol.checkpoint_path(args.model)

    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    model = restore_model(checkpoint, device)
    model_dir = args.output_dir / args.model
    reference_dir = args.output_dir / protocol.REFERENCE_NAME
    print(f"{device_name(device)} | checkpoint da época {checkpoint['epoch']}", flush=True)

    model_rows, reference_rows = [], []

    for index, path in enumerate(protocol.prompt_paths()[:args.limit]):
        name = path.stem
        ids = load_ids(path)
        prompt = cut_prompt(ids, protocol.PROMPT_SECONDS)
        wav = index < protocol.N_WAV

        start = time.monotonic()
        result = sample(
            model, prompt, device,
            seed=protocol.generation_seed(index),
            temperature=protocol.TEMPERATURE,
            top_k=protocol.TOP_K,
            min_seconds=protocol.MIN_GEN_SECONDS,
            max_seconds=protocol.GEN_SECONDS,
        )
        seconds = time.monotonic() - start
        save_generation(result, model_dir / f"{name}.mid", wav=wav)

        reference, prompt_length = reference_continuation(ids, protocol.PROMPT_SECONDS, protocol.GEN_SECONDS)
        save_generation({"ids": reference, "prompt_length": prompt_length}, reference_dir / f"{name}.mid", wav=wav)

        model_rows.append({
            "name": name,
            "seed": protocol.generation_seed(index),
            "prompt_tokens": result["prompt_length"],
            "generated_tokens": result["generated_tokens"],
            "generated_seconds": result["generated_steps"] / TIME_RESOLUTION_HZ,
            "stop_reason": result["stop_reason"],
            "wall_seconds": round(seconds, 2),
            "wav": wav,
        })
        reference_rows.append({
            "name": name,
            "prompt_tokens": prompt_length,
            "continuation_tokens": len(reference) - prompt_length,
            "continuation_seconds": duration_steps(reference[prompt_length:]) / TIME_RESOLUTION_HZ,
            "wav": wav,
        })
        print(
            f"[{index + 1}/{len(protocol.PROMPT_SONGS)}] {name}: {result['generated_tokens']} tokens, "
            f"{result['generated_steps'] / TIME_RESOLUTION_HZ:.2f} s, {result['stop_reason']}, {seconds:.1f} s",
            flush=True,
        )

    write_summary(model_dir, {
        "model": args.model,
        "checkpoint": portable_path(args.checkpoint),
        "epoch": checkpoint["epoch"],
        "protocol": settings(),
        "songs": model_rows,
    })
    write_summary(reference_dir, {"model": "reference", "protocol": settings(), "songs": reference_rows})
    print(f"Saída: {model_dir} e {reference_dir}")

if __name__ == "__main__":
    main()
