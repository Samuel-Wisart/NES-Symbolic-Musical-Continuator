"""Treinamento e avaliação por entropia cruzada do próximo token."""

import json
import math
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import NESWindows, collate_windows
from .model import ModelConfig, NESTransformer
from .tokenizer import EVENTS, PAD

CHECKPOINT_VERSION = 1


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != CHECKPOINT_VERSION or checkpoint.get("events") != EVENTS:
        raise ValueError(f"Checkpoint incompatível com a versão ou o vocabulário atual: {path}")
    return checkpoint


def restore_model(checkpoint: dict, device: torch.device) -> NESTransformer:
    model = NESTransformer(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def save_checkpoint(checkpoint: dict, path: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def make_dataset(root: Path, split: str, context: int, limit: int | None) -> NESWindows:
    stride = None if split == "train" else context
    dataset = NESWindows(root, split, context=context, stride=stride, limit=limit)
    print(
        f"{split}: {len(dataset.songs)} músicas, {len(dataset)} janelas, "
        f"{dataset.skipped_silent} sem ataques descartadas",
        flush=True,
    )
    return dataset


def make_loader(
    dataset: NESWindows, batch_size: int, device: torch.device, seed: int | None = None,
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=seed is not None,
        generator=torch.Generator().manual_seed(seed if seed is not None else 0),
        collate_fn=collate_windows, pin_memory=device.type == "cuda",
    )


def token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), ignore_index=PAD, reduction="sum",
    )


def train_epoch(model, batches, optimizer, device, bf16: bool, log_every: int) -> float:
    model.train()
    loss_sum = torch.zeros((), device=device)
    total_tokens = 0
    start = time.monotonic()
    for step, (inputs, targets) in enumerate(batches, 1):
        tokens = int(targets.ne(PAD).sum())
        total_tokens += tokens
        optimizer.zero_grad(set_to_none=True)
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            loss = token_loss(model(inputs), targets)
        (loss / tokens).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_sum += loss.detach()
        if step % log_every == 0 or step == len(batches):
            elapsed = time.monotonic() - start
            average = loss_sum.item() / total_tokens
            remaining = elapsed / step * (len(batches) - step)
            print(
                f"  lote {step}/{len(batches)} | loss={average:.4f} | "
                f"{elapsed / step:.3f} s/lote | treino da época: faltam ~{remaining / 60:.1f} min",
                flush=True,
            )
    return loss_sum.item() / total_tokens


@torch.inference_mode()
def score(model, batches, device) -> dict:
    model.eval()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    total_tokens = 0
    for inputs, targets in batches:
        total_tokens += int(targets.ne(PAD).sum())
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss_sum += token_loss(logits, targets)
        correct += ((logits.argmax(dim=-1) == targets) & targets.ne(PAD)).sum()
    loss = loss_sum.item() / total_tokens
    return {
        "tokens": total_tokens,
        "loss": loss,
        "perplexity": math.exp(loss),
        "accuracy": correct.item() / total_tokens,
    }


def show_metrics(split: str, metrics: dict) -> None:
    print(
        f"{split}: loss={metrics['loss']:.4f} | perplexidade={metrics['perplexity']:.2f} | "
        f"acurácia={metrics['accuracy']:.2%} | {metrics['tokens']} tokens",
        flush=True,
    )


def train(args, device: torch.device) -> None:
    latest = args.run_dir / "last.pt"
    best = args.run_dir / "best.pt"
    if not args.resume and (latest.exists() or best.exists()):
        raise ValueError("Já há checkpoints em --run-dir. Use --resume ou outra pasta.")

    run_config = {
        "data": str(args.data.resolve()),
        "train_limit": args.train_limit,
        "valid_limit": args.valid_limit,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
    }
    start_epoch = 1
    best_loss = math.inf
    history = []
    torch.manual_seed(args.seed)
    if args.resume:
        checkpoint = load_checkpoint(latest)
        if checkpoint["run_config"] != run_config:
            raise ValueError("Repita os argumentos originais de dados, lote, taxa e semente ao retomar.")
        model = restore_model(checkpoint, device)
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        history = checkpoint["history"]
    else:
        config = ModelConfig(args.context, args.width, args.heads, args.layers, args.dropout)
        model = NESTransformer(config).to(device)

    if start_epoch > args.epochs:
        print(f"Treino já concluído até a época {start_epoch - 1}. Aumente --epochs para continuar.")
        return

    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["cpu_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)

    context = model.config.context
    train_set = make_dataset(args.data, "train", context, args.train_limit)
    valid_set = make_dataset(args.data, "valid", context, args.valid_limit)
    valid_batches = make_loader(valid_set, args.batch_size, device)
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"{device} | {'BF16 misto' if bf16 else 'FP32'} | {params:,} parâmetros | "
        f"{math.ceil(len(train_set) / args.batch_size)} lotes de treino/época | "
        f"{len(valid_batches)} lotes de validação/época",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Época {epoch}/{args.epochs}", flush=True)
        start = time.monotonic()
        batches = make_loader(train_set, args.batch_size, device, args.seed + epoch)
        train_loss = train_epoch(model, batches, optimizer, device, bf16, args.log_every)
        metrics = score(model, valid_batches, device)
        show_metrics("valid", metrics)
        if not math.isfinite(train_loss) or not math.isfinite(metrics["loss"]):
            raise ValueError("Loss não finita. Confira a taxa de aprendizado e retome o último checkpoint.")
        improved = metrics["loss"] < best_loss
        best_loss = min(best_loss, metrics["loss"])
        elapsed = time.monotonic() - start
        history.append({"epoch": epoch, "seconds": elapsed, "train_loss": train_loss, "valid": metrics})
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "events": EVENTS,
            "model_config": asdict(model.config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "run_config": run_config,
            "history": history,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }
        if improved:
            save_checkpoint(checkpoint, best)
        save_checkpoint(checkpoint, latest)
        (args.run_dir / "metrics.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8",
        )
        print(
            f"Época em {elapsed:.1f} s | ETA total ~{elapsed * (args.epochs - epoch) / 60:.1f} min | "
            f"checkpoint: {latest}" + (" | novo best.pt" if improved else ""),
            flush=True,
        )


def evaluate(args, device: torch.device) -> None:
    checkpoint = load_checkpoint(args.checkpoint)
    model = restore_model(checkpoint, device)
    dataset = make_dataset(args.data, args.split, model.config.context, args.limit)
    batches = make_loader(dataset, args.batch_size, device)
    print(f"{device} | checkpoint da época {checkpoint['epoch']}", flush=True)
    show_metrics(args.split, score(model, batches, device))
