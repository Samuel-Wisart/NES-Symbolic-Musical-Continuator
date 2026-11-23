"""Treinamento e avaliação por entropia cruzada do próximo token."""

import json
import math
import platform
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import NESWindows, collate_windows
from .model import ModelConfig, NESTransformer
from .representation import PAD, ROOT, VOCAB

# Versão 2: vocabulário do step01 (VOCAB) no lugar dos EVENTS do tokenizador antigo.
CHECKPOINT_VERSION = 2


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != CHECKPOINT_VERSION or checkpoint.get("vocab") != list(VOCAB):
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
        f"{dataset.target_count} tokens-alvo",
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


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """BF16 nas GPUs que suportam; FP16 (com GradScaler) nas demais GPUs; FP32 na CPU."""
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16
    return torch.float16


def precision_name(amp_dtype: torch.dtype | None) -> str:
    return {torch.bfloat16: "BF16 misto", torch.float16: "FP16 misto"}.get(amp_dtype, "FP32")


def train_step(model, inputs, targets, optimizer, scaler, device, amp_dtype) -> tuple[torch.Tensor, int]:
    """Um passo de otimização; devolve a soma da loss do lote e o número de tokens-alvo."""
    tokens = int(targets.ne(PAD).sum())
    optimizer.zero_grad(set_to_none=True)
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
        loss = token_loss(model(inputs), targets)
    scaler.scale(loss / tokens).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    return loss.detach(), tokens


def train_epoch(model, batches, optimizer, scaler, device, amp_dtype, log_every: int) -> float:
    model.train()
    loss_sum = torch.zeros((), device=device)
    total_tokens = 0
    start = time.monotonic()
    for step, (inputs, targets) in enumerate(batches, 1):
        loss, tokens = train_step(model, inputs, targets, optimizer, scaler, device, amp_dtype)
        total_tokens += tokens
        loss_sum += loss
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


def portable_path(path: Path) -> str:
    """Caminho relativo à raiz do repositório, para retomar o treino em outra máquina."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"{device.type} ({platform.processor() or platform.machine()})"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(model, train_set, valid_set, optimizer, scaler, device, amp_dtype, args) -> None:
    """Mede segundos por lote em lotes sorteados do treino e estima o tempo de uma época."""
    warmup = 3
    batches = make_loader(train_set, args.batch_size, device, args.seed)
    valid_batches = make_loader(valid_set, args.batch_size, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    times = []
    for index, (inputs, targets) in enumerate(batches):
        if index >= warmup + args.benchmark:
            break
        synchronize(device)
        start = time.monotonic()
        train_step(model, inputs, targets, optimizer, scaler, device, amp_dtype)
        synchronize(device)
        if index >= warmup:
            times.append(time.monotonic() - start)

    # Validação: mede alguns lotes e extrapola.
    valid_sample = min(10, len(valid_batches))
    start = time.monotonic()
    score(model, (batch for _, batch in zip(range(valid_sample), valid_batches)), device)
    synchronize(device)
    valid_seconds = (time.monotonic() - start) / valid_sample * len(valid_batches)

    per_batch = statistics.mean(times)
    epoch_seconds = per_batch * len(batches) + valid_seconds
    memory = ""
    if device.type == "cuda":
        memory = f" | pico de memória {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB"
    print(
        f"benchmark: {len(times)} lotes de {args.batch_size} | {per_batch:.3f} s/lote "
        f"(mín {min(times):.3f}, máx {max(times):.3f}){memory}\n"
        f"estimativa por época: {len(batches)} lotes x {per_batch:.3f} s + validação ~{valid_seconds / 60:.1f} min "
        f"= ~{epoch_seconds / 60:.1f} min ({epoch_seconds / 3600:.2f} h)",
        flush=True,
    )


def train(args, device: torch.device) -> None:
    latest = args.run_dir / "last.pt"
    best = args.run_dir / "best.pt"
    if not args.benchmark and not args.resume and (latest.exists() or best.exists()):
        raise ValueError("Já há checkpoints em --run-dir. Use --resume ou outra pasta.")

    run_config = {
        "data": portable_path(args.data),
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
    if args.resume and not args.benchmark:
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    amp_dtype = autocast_dtype(device)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    if args.resume and not args.benchmark:
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["cpu_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)

    context = model.config.context
    train_set = make_dataset(args.data, "train", context, args.train_limit)
    valid_set = make_dataset(args.data, "valid", context, args.valid_limit)
    valid_batches = make_loader(valid_set, args.batch_size, device)
    params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"{device_name(device)} | {precision_name(amp_dtype)} | {params:,} parâmetros | "
        f"{math.ceil(len(train_set) / args.batch_size)} lotes de treino/época | "
        f"{len(valid_batches)} lotes de validação/época",
        flush=True,
    )
    if args.benchmark:
        benchmark(model, train_set, valid_set, optimizer, scaler, device, amp_dtype, args)
        return

    # Informações da execução para o relatório (reescritas a cada início ou retomada).
    args.run_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "device": device_name(device),
        "precision": precision_name(amp_dtype),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "parameters": params,
        "model_config": asdict(model.config),
        "run_config": run_config,
        "epochs_requested": args.epochs,
        "train": {"songs": len(train_set.songs), "windows": len(train_set), "targets": train_set.target_count},
        "valid": {"songs": len(valid_set.songs), "windows": len(valid_set), "targets": valid_set.target_count},
        "batches_per_epoch": math.ceil(len(train_set) / args.batch_size),
    }
    (args.run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Época {epoch}/{args.epochs}", flush=True)
        start = time.monotonic()
        batches = make_loader(train_set, args.batch_size, device, args.seed + epoch)
        train_loss = train_epoch(model, batches, optimizer, scaler, device, amp_dtype, args.log_every)
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
            "vocab": list(VOCAB),
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


def evaluate(args, device: torch.device) -> dict:
    checkpoint = load_checkpoint(args.checkpoint)
    model = restore_model(checkpoint, device)
    dataset = make_dataset(args.data, args.split, model.config.context, args.limit)
    batches = make_loader(dataset, args.batch_size, device)
    print(f"{device} | checkpoint da época {checkpoint['epoch']}", flush=True)
    metrics = score(model, batches, device)
    show_metrics(args.split, metrics)
    return metrics
