"""
training

Treinamento e avaliação por entropia cruzada do próximo token.
"""

### Imports
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
from .model import ModelConfig, build_model
from .representation import PAD, ROOT, VOCAB
from .stop import should_stop

### Checkpoints
CHECKPOINT_VERSION = 2

def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != CHECKPOINT_VERSION or checkpoint.get("vocab") != list(VOCAB):
        raise ValueError(f"Checkpoint incompatível com a versão ou o vocabulário atual: {path}")
    return checkpoint

def restore_model(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    model = build_model(ModelConfig(**checkpoint["model_config"])).to(device)
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

def save_json(data, path: Path) -> None:
    """Grava JSON de forma atômica (temporário na mesma pasta + replace)."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def export_model(checkpoint: dict, path: Path) -> None:
    """Exporta os pesos e metadados necessários para avaliar e gerar músicas."""
    keys = ("version", "vocab", "model_config", "model", "epoch", "best_loss", "run_config", "history")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint({key: checkpoint[key] for key in keys if key in checkpoint}, path)

### Dados
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

### Loss e métricas
def token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), ignore_index=PAD, reduction="sum",
    )

def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """Escolhe BF16, FP16 ou FP32 conforme o suporte do dispositivo."""
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16
    return torch.float16

def precision_name(amp_dtype: torch.dtype | None) -> str:
    return {torch.bfloat16: "BF16 misto", torch.float16: "FP16 misto"}.get(amp_dtype, "FP32")

def train_step(model, inputs, targets, optimizer, scaler, device, amp_dtype) -> tuple[torch.Tensor, int]:
    """Atualiza os pesos e retorna a loss somada e o total de tokens-alvo."""
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

### Utilitários
def show_metrics(split: str, metrics: dict) -> None:
    print(
        f"{split}: loss={metrics['loss']:.4f} | perplexidade={metrics['perplexity']:.2f} | "
        f"acurácia={metrics['accuracy']:.2%} | {metrics['tokens']} tokens",
        flush=True,
    )

### Caminhos e dispositivo
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

### Benchmark
def benchmark(model, train_set, valid_set, optimizer, scaler, device, amp_dtype, args) -> None:
    """Estima a duração de uma época a partir de alguns lotes."""
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

    # Mede alguns lotes de validação e extrapola o tempo total.
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

### Treinamento
def epoch_order(dataset_size: int, seed: int, epoch: int) -> list[int]:
    """Gera a ordem das janelas a partir da semente e do número da época."""
    return torch.randperm(dataset_size, generator=torch.Generator().manual_seed(seed + epoch)).tolist()

def train(args, device: torch.device) -> None:
    """Treina ou retoma automaticamente de `last.pt`.

    Salva checkpoints periodicamente, ao fim de cada época e antes de parar.
    Registra o lote atual para retomar do ponto salvo e grava `CONCLUIDO` ao terminar.
    """
    latest = args.run_dir / "last.pt"
    best = args.run_dir / "best.pt"
    progress_file = args.run_dir / "progresso.json"
    done_file = args.run_dir / "CONCLUIDO"
    if args.do_zero and not args.benchmark:
        for path in (latest, best, progress_file, done_file, args.run_dir / "metrics.json"):
            path.unlink(missing_ok=True)
        print(f"[do zero] estado anterior apagado em {args.run_dir}", flush=True)
    if not args.benchmark and done_file.exists():
        done = json.loads(done_file.read_text(encoding="utf-8"))
        if done["epochs"] >= args.epochs:
            print(f"Treino já concluído ({done['epochs']} épocas, {done_file}). "
                  "Aumente --epochs para continuar ou use --do-zero.", flush=True)
            return

    run_config = {
        "data": portable_path(args.data),
        "train_limit": args.train_limit,
        "valid_limit": args.valid_limit,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
    }
    requested_model = ModelConfig(args.context, args.width, args.heads, args.layers, args.dropout, args.arch)
    resume = latest.exists() and not args.benchmark
    completed_epochs, next_batch = 0, 0
    partial_loss, partial_tokens, partial_seconds = 0.0, 0, 0.0
    best_loss = math.inf
    history = []
    torch.manual_seed(args.seed)
    if resume:
        checkpoint = load_checkpoint(latest)
        saved_model = asdict(ModelConfig(**checkpoint["model_config"]))  # Sem `arch`, usa Transformer.
        if checkpoint["run_config"] != run_config or saved_model != asdict(requested_model):
            raise ValueError(
                f"Configuração diferente da salva em {latest}:\n"
                f"  salva: {checkpoint['run_config']} {checkpoint['model_config']}\n"
                f"  atual: {run_config} {asdict(requested_model)}\n"
                "Repita os argumentos originais, use outra --run-dir ou --do-zero."
            )
        model = restore_model(checkpoint, device)
        completed_epochs = checkpoint["epoch"]
        next_batch = checkpoint.get("batch", 0)
        partial_loss, partial_tokens, partial_seconds = checkpoint.get("partial", (0.0, 0, 0.0))
        best_loss = checkpoint["best_loss"]
        history = checkpoint["history"]
        print(f"[retomada] época {completed_epochs + 1}, lote {next_batch} ({latest})", flush=True)
    else:
        model = build_model(requested_model).to(device)
        if not args.benchmark:
            print("[início] sem checkpoint: começando do zero", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    amp_dtype = autocast_dtype(device)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    if resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        torch.set_rng_state(checkpoint["cpu_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)

    context = model.config.context
    train_set = make_dataset(args.data, "train", context, args.train_limit)
    valid_set = make_dataset(args.data, "valid", context, args.valid_limit)
    valid_batches = make_loader(valid_set, args.batch_size, device)
    batches_per_epoch = math.ceil(len(train_set) / args.batch_size)
    params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"{device_name(device)} | {precision_name(amp_dtype)} | {params:,} parâmetros | "
        f"{batches_per_epoch} lotes de treino/época | "
        f"{len(valid_batches)} lotes de validação/época",
        flush=True,
    )
    if args.benchmark:
        benchmark(model, train_set, valid_set, optimizer, scaler, device, amp_dtype, args)
        return

    # Registra a configuração e o tamanho de cada partição.
    args.run_dir.mkdir(parents=True, exist_ok=True)
    done_file.unlink(missing_ok=True)  # Permite continuar se --epochs aumentar.
    save_json({
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
        "batches_per_epoch": batches_per_epoch,
    }, args.run_dir / "run_info.json")

    def save_state(epoch: int, batch: int, partial: tuple, improved: bool = False) -> None:
        """Salva o estado do treino e os acumulados da época atual."""
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "vocab": list(VOCAB),
            "model_config": asdict(model.config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "batch": batch,
            "partial": partial,
            "best_loss": best_loss,
            "run_config": run_config,
            "history": history,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }
        if improved:
            save_checkpoint(checkpoint, best)
        save_checkpoint(checkpoint, latest)
        save_json({
            "epoch": epoch,
            "batch": batch,
            "batches_per_epoch": batches_per_epoch,
            "epochs_requested": args.epochs,
            "best_valid_loss": None if math.isinf(best_loss) else best_loss,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, progress_file)

    last_save = time.monotonic()
    epoch = completed_epochs + 1
    while epoch <= args.epochs:
        print(f"Época {epoch}/{args.epochs}" + (f" (a partir do lote {next_batch})" if next_batch else ""), flush=True)
        # Inclui no tempo o trecho executado antes da retomada.
        start = time.monotonic() - partial_seconds
        order = epoch_order(len(train_set), args.seed, epoch)
        batches = DataLoader(
            train_set, batch_size=args.batch_size, sampler=order[next_batch * args.batch_size:],
            collate_fn=collate_windows, pin_memory=device.type == "cuda",
            # Preserva a aleatoriedade do dropout ao retomar no meio da época.
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        model.train()
        segment_start = time.monotonic()
        loss_sum = torch.tensor(partial_loss, device=device, dtype=torch.float64)
        total_tokens = partial_tokens
        for step, (inputs, targets) in enumerate(batches, next_batch + 1):
            loss, tokens = train_step(model, inputs, targets, optimizer, scaler, device, amp_dtype)
            total_tokens += tokens
            loss_sum += loss
            stop = should_stop()
            if step % args.log_every == 0 or step == batches_per_epoch:
                per_batch = (time.monotonic() - segment_start) / (step - next_batch)
                remaining = per_batch * (batches_per_epoch - step)
                print(
                    f"  lote {step}/{batches_per_epoch} | loss={loss_sum.item() / total_tokens:.4f} | "
                    f"{per_batch:.3f} s/lote | treino da época: faltam ~{remaining / 60:.1f} min",
                    flush=True,
                )
            if step < batches_per_epoch and (stop or time.monotonic() - last_save >= args.checkpoint_minutes * 60):
                save_state(epoch - 1, step, (loss_sum.item(), total_tokens, time.monotonic() - start))
                last_save = time.monotonic()
                if stop:
                    print(f"[parada] salvo na época {epoch}, lote {step}; rode o mesmo comando para continuar",
                          flush=True)
                    return
                print(f"  [checkpoint] época {epoch}, lote {step}", flush=True)

        train_loss = loss_sum.item() / total_tokens
        metrics = score(model, valid_batches, device)
        show_metrics("valid", metrics)
        if not math.isfinite(train_loss) or not math.isfinite(metrics["loss"]):
            raise ValueError("Loss não finita. Confira a taxa de aprendizado e retome o último checkpoint.")
        improved = metrics["loss"] < best_loss
        best_loss = min(best_loss, metrics["loss"])
        elapsed = time.monotonic() - start
        history.append({"epoch": epoch, "seconds": elapsed, "train_loss": train_loss, "valid": metrics})
        save_state(epoch, 0, (0.0, 0, 0.0), improved)
        last_save = time.monotonic()
        save_json(history, args.run_dir / "metrics.json")
        print(
            f"Época em {elapsed:.1f} s | ETA total ~{elapsed * (args.epochs - epoch) / 60:.1f} min | "
            f"checkpoint: {latest}" + (" | novo best.pt" if improved else ""),
            flush=True,
        )
        epoch, next_batch = epoch + 1, 0
        partial_loss, partial_tokens, partial_seconds = 0.0, 0, 0.0
        if epoch <= args.epochs and should_stop():
            print("[parada] fim de época salvo; rode o mesmo comando para continuar", flush=True)
            return

    # Registra o total de épocas concluídas.
    completed = epoch - 1
    save_json({"epochs": completed, "best_valid_loss": best_loss, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
              done_file)
    print(f"Concluído: {completed} épocas, melhor loss de validação {best_loss:.4f}", flush=True)

### Avaliação
def evaluate(args, device: torch.device) -> dict:
    checkpoint = load_checkpoint(args.checkpoint)
    model = restore_model(checkpoint, device)
    dataset = make_dataset(args.data, args.split, model.config.context, args.limit)
    batches = make_loader(dataset, args.batch_size, device)
    print(f"{device} | checkpoint da época {checkpoint['epoch']}", flush=True)
    metrics = score(model, batches, device)
    show_metrics(args.split, metrics)
    return metrics
