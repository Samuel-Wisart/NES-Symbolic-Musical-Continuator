"""
step07_compare

Compara as métricas dos modelos e as distribuições das amostras geradas.

    python scripts/step07_compare.py

Lê os JSONs de resultados/<modelo>/ e as amostras em resultados/amostras/.
Gera tabelas e gráficos em resultados/. As métricas das amostras consideram
somente a continuação após o prompt.
"""

### Imports
from pathlib import Path
import json, logging, statistics, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nes_continuation.representation import CHANNELS, TIME_RESOLUTION_HZ, decode, duration_steps
import protocol

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

### Nomes e cores
NAMES = {"transformer": "Transformer", "lstm": "LSTM", "markov": "Markov", protocol.REFERENCE_NAME: "Referência (real)"}
# Cores dos modelos na ordem de exibição.
COLORS = {"transformer": "#2a78d6", "lstm": "#eb6834", "markov": "#1e9e6a"}
EXTRA_COLORS = ["#8a5cc2", "#c2406e"]
ORDER = ["transformer", "lstm", "markov"]
BOXPLOT_FILENAME = "boxplots_amostras.png"
MUSICAL_BOXPLOT_FILENAME = "boxplots_musicais.png"
BOXPLOT_METRICS = (
    ("notes_per_second", "Densidade de notas", "Notas por segundo"),
    ("mean_duration", "Duração média das notas", "Segundos"),
    ("active_voices", "Vozes ativas", "Vozes (de 4)"),
)
MUSICAL_BOXPLOT_METRICS = (
    ("polyphony", "Polifonia", "Alturas simultâneas"),
    ("pitch_class_entropy", "Entropia de classes de altura", "Bits"),
    ("scale_consistency", "Consistência de escala", "Proporção de notas"),
    ("groove_consistency", "Consistência de groove", "Similaridade entre blocos"),
)
SURFACE, INK_PRIMARY, INK_SECONDARY, INK_MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)
MELODIC = ("p1", "p2", "tr")
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
NATURAL_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)
# Usa blocos de 2 s como compassos de referência a 120 BPM.
GROOVE_BLOCK_STEPS = 2 * TIME_RESOLUTION_HZ

### Leitura
def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def name(key):
    return NAMES.get(key, key)

def decimal(number, digits=2):
    return f"{number:.{digits}f}".replace(".", ",")

def inteiro(number):
    return f"{number:,}".replace(",", ".")

### Modelos
def load_models():
    """Métricas de cada modelo com test_metrics.json em resultados/."""
    models = {}
    for folder in sorted(protocol.RESULTS_DIR.iterdir()):
        test = read_json(folder / "test_metrics.json")
        if not folder.is_dir() or test is None:
            continue
        history = read_json(folder / "train_history.json") or []
        run_info = read_json(folder / "run_info.json") or {}
        best = min(history, key=lambda row: row["valid"]["loss"]) if history else None
        models[folder.name] = {
            "parameters": run_info.get("parameters"),
            "epochs_trained": len(history),
            "best_epoch": best["epoch"] if best else test.get("epoch"),
            "best_valid_loss": best["valid"]["loss"] if best else test.get("best_valid_loss"),
            "best_valid_perplexity": best["valid"]["perplexity"] if best else None,
            "train_seconds": sum(row["seconds"] for row in history),
            "test_loss": test["loss"],
            "test_perplexity": test["perplexity"],
            "test_accuracy": test["accuracy"],
            "test_disjoint_perplexity": test["disjoint_windows"]["perplexity"],
            "history": [(row["epoch"], row["train_loss"], row["valid"]["loss"]) for row in history],
        }
    return dict(sorted(models.items(), key=lambda item: (ORDER.index(item[0]) if item[0] in ORDER else len(ORDER), item[0])))

### Amostras
def pitch_class_stats(continuation):
    """Entropia de Shannon e aderência à melhor escala maior ou menor natural."""
    classes = [pitch % 12 for voice, pitch, _, _ in continuation if voice in MELODIC]
    if not classes:
        return None, None
    counts = np.bincount(classes, minlength=12)
    probabilities = counts[counts > 0] / len(classes)
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    best_scale_count = max(
        sum(counts[(root + interval) % 12] for interval in scale)
        for scale in (MAJOR_SCALE, NATURAL_MINOR_SCALE)
        for root in range(12)
    )
    return entropy, float(best_scale_count / len(classes))

def polyphony(notes, start, steps):
    """Média de alturas distintas simultâneas nos passos com som melódico."""
    sounding = np.zeros((steps, 128), dtype=bool)
    for voice, pitch, onset, offset in notes:
        if voice in MELODIC and offset > start:
            first_step = max(0, round((onset - start) * TIME_RESOLUTION_HZ))
            last_step = round((offset - start) * TIME_RESOLUTION_HZ)
            sounding[first_step:last_step, pitch] = True
    counts = sounding.sum(axis=1)
    active = counts[counts > 0]
    return float(active.mean()) if active.size else None

def groove_consistency(continuation, start, steps):
    """Similaridade média dos vetores binários de ataques em blocos vizinhos."""
    blocks = steps // GROOVE_BLOCK_STEPS
    if blocks < 2:
        return None
    complete_steps = blocks * GROOVE_BLOCK_STEPS
    onsets = np.zeros(complete_steps, dtype=bool)
    for _, _, onset, _ in continuation:
        step = round((onset - start) * TIME_RESOLUTION_HZ)
        if 0 <= step < complete_steps:
            onsets[step] = True
    patterns = onsets.reshape(blocks, GROOVE_BLOCK_STEPS)
    return float(np.mean(patterns[:-1] == patterns[1:]))

def continuation_stats(path):
    """Estatísticas da continuação de uma amostra (.npz com ids e prompt_length)."""
    data = np.load(path)
    ids, prompt_length = data["ids"], int(data["prompt_length"])
    start = duration_steps(ids[:prompt_length]) / TIME_RESOLUTION_HZ
    steps = duration_steps(ids[prompt_length:])
    seconds = max(steps / TIME_RESOLUTION_HZ, 1e-9)
    notes = decode(ids)
    prompt = [note for note in notes if note[2] < start]
    continuation = [note for note in notes if note[2] >= start]

    stats = {"seconds": seconds, "notes_per_second": len(continuation) / seconds}
    for voice in CHANNELS:
        voice_notes = [note for note in continuation if note[0] == voice]
        pitches = [note[1] for note in voice_notes]
        stats[f"{voice}_notes_per_second"] = len(voice_notes) / seconds
        stats[f"{voice}_occupancy"] = min(1.0, sum(note[3] - note[2] for note in voice_notes) / seconds)
        stats[f"{voice}_mean_duration"] = statistics.mean(note[3] - note[2] for note in voice_notes) if voice_notes else None
        stats[f"{voice}_distinct_pitches"] = len(set(pitches))
        stats[f"{voice}_mean_pitch"] = statistics.mean(pitches) if pitches else None
        prompt_pitches = [note[1] for note in prompt if note[0] == voice]
        stats[f"{voice}_pitch_shift"] = (
            abs(statistics.mean(pitches) - statistics.mean(prompt_pitches)) if pitches and prompt_pitches else None
        )
    stats["active_voices"] = sum(stats[f"{voice}_notes_per_second"] > 0 for voice in CHANNELS)
    durations = [note[3] - note[2] for note in continuation]
    stats["mean_duration"] = statistics.mean(durations) if durations else None
    shifts = [stats[f"{voice}_pitch_shift"] for voice in MELODIC if stats[f"{voice}_pitch_shift"] is not None]
    stats["melodic_pitch_shift"] = statistics.mean(shifts) if shifts else None
    stats["polyphony"] = polyphony(notes, start, steps)
    stats["pitch_class_entropy"], stats["scale_consistency"] = pitch_class_stats(continuation)
    stats["groove_consistency"] = groove_consistency(continuation, start, steps)
    return stats

def mean_of(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return statistics.mean(values) if values else None

def load_samples():
    """Estatísticas individuais das músicas de cada conjunto de amostras."""
    sets = {}
    if not protocol.GENERATED_DIR.is_dir():
        return sets
    for folder in sorted(protocol.GENERATED_DIR.iterdir()):
        paths = sorted(folder.glob("*.npz")) if folder.is_dir() else []
        if not paths:
            continue
        sets[folder.name] = [continuation_stats(path) for path in paths]
    # Exibe modelos primeiro e a referência humana por último.
    rank = lambda key: (key == protocol.REFERENCE_NAME, ORDER.index(key) if key in ORDER else len(ORDER), key)
    return dict(sorted(sets.items(), key=lambda item: rank(item[0])))

### Figura
def plot_validation(models, output):
    plt.rcParams.update({"font.family": ["DejaVu Sans", "sans-serif"], "font.size": 10})
    figure, axis = plt.subplots(figsize=(7.5, 4.2), dpi=160)
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    extra = iter(EXTRA_COLORS)
    for key, model in models.items():
        color = COLORS.get(key) or next(extra)
        if not model["history"]:
            continue
        epochs, train, valid = zip(*model["history"])
        axis.plot(epochs, valid, color=color, linewidth=2, label=f"{name(key)} (validação)", zorder=3)
        axis.plot(epochs, train, color=color, linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.7,
                  label=f"{name(key)} (treino)", zorder=2)
        best = model["best_epoch"]
        axis.plot([best], [model["best_valid_loss"]], marker="o", markersize=6, color=color,
                  markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        axis.annotate(f"melhor: época {best}", (best, model["best_valid_loss"]), xytext=(6, 8),
                      textcoords="offset points", color=INK_SECONDARY, fontsize=8.5)
    axis.set_xlabel("época", color=INK_SECONDARY)
    axis.set_ylabel("entropia cruzada (nats por token)", color=INK_SECONDARY)
    axis.set_title("Loss por época: validação (sólida) e treino (tracejada)", loc="left",
                   color=INK_PRIMARY, fontsize=12, pad=38)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(AXIS)
    axis.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5,
                labelcolor=INK_SECONDARY, borderaxespad=0.2)
    figure.tight_layout()
    figure.savefig(output, facecolor=SURFACE)
    plt.close(figure)

def plot_sample_distributions(sample_rows, output, metrics, rows, columns):
    """Compara os valores por música das métricas das continuações."""
    groups = [*ORDER, protocol.REFERENCE_NAME]
    positions = range(1, len(groups) + 1)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4.4 * rows),
                                dpi=160, layout="constrained")
    figure.patch.set_facecolor(SURFACE)
    figure.suptitle("Distribuições das métricas das continuações", color=INK_PRIMARY)

    for axis, (metric, title, unit) in zip(axes.flat, metrics):
        labels = []
        for position, group in zip(positions, groups):
            values = [row[metric] for row in sample_rows.get(group, []) if row[metric] is not None]
            label = "Referência\nhumana" if group == protocol.REFERENCE_NAME else name(group)
            labels.append(f"{label}\n(n={len(values)})")
            if not values:
                continue
            color = COLORS.get(group, INK_MUTED)
            axis.boxplot(
                values, positions=[position], widths=0.55, patch_artist=True, manage_ticks=False,
                boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.65},
                medianprops={"color": INK_PRIMARY, "linewidth": 1.5},
                flierprops={"marker": "o", "markersize": 3, "markeredgecolor": color, "alpha": 0.65},
            )

        axis.set_facecolor(SURFACE)
        axis.set_title(title, color=INK_PRIMARY)
        axis.set_ylabel(unit, color=INK_SECONDARY)
        axis.set_xticks(positions, labels)
        axis.set_xlim(0.5, len(groups) + 0.5)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(labelcolor=INK_SECONDARY, length=0, labelsize=8.5)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(AXIS)
        if metric == "active_voices":
            axis.set_yticks(range(len(CHANNELS) + 1))
            axis.set_ylim(-0.2, len(CHANNELS) + 0.2)

    figure.savefig(output, facecolor=SURFACE)
    plt.close(figure)

### Markdown
def fmt(value, digits=2):
    return "—" if value is None else decimal(value, digits)

def write_markdown(models, samples, output):
    lines = [
        "# Comparação entre os modelos",
        "",
        "Gerado por `python scripts/step07_compare.py` a partir de `resultados/<modelo>/` e",
        "`resultados/amostras/`. Os números também estão em `comparacao.json`.",
        "",
        "## Métricas de predição (teste)",
        "",
        "Perplexidade = exp(média da entropia cruzada, em log natural, de todos os tokens depois do BOS",
        "das 372 músicas de teste; 435.683 tokens). Medida principal com janelas deslizantes (passo = contexto/2).",
        "",
        "| Modelo | Parâmetros | Épocas treinadas | Melhor época | Loss de validação | Loss de teste | **Perplexidade de teste** | Perplexidade (janelas disjuntas) | Acurácia top-1 | Tempo de treino |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, model in models.items():
        minutes = model["train_seconds"] / 60
        lines.append(
            f"| {name(key)} | {inteiro(model['parameters']) if model['parameters'] else '—'} | {model['epochs_trained']} "
            f"| {model['best_epoch']} | {fmt(model['best_valid_loss'], 4)} | {fmt(model['test_loss'], 4)} "
            f"| **{fmt(model['test_perplexity'], 3)}** | {fmt(model['test_disjoint_perplexity'], 3)} "
            f"| {decimal(100 * model['test_accuracy'], 2)}% | {decimal(minutes, 0)} min |"
        )
    lines += [
        "",
        "![Loss por época](curvas_validacao.png)",
        "",
        "## Estatísticas das amostras geradas",
        "",
        f"Só a continuação ({protocol.GEN_SECONDS} s depois do prompt de {protocol.PROMPT_SECONDS} s), média por música.",
        "A referência é a continuação real das mesmas músicas: quanto mais perto dela, mais parecido com o",
        "NES-MDB. *Desvio de altura* = |altura média da continuação − altura média do prompt| na mesma voz",
        "(média de p1, p2 e tr, em semitons): mede se a continuação fica no mesmo registro do prompt.",
        "",
        "| Conjunto | Músicas | Notas/s | Vozes ativas (de 4) | Duração média da nota (s) | Desvio de altura (semitons) |",
        "|---|---|---|---|---|---|",
    ]
    for key, row in samples.items():
        lines.append(
            f"| {name(key)} | {row['songs']} | {fmt(row['notes_per_second'])} | {fmt(row['active_voices'])} "
            f"| {fmt(row['mean_duration'], 3)} | {fmt(row['melodic_pitch_shift'])} |"
        )
    if samples:
        lines += [
            "",
            f"![Distribuições das métricas por música]({BOXPLOT_FILENAME})",
            "",
            "Cada caixa mostra a mediana e os quartis. Os bigodes vão até os valores dentro de 1,5",
            "vez a amplitude interquartil, e os demais valores aparecem como pontos.",
            "O n em cada grupo indica quantas músicas têm valor definido para a métrica.",
            "",
            f"![Distribuições de polifonia, entropia, escala e groove]({MUSICAL_BOXPLOT_FILENAME})",
            "",
            "| Conjunto | Polifonia | Entropia de classes de altura (bits) | Consistência de escala | Consistência de groove |",
            "|---|---:|---:|---:|---:|",
        ]
        for key in [*ORDER, protocol.REFERENCE_NAME]:
            row = samples.get(key, {})
            cells = [fmt(row.get(metric), 3) for metric, _, _ in MUSICAL_BOXPLOT_METRICS]
            lines.append(f"| {name(key)} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "Os valores da tabela são médias por música, considerando as amostras com valor definido",
            "para cada métrica. O travessão indica que não há valor disponível.",
            "",
            "A polifonia conta alturas distintas simultâneas nos instantes com som melódico, inclusive",
            "notas do prompt que ainda soam na continuação. A entropia de Shannon (em bits) e a proporção",
            "de notas na melhor escala maior ou menor natural usam os ataques da continuação.",
            "As três métricas de altura excluem o canal de ruído.",
            "",
            "A consistência de groove usa todos os canais e mede a proporção de posições com o mesmo",
            "estado de ataque em blocos consecutivos de 2 s. Como os tokens guardam tempo absoluto,",
            "a grade de referência equivale a 4/4 a 120 BPM, ancorada no início da continuação.",
            "São usados apenas blocos completos, com pelo menos dois por amostra.",
            "",
            "[Definições das métricas na documentação do MusPy](https://muspy.readthedocs.io/en/latest/metrics.html).",
        ]
    lines += [
        "",
        "Por voz (notas por segundo / fração do tempo soando / alturas distintas):",
        "",
        "| Conjunto | " + " | ".join(CHANNELS) + " |",
        "|---|" + "---|" * len(CHANNELS),
    ]
    for key, row in samples.items():
        cells = [
            f"{fmt(row[f'{voice}_notes_per_second'])} / {fmt(row[f'{voice}_occupancy'])} / {fmt(row[f'{voice}_distinct_pitches'], 1)}"
            for voice in CHANNELS
        ]
        lines.append(f"| {name(key)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Vozes: p1 e p2 = pulsos, tr = triângulo, no = ruído (a \"altura\" do ruído é o período, 1–16).",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")

### Execução
def main():
    models = load_models()
    sample_rows = load_samples()
    if not models:
        raise SystemExit("Nenhum arquivo resultados/<modelo>/test_metrics.json encontrado. Rode o step04 antes.")
    samples = {
        key: {"songs": len(rows), **{metric: mean_of(rows, metric) for metric in rows[0]}}
        for key, rows in sample_rows.items()
    }
    plot_validation(models, protocol.RESULTS_DIR / "curvas_validacao.png")
    if sample_rows:
        plot_sample_distributions(sample_rows, protocol.RESULTS_DIR / BOXPLOT_FILENAME, BOXPLOT_METRICS, 1, 3)
        plot_sample_distributions(sample_rows, protocol.RESULTS_DIR / MUSICAL_BOXPLOT_FILENAME,
                                  MUSICAL_BOXPLOT_METRICS, 2, 2)
    write_markdown(models, samples, protocol.RESULTS_DIR / "comparacao.md")
    summary = {
        "models": {key: {k: v for k, v in model.items() if k != "history"} for key, model in models.items()},
        "samples": samples,
    }
    (protocol.RESULTS_DIR / "comparacao.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(f"Comparação em {protocol.RESULTS_DIR / 'comparacao.md'}")

if __name__ == "__main__":
    main()
