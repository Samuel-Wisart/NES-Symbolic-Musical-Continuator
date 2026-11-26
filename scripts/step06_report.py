"""
step06_report

Gera a figura de loss por época e o README de resultados de um modelo
(Transformer ou LSTM) a partir dos JSONs salvos pelo treino (step03), pela
avaliação (step04) e pela geração (step05).

    python scripts/step06_report.py --model transformer
    python scripts/step06_report.py --model lstm

Entradas (em resultados/<modelo>/, copiadas pelo step04):
    train_history.json, run_info.json, test_metrics.json
    ../amostras/<modelo>/summary.json (opcional)
Saídas:
    resultados/<modelo>/curvas_loss.png
    resultados/<modelo>/README.md
"""

### Imports
from pathlib import Path
import argparse, json, logging, math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from protocol import GENERATED_DIR, MODELS, REFERENCE_NAME, RESULTS_DIR

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

### Nomes
MODEL_NAMES = {"transformer": "Transformer", "lstm": "LSTM"}

### Estilo da figura
SERIES = {"train": "#2a78d6", "valid": "#eb6834"}
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

### Leitura
def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

### Figura
def plot_losses(history, output, name):
    """Loss de treino e validação (nats por token) por época, com a melhor época marcada."""
    epochs = [row["epoch"] for row in history]
    train = [row["train_loss"] for row in history]
    valid = [row["valid"]["loss"] for row in history]
    best = min(history, key=lambda row: row["valid"]["loss"])

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10})
    figure, axis = plt.subplots(figsize=(7, 4.2), dpi=160)
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)

    for series, values, label in (("train", train, "treino"), ("valid", valid, "validação")):
        axis.plot(epochs, values, color=SERIES[series], linewidth=2, marker="o" if len(epochs) <= 30 else None,
                  markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        # Rótulo direto no fim da linha, em tinta de texto.
        axis.annotate(f"{label} {values[-1]:.3f}", (epochs[-1], values[-1]), xytext=(8, 0),
                      textcoords="offset points", va="center", color=INK_SECONDARY, fontsize=9)

    axis.axvline(best["epoch"], color=INK_MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    axis.annotate(f"melhor validação: época {best['epoch']}", (best["epoch"], 1), xycoords=("data", "axes fraction"),
                  xytext=(-5, -2), textcoords="offset points", ha="right", va="top", color=INK_MUTED, fontsize=8.5)

    axis.set_xlabel("época", color=INK_SECONDARY)
    axis.set_ylabel("entropia cruzada (nats por token)", color=INK_SECONDARY)
    axis.set_title(f"{name}: loss por época", loc="left", color=INK_PRIMARY, fontsize=12, pad=28)
    axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(AXIS)
    axis.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2,
                handlelength=1.8, labelcolor=INK_SECONDARY)
    axis.margins(x=0.08)

    figure.text(0.01, 0.01, "Treino: média da época (com dropout). Validação: ao fim de cada época.",
                color=INK_MUTED, fontsize=8)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(output, facecolor=SURFACE)
    plt.close(figure)

### README
LIMITATIONS = """\
- **Sobreposição na mesma voz.** No step01, notas mais curtas que meio passo (1/128 s)
  são esticadas para 1 passo e passam a se sobrepor à nota seguinte da mesma voz:
  13,8% dos NOTE_ON do treino ocorrem com a voz já soando. O modelo aprende esse
  padrão, mas a máscara de geração o proíbe (as vozes do NES são monofônicas). A
  perplexidade inclui esses tokens; o Markov é medido sobre os mesmos dados.
- **Representação sem dinâmica nem timbre.** Os tokens guardam só altura e tempo
  (64 passos/s). Volume, duty cycle e envelopes do NES-MDB se perdem; o WAV é uma
  prévia chiptune aproximada.
- **Contexto limitado.** {context} tokens equivalem a cerca de {context_seconds:.0f} s
  de música (~60 tokens/s no treino). Estrutura de prazo mais longo (repetições de
  seção, forma) fica fora do alcance do modelo.
- **Perplexidade por token.** Depende da tokenização; só é comparável com modelos
  que usam exatamente o mesmo vocabulário de 1155 tokens e a mesma definição.
- **Acurácia top-1.** Mistura tipos de token (TIME_SHIFT, NOTE_ON, NOTE_OFF) e só
  conta o acerto exato do próximo token; não mede a qualidade musical da geração.
"""

def inteiro(number):
    """435683 -> '435.683'."""
    return f"{number:,}".replace(",", ".")

def decimal(number, digits):
    """4.6007 -> '4,6007'."""
    return f"{number:.{digits}f}".replace(".", ",")

def porcentagem(fraction):
    """0.1113 -> '11,13%'."""
    return decimal(100 * fraction, 2) + "%"

def fmt_duration(seconds):
    hours, rest = divmod(round(seconds), 3600)
    return f"{hours} h {rest // 60:02d} min" if hours else f"{rest // 60} min {rest % 60:02d} s"

def architecture(config):
    """Descrição da arquitetura a partir do model_config salvo no checkpoint."""
    if config.get("arch", "transformer") == "lstm":
        return (
            f"LSTM empilhado (PyTorch nativo), com {config['layers']} camadas de largura {config['width']} "
            f"(embedding e estado), dropout entre camadas, LayerNorm e projeção de saída compartilhada com o "
            f"embedding de entrada. Treinado e avaliado nas mesmas janelas de {config['context']} tokens do "
            "Transformer (o estado começa zerado em cada janela)."
        )
    return (
        f"Transformer somente decodificador (PyTorch nativo), com {config['layers']} blocos pre-LN, "
        f"largura {config['width']}, {config['heads']} cabeças de atenção causal, feed-forward de "
        f"{4 * config['width']} (GELU), embeddings de posição aprendidos e projeção de saída "
        f"compartilhada com o embedding de entrada. Contexto de {config['context']} tokens."
    )

def write_readme(model, history, run_info, test, generation, output):
    config = run_info["model_config"]
    run = run_info["run_config"]
    best = min(history, key=lambda row: row["valid"]["loss"])
    train_seconds = sum(row["seconds"] for row in history)
    sliding, disjoint = test["sliding_windows"], test["disjoint_windows"]

    lines = [
        f"# {MODEL_NAMES[model]}: resultados",
        "",
        f"Gerado por `python scripts/step06_report.py --model {model}` a partir de `train_history.json`,",
        "`run_info.json`, `test_metrics.json` e do resumo da geração.",
        "",
        "## Arquitetura",
        "",
        architecture(config),
        "",
        f"- Vocabulário: 1155 tokens do step01 (MIDI-like, 64 passos/s, 4 vozes)",
        f"- Parâmetros: {inteiro(run_info['parameters'])}",
        "",
        "## Treino",
        "",
        "| Hiperparâmetro | Valor |",
        "|---|---|",
        f"| Otimizador | AdamW, lr {run['lr']:g}, weight decay 0.01, clipping 1.0 |",
        f"| Lote | {run['batch_size']} janelas de até {config['context']} tokens |",
        f"| Janelas de treino | passo {config['context'] // 2} (contexto / 2), {inteiro(run_info['train']['windows'])} janelas |",
        f"| Dropout | {config['dropout']} |",
        f"| Precisão | {run_info['precision']} |",
        f"| Semente | {run['seed']} |",
        f"| Épocas | {len(history)} |",
        f"| Tempo de treino | {fmt_duration(train_seconds)} ({fmt_duration(train_seconds / len(history))} por época) |",
        f"| Hardware | {run_info['device']} (PyTorch {run_info['torch']}) |",
        "",
        f"Dados: partições oficiais do NES-MDB, {run_info['train']['songs']} músicas de treino "
        f"({inteiro(run_info['train']['targets'])} tokens-alvo) e {run_info['valid']['songs']} de validação.",
        "",
        f"**Melhor época (menor loss de validação): {best['epoch']}**, com loss {decimal(best['valid']['loss'], 4)}, "
        f"perplexidade {decimal(best['valid']['perplexity'], 2)} e acurácia {porcentagem(best['valid']['accuracy'])}. "
        "O modelo entregue (`modelo.pt`) é o checkpoint dessa época.",
        "",
        "![Loss por época](curvas_loss.png)",
        "",
        "## Teste",
        "",
        "Perplexidade = exp(média da entropia cruzada, em log natural, de todos os tokens depois do BOS "
        f"em cada música de teste): {inteiro(test['tokens'])} tokens de {sliding['songs']} músicas.",
        "",
        "| Medida | Loss (nats/token) | Perplexidade | Acurácia top-1 |",
        "|---|---|---|---|",
        f"| **Janelas deslizantes** (passo {sliding['stride']}) | **{decimal(sliding['loss'], 4)}** | **{decimal(sliding['perplexity'], 2)}** | **{porcentagem(sliding['accuracy'])}** |",
        f"| Janelas disjuntas (passo {disjoint['stride']}) | {decimal(disjoint['loss'], 4)} | {decimal(disjoint['perplexity'], 2)} | {porcentagem(disjoint['accuracy'])} |",
        "",
        "A medida principal é a de janelas deslizantes: cada token é pontuado uma vez, com pelo menos "
        "meio contexto de histórico. A de janelas disjuntas segue o mesmo método da validação no treino.",
        "",
    ]

    if generation:
        protocol = generation["protocol"]
        songs = generation["songs"]
        tokens = [song["generated_tokens"] for song in songs]
        lines += [
            "## Geração",
            "",
            f"{len(songs)} continuações (protocolo em `scripts/protocol.py`): prompt de {protocol['prompt_seconds']} s "
            f"de uma música de teste, continuação de {protocol['gen_seconds']} s, temperatura {protocol['temperature']}, "
            f"top-k {protocol['top_k']}, máscara de eventos válidos. Em média {sum(tokens) / len(tokens):.0f} tokens "
            f"por continuação (mín. {min(tokens)}, máx. {max(tokens)}). Arquivos em `resultados/amostras/{model}/`; "
            f"referência real dos mesmos segundos em `resultados/amostras/{REFERENCE_NAME}/`.",
            "",
        ]

    lines += [
        "## Limitações",
        "",
        LIMITATIONS.format(context=config["context"], context_seconds=config["context"] / 60).rstrip(),
        "",
        "### Observações de escuta",
        "",
        "_A preencher depois de ouvir as amostras de `resultados/amostras/` (coerência com o prompt,",
        "repetição, uso das quatro vozes, ritmo)._",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")

### Execução
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True, help="modelo do relatório")
    parser.add_argument("--results-dir", type=Path, help="padrão: resultados/<modelo>/")
    parser.add_argument("--generation-summary", type=Path, help="padrão: resultados/amostras/<modelo>/summary.json")
    args = parser.parse_args()
    args.results_dir = args.results_dir or RESULTS_DIR / args.model
    args.generation_summary = args.generation_summary or GENERATED_DIR / args.model / "summary.json"

    history = read_json(args.results_dir / "train_history.json")
    run_info = read_json(args.results_dir / "run_info.json")
    test = read_json(args.results_dir / "test_metrics.json")
    if not (history and run_info and test):
        raise SystemExit(f"Faltam JSONs em {args.results_dir}. Rode antes o step04 (que copia os do treino).")
    if any(not math.isfinite(row["valid"]["loss"]) for row in history):
        raise SystemExit("O histórico tem loss não finita.")

    plot_losses(history, args.results_dir / "curvas_loss.png", MODEL_NAMES[args.model])
    write_readme(args.model, history, run_info, test, read_json(args.generation_summary), args.results_dir / "README.md")
    print(f"Figura e README em {args.results_dir}")

if __name__ == "__main__":
    main()
