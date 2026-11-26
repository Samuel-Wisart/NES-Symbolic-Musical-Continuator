"""
train_markov

Treina uma cadeia de Markov de ordem 1 sobre os tokens MIDI-Like
produzidos pelo step01. Salva a matriz de transição normalizada
e a distribuição global (fallback para linhas vazias).

Também mede a perplexidade no split de validação, comparada a um
baseline unigrama, para dar uma métrica objetiva do modelo.
"""

### Imports
from pathlib import Path
import json
import numpy as np

### Caminhos
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "nesmdb"
TOKENS_DIR = PROCESSED / "tokens" / "train"
VALID_DIR = PROCESSED / "tokens" / "valid"
MATRIX_PATH = PROCESSED / "markov_order1.npz"

### Suavização
# Peso do unigrama ao avaliar (evita probabilidade zero em transições não vistas)
SMOOTHING = 0.01

### Leitura dos tokens
def carregar_ids(pasta):
    """Itera sobre todas as sequências de IDs em um split."""
    for npz_path in sorted(pasta.glob("*.npz")):
        with np.load(npz_path) as data:
            yield data["ids"]

### Contagem de transições
def contar_transicoes(pasta, vocab_size):
    """Conta transições a->b de forma vetorizada."""
    counts = np.zeros(vocab_size * vocab_size, dtype=np.float64)
    n_musicas = 0

    for ids in carregar_ids(pasta):
        n_musicas += 1
        if len(ids) < 2:
            continue
        # Os ids vêm em uint16: converter antes de multiplicar, senão estoura
        ids = ids.astype(np.int64)
        pares = ids[:-1] * vocab_size + ids[1:]
        counts += np.bincount(pares, minlength=vocab_size * vocab_size)

    if n_musicas == 0:
        raise FileNotFoundError(f"Nenhum .npz encontrado em {pasta}")

    return counts.reshape(vocab_size, vocab_size), n_musicas

### Treinamento
def treinar_order1(pasta, vocab_size):
    counts, n_musicas = contar_transicoes(pasta, vocab_size)

    totais = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, totais, where=totais > 0,
                      out=np.zeros_like(counts))

    # Frequência marginal de cada token (usada se uma linha estiver vazia)
    global_probs = counts.sum(axis=0)
    if global_probs.sum() > 0:
        global_probs /= global_probs.sum()

    return probs, global_probs, n_musicas

### Avaliação
def perplexidade(pasta, probs, global_probs, peso_unigrama):
    """Perplexidade por token. peso_unigrama=1.0 equivale ao baseline unigrama."""
    log_lik, n = 0.0, 0

    for ids in carregar_ids(pasta):
        if len(ids) < 2:
            continue
        a = ids[:-1].astype(np.int64)
        b = ids[1:].astype(np.int64)
        p = (1 - peso_unigrama) * probs[a, b] + peso_unigrama * global_probs[b]
        log_lik += np.log(np.maximum(p, 1e-12)).sum()
        n += len(a)

    return float(np.exp(-log_lik / n)) if n else float("nan")

### Execução
def main():
    vocab = json.loads((PROCESSED / "vocab.json").read_text(encoding="utf-8"))
    vocab_size = len(vocab)
    print(f"Vocabulário: {vocab_size} tokens")

    probs, global_probs, n_musicas = treinar_order1(TOKENS_DIR, vocab_size)
    print(f"Treinado em {n_musicas} músicas")

    if VALID_DIR.exists() and any(VALID_DIR.glob("*.npz")):
        ppl_markov = perplexidade(VALID_DIR, probs, global_probs, SMOOTHING)
        ppl_unigrama = perplexidade(VALID_DIR, probs, global_probs, 1.0)
        print(f"Perplexidade (validação): Markov={ppl_markov:.2f} | "
              f"unigrama={ppl_unigrama:.2f}")

    np.savez_compressed(
        MATRIX_PATH,
        probs=probs.astype(np.float32),
        global_probs=global_probs.astype(np.float32),
        vocab_size=vocab_size,
    )
    print(f"Matriz salva em {MATRIX_PATH}")


if __name__ == "__main__":
    main()
