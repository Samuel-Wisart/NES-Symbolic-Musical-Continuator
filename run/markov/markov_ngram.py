"""
markov_ngram

Cadeia de Markov de ordem N (contexto de N tokens) com backoff para ordens
menores, amostragem top-p e as mesmas restrições musicais do generate_markov
(NOTE_OFF só para notas abertas, canais monofônicos, fechamento no fim).

Uso:
    python markov_ngram.py train --order 4
    python markov_ngram.py generate --order 4 --temperature 0.9 --top-p 0.9

O modelo é guardado de forma esparsa: cada par (contexto, próximo token) vira
uma chave inteira, e as contagens ficam em arrays ordenados consultados por
busca binária. Isso evita dicionários gigantes de tuplas.
"""

### Imports
import argparse
from pathlib import Path

import numpy as np

from prepareData_markov import decode, TO_ID, VOCAB
from generate_markov import (
    BASE_VALID, TOKEN_INFO, EOS_ID, BOS_ID, OUTPUT_DIR, PROCESSED,
    _off_id, notas_para_midi,
)

### Configuração
V = len(VOCAB)
MAX_ORDER = 5  # V**(order+1) precisa caber em int64
TOKENS = PROCESSED / "tokens"


### Caminhos
def model_path(order):
    return PROCESSED / f"ngram_order{order}.npz"

### Modelo
class NGram:
    def __init__(self, niveis):
        """niveis: {k: (chaves_ordenadas, contagens)} para k = 1..order."""
        self.order = max(niveis)
        self.niveis = {}
        for k, (pk, cnt) in niveis.items():
            ctx_unicos, inicio = np.unique(pk // V, return_index=True)
            fim = np.append(inicio[1:], len(pk))
            cnt = cnt.astype(np.int64)
            self.niveis[k] = dict(
                pk=pk, cnt=cnt, ctx=ctx_unicos, ini=inicio, fim=fim,
                tot=np.add.reduceat(cnt, inicio),
            )

    @classmethod
    def treinar(cls, seqs, order):
        niveis = {}
        for k in range(1, order + 1):
            chaves = []
            for ids in seqs:
                n = len(ids) - k
                if n <= 0:
                    continue
                ctx = np.zeros(n, dtype=np.int64)
                for j in range(k):
                    ctx = ctx * V + ids[j:j + n]
                chaves.append(ctx * V + ids[k:])
            pk, cnt = np.unique(np.concatenate(chaves), return_counts=True)
            niveis[k] = (pk, cnt)
            print(f"  contexto {k}: {len(pk):,} transições distintas")
        return cls(niveis)

    def salvar(self, path):
        arrays = {}
        for k, n in self.niveis.items():
            arrays[f"pk{k}"] = n["pk"]
            arrays[f"cnt{k}"] = n["cnt"].astype(np.int32)
        np.savez_compressed(path, **arrays)

    @classmethod
    def carregar(cls, path):
        d = np.load(path)
        ks = sorted(int(n[2:]) for n in d.files if n.startswith("pk"))
        return cls({k: (d[f"pk{k}"], d[f"cnt{k}"]) for k in ks})

    def candidatos(self, k, contexto):
        """(tokens, contagens) observados após `contexto`, ou None."""
        nv = self.niveis[k]
        chave = 0
        for t in contexto:
            chave = chave * V + int(t)
        i = np.searchsorted(nv["ctx"], chave)
        if i == len(nv["ctx"]) or nv["ctx"][i] != chave:
            return None
        sl = slice(nv["ini"][i], nv["fim"][i])
        return nv["pk"][sl] % V, nv["cnt"][sl]

    def log_probs(self, ids, ordem, alpha):
        """Log-prob de cada token via stupid backoff (não normalizado)."""
        n = len(ids)
        logp = np.full(n, np.log(alpha ** ordem / V))
        resolvido = np.zeros(n, dtype=bool)
        resolvido[0] = True  # BOS não é predito

        for k in range(ordem, 0, -1):
            m = n - k
            if m <= 0:
                continue
            ctx = np.zeros(m, dtype=np.int64)
            for j in range(k):
                ctx = ctx * V + ids[j:j + m]
            par = ctx * V + ids[k:]

            nv = self.niveis[k]
            ip = np.minimum(np.searchsorted(nv["pk"], par), len(nv["pk"]) - 1)
            achou = nv["pk"][ip] == par
            ic = np.minimum(np.searchsorted(nv["ctx"], ctx), len(nv["ctx"]) - 1)
            p = nv["cnt"][ip] / nv["tot"][ic] * alpha ** (ordem - k)

            pos = np.arange(k, n)
            usar = achou & ~resolvido[pos]
            logp[pos[usar]] = np.log(p[usar])
            resolvido[pos[usar]] = True

        return logp[1:]

    def perplexidade(self, seqs, ordem=None, alpha=0.4):
        ordem = ordem or self.order
        total, n = 0.0, 0
        for ids in seqs:
            if len(ids) < 2:
                continue
            lp = self.log_probs(ids, ordem, alpha)
            total += lp.sum()
            n += len(lp)
        return float(np.exp(-total / n)) if n else float("nan")

### Amostragem
def _distribuicao(modelo, seq, valido, min_count):
    """Maior contexto com suporte suficiente. Sem suporte, faz backoff."""
    for k in range(min(modelo.order, len(seq)), 0, -1):
        c = modelo.candidatos(k, seq[-k:])
        if c is None:
            continue
        toks, cnts = c
        ok = valido[toks]
        minimo = min_count if k > 1 else 1
        if ok.any() and cnts[ok].sum() >= minimo:
            return toks[ok], cnts[ok].astype(np.float64)
    return None

def _escolher(toks, pesos, temperature, top_p, rng):
    logits = np.log(pesos) / temperature
    logits -= logits.max()
    w = np.exp(logits)
    w /= w.sum()

    if top_p < 1.0:
        ordem = np.argsort(-w)
        acum = np.cumsum(w[ordem])
        corte = int(np.searchsorted(acum, top_p)) + 1
        manter = ordem[:corte]
        w2 = np.zeros_like(w)
        w2[manter] = w[manter]
        w = w2 / w2.sum()

    return int(rng.choice(toks, p=w))

def gerar(modelo, n_max=2000, min_tokens=200, temperature=1.0,
          top_p=0.9, min_count=2, rng=None):
    if temperature <= 0:
        raise ValueError("temperature deve ser > 0")
    rng = rng or np.random.default_rng()

    seq = [BOS_ID]
    abertas = {}  # canal -> pitch ligado

    while len(seq) < n_max:
        valido = BASE_VALID.copy()
        for canal, pitch in abertas.items():
            valido[_off_id(canal, pitch)] = True
        if len(seq) < min_tokens:
            valido[EOS_ID] = False

        dist = _distribuicao(modelo, seq, valido, min_count)
        if dist is None:
            prox = TO_ID["TIME_SHIFT_8"]  # nenhum contexto serve: avança o tempo
        else:
            prox = _escolher(*dist, temperature, top_p, rng)

        if prox == EOS_ID:
            break

        info = TOKEN_INFO.get(prox)
        if info is not None:
            estado, canal, pitch = info
            if estado == "ON":
                if canal in abertas:
                    seq.append(_off_id(canal, abertas.pop(canal)))
                abertas[canal] = pitch
            else:
                abertas.pop(canal)

        seq.append(prox)

    if abertas:
        seq.append(TO_ID["TIME_SHIFT_16"])
        for canal, pitch in abertas.items():
            seq.append(_off_id(canal, pitch))

    seq.append(EOS_ID)
    return seq

### Dados para avaliação
def carregar_split(nome):
    pasta = TOKENS / nome
    seqs = []
    for p in sorted(pasta.glob("*.npz")):
        with np.load(p) as d:
            seqs.append(d["ids"].astype(np.int64))
    if not seqs:
        raise FileNotFoundError(f"Nenhum .npz em {pasta}")
    return seqs

### Treinamento e geração
def cmd_train(args):
    if not 1 <= args.order <= MAX_ORDER:
        raise ValueError(f"--order deve estar entre 1 e {MAX_ORDER}")

    train = carregar_split("train")
    print(f"Treinando ordem {args.order} em {len(train)} músicas")
    modelo = NGram.treinar(train, args.order)

    valid_dir = TOKENS / "valid"
    if valid_dir.exists() and any(valid_dir.glob("*.npz")):
        valid = carregar_split("valid")
        print("Perplexidade na validação (stupid backoff, só comparativa):")
        for o in range(1, args.order + 1):
            print(f"  ordem {o}: {modelo.perplexidade(valid, ordem=o):.2f}")

    modelo.salvar(model_path(args.order))
    print(f"Modelo salvo em {model_path(args.order)}")

### Geração por CLI
def cmd_generate(args):
    modelo = NGram.carregar(model_path(args.order))
    rng = np.random.default_rng(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_samples):
        ids = gerar(modelo, args.n_max, args.min_tokens, args.temperature,
                    args.top_p, args.min_count, rng)
        notes = decode(ids)
        if not notes:
            print(f"[{i:03d}] sequência sem notas, pulando")
            continue
        out = OUTPUT_DIR / f"ngram{args.order}_T{args.temperature:.2f}_{i:03d}.mid"
        notas_para_midi(notes, out)
        print(f"[{i:03d}] {len(ids):4d} tokens → {len(notes):4d} notas → {out.name}")


### Execução
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--order", type=int, default=4)
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("generate")
    g.add_argument("--order", type=int, default=4)
    g.add_argument("--n-samples", type=int, default=10)
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--top-p", type=float, default=0.9)
    g.add_argument("--min-count", type=int, default=2)
    g.add_argument("--n-max", type=int, default=2000)
    g.add_argument("--min-tokens", type=int, default=200)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    args.func(args)
