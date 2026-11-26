#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
ARCHIVE="$ROOT/resultados/markov_original_sem_prompt"
TEMP_SAMPLES="$ROOT/resultados/amostras/markov"
BACKUP="$(mktemp -d)"

cleanup() {
    rm -rf "$TEMP_SAMPLES"
    for name in comparacao.md comparacao.json curvas_validacao.png boxplots_amostras.png boxplots_musicais.png; do
        if [ -f "$BACKUP/$name" ]; then
            cp "$BACKUP/$name" "$ROOT/resultados/$name"
        else
            rm -f "$ROOT/resultados/$name"
        fi
    done
    rm -rf "$BACKUP"
}
trap cleanup EXIT

if [ ! -x "$PY" ]; then
    echo "ERRO: .venv não encontrado. Ative/crie o ambiente antes."
    exit 1
fi

if [ ! -f "$ROOT/data/raw/nesmdb_midi.tar.gz" ]; then
    echo "NES-MDB não encontrado; baixando..."
    "$PY" -m gdown \
        "https://drive.google.com/uc?id=1w2uo1Cmio4gz6nGUhZOtzF54kPkoKyo7" \
        -O "$ROOT/data/raw/nesmdb_midi.tar.gz"
fi

for name in comparacao.md comparacao.json curvas_validacao.png boxplots_amostras.png boxplots_musicais.png; do
    [ -f "$ROOT/resultados/$name" ] && cp "$ROOT/resultados/$name" "$BACKUP/$name"
done

echo "=== 1. Preparando dataset, se necessário ==="
if [ ! -f "$ROOT/data/processed/nesmdb/vocab.json" ]; then
    "$PY" "$ROOT/run/markov/prepareData_markov.py"
else
    echo "Dataset processado já existe; mantendo."
fi

echo "=== 2. Treinando Markov original (ordem 1) ==="
mkdir -p "$ARCHIVE"
"$PY" "$ROOT/run/markov/train_markov.py" | tee "$ARCHIVE/train.log"

echo "=== 3. Calculando perplexidade do Markov original ==="
"$PY" - <<'PY'
from pathlib import Path
import json, sys
import numpy as np

root = Path.cwd()
sys.path.insert(0, str(root / "run" / "markov"))

from train_markov import PROCESSED, MATRIX_PATH, SMOOTHING, perplexidade

data = np.load(MATRIX_PATH)
probs = data["probs"]
global_probs = data["global_probs"]

metrics = {
    "model": "markov_original_sem_prompt",
    "order": 1,
    "smoothing": SMOOTHING,
    "validation_perplexity": perplexidade(PROCESSED / "tokens" / "valid", probs, global_probs, SMOOTHING),
    "test_perplexity": perplexidade(PROCESSED / "tokens" / "test", probs, global_probs, SMOOTHING),
    "validation_unigram_perplexity": perplexidade(PROCESSED / "tokens" / "valid", probs, global_probs, 1.0),
    "test_unigram_perplexity": perplexidade(PROCESSED / "tokens" / "test", probs, global_probs, 1.0),
}

out = root / "resultados" / "markov_original_sem_prompt" / "metrics.json"
out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(metrics, indent=2, ensure_ascii=False))
PY

echo "=== 4. Gerando 100 amostras com o comportamento ORIGINAL (BOS, sem prompt) ==="
rm -rf "$TEMP_SAMPLES"
mkdir -p "$TEMP_SAMPLES"

"$PY" - <<'PY' | tee "resultados/markov_original_sem_prompt/generation.log"
from pathlib import Path
import json, sys
import numpy as np

root = Path.cwd()
sys.path.insert(0, str(root / "run" / "markov"))
sys.path.insert(0, str(root))

from generate_markov import MATRIX_PATH, amostrar, decode, notas_para_midi
from nes_continuation.audio import render_wav
from nes_continuation.representation import TIME_RESOLUTION_HZ, duration_steps

output = root / "resultados" / "amostras" / "markov"
data = np.load(MATRIX_PATH)
probs = data["probs"]
global_probs = data["global_probs"]

rng = np.random.default_rng(42)
rows = []

for i in range(100):
    ids = np.asarray(
        amostrar(
            probs,
            global_probs,
            n_max=2000,
            min_tokens=200,
            temperature=1.0,
            rng=rng,
        ),
        dtype=np.uint16,
    )
    notes = decode(ids)
    if not notes:
        raise RuntimeError(f"Amostra {i} não contém notas.")

    stem = f"sample_T1.00_{i:03d}"
    midi_path = output / f"{stem}.mid"
    notas_para_midi(notes, midi_path)

    # BOS é apenas o estado inicial; toda a música depois dele é tratada como geração.
    np.savez_compressed(
        output / f"{stem}.npz",
        ids=ids,
        prompt_length=1,
    )

    if i < 5:
        render_wav(midi_path, output / f"{stem}.wav")

    seconds = duration_steps(ids[1:]) / TIME_RESOLUTION_HZ
    rows.append({
        "name": stem,
        "seed_stream": 42,
        "temperature": 1.0,
        "n_max": 2000,
        "min_tokens": 200,
        "tokens": len(ids),
        "seconds": seconds,
    })
    print(f"[{i + 1:03d}/100] {stem}: {len(ids)} tokens | {seconds:.2f} s")

(output / "summary.json").write_text(
    json.dumps(
        {
            "model": "markov_original_sem_prompt",
            "description": "Implementação original: geração iniciada em BOS, sem prompt musical.",
            "order": 1,
            "seed": 42,
            "temperature": 1.0,
            "n_samples": 100,
            "n_max": 2000,
            "min_tokens": 200,
            "songs": rows,
        },
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)
PY

echo "=== 5. Rodando step07 com o Markov original ==="
"$PY" "$ROOT/scripts/step07_compare.py"

echo "=== 6. Arquivando resultados e gráficos ==="
rm -rf "$ARCHIVE/amostras" "$ARCHIVE/analise"
mkdir -p "$ARCHIVE/amostras" "$ARCHIVE/analise"

cp -a "$TEMP_SAMPLES/." "$ARCHIVE/amostras/"
cp "$ROOT/data/processed/nesmdb/markov_order1.npz" "$ARCHIVE/markov_order1.npz"
git rev-parse HEAD > "$ARCHIVE/COMMIT.txt"

for name in comparacao.md comparacao.json curvas_validacao.png boxplots_amostras.png boxplots_musicais.png; do
    [ -f "$ROOT/resultados/$name" ] && cp "$ROOT/resultados/$name" "$ARCHIVE/analise/$name"
done

cat > "$ARCHIVE/README.md" <<'EOF'
# Markov original — sem prompt

Snapshot da implementação original do Markov antes da integração com o protocolo
de continuação musical.

Esta versão:

- usa cadeia de Markov de ordem 1;
- usa a representação MIDI-Like do NES-MDB;
- inicia a geração em `BOS`;
- não recebe os 4 segundos de prompt do Transformer/LSTM;
- usa temperatura 1.0, `min_tokens=200`, `n_max=2000` e seed 42;
- preserva 100 amostras para comparação histórica.

`amostras/` contém MIDI e NPZ das 100 gerações e WAV das 5 primeiras.
`analise/` contém a comparação e os boxplots produzidos pelo `step07_compare.py`.
`metrics.json` contém perplexidade de validação/teste do Markov e do unigrama.

Os `.npz` desta pasta foram criados apenas como adaptador de análise. Como esta
versão não usa prompt musical, `prompt_length=1` representa somente o token BOS.

Estes resultados não devem ser confundidos com a versão final condicionada por
prompt.
EOF

MIDIS=$(find "$ARCHIVE/amostras" -maxdepth 1 -name '*.mid' | wc -l)
NPZS=$(find "$ARCHIVE/amostras" -maxdepth 1 -name '*.npz' | wc -l)
WAVS=$(find "$ARCHIVE/amostras" -maxdepth 1 -name '*.wav' | wc -l)

echo
echo "=== SNAPSHOT CONCLUÍDO ==="
echo "MIDI: $MIDIS"
echo "NPZ:  $NPZS"
echo "WAV:  $WAVS"
echo "Pasta: resultados/markov_original_sem_prompt/"
echo
echo "Os arquivos globais de resultados serão restaurados ao sair."
