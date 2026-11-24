#!/bin/bash
# Teste de retomada na CPU (segundos, dados mínimos; pode rodar na phocus4; ARCH=transformer|lstm):
#   ARCH=lstm bash scripts/slurm/comum/teste_retomada.sh
# Treina um modelo minúsculo duas vezes: (a) sem interrupção; (b) parado pelo PRAZO_EPOCH logo
# no primeiro lote e retomado com o mesmo comando. As losses de (a) e (b) têm que ser iguais.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../../.."
PY=${PY:-.venv/bin/python}
unset PYTHONHOME PYTHONPATH
BASE=runs/teste_retomada_${ARCH:-transformer}
rm -rf "$BASE"
mkdir -p runs
ARGS=(-m nes_continuation train --arch "${ARCH:-transformer}" --device cpu --threads 2 --train-limit 20 --valid-limit 2
      --context 64 --width 32 --heads 4 --layers 2 --dropout 0.1 --batch-size 8 --epochs 2 --log-every 1000)

"$PY" "${ARGS[@]}" --run-dir "$BASE/a" > "$BASE.a.log" 2>&1 || { mkdir -p "$BASE"; cat "$BASE.a.log"; exit 1; }
PRAZO_EPOCH=$(date +%s) "$PY" "${ARGS[@]}" --run-dir "$BASE/b" > "$BASE.b1.log" 2>&1
grep -q "\[parada\]" "$BASE.b1.log" || { echo "FALHOU: sem [parada]"; cat "$BASE.b1.log"; exit 1; }
"$PY" "${ARGS[@]}" --run-dir "$BASE/b" > "$BASE.b2.log" 2>&1
grep -q "\[retomada\] época 1, lote 1 " "$BASE.b2.log" || { echo "FALHOU: sem [retomada]"; cat "$BASE.b2.log"; exit 1; }
"$PY" "${ARGS[@]}" --run-dir "$BASE/b" > "$BASE.b3.log" 2>&1
grep -q "já concluído" "$BASE.b3.log" || { echo "FALHOU: terceira execução não disse 'já concluído'"; exit 1; }
"$PY" - "$BASE" <<'PY'
import json, math, sys
from pathlib import Path
base = Path(sys.argv[1])
a = json.loads((base / "a" / "metrics.json").read_text())
b = json.loads((base / "b" / "metrics.json").read_text())
for x, y in zip(a, b, strict=True):
    for key in ("train_loss",):
        assert math.isclose(x[key], y[key], rel_tol=1e-9), (x[key], y[key])
    assert math.isclose(x["valid"]["loss"], y["valid"]["loss"], rel_tol=1e-9), (x["valid"], y["valid"])
print("retomada OK:", [(row["epoch"], round(row["train_loss"], 6), round(row["valid"]["loss"], 6)) for row in b])
PY
rm -rf "$BASE" "$BASE".*.log
