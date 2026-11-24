#!/bin/bash
# Testes e benchmark de um modelo num nó (rodado pelo job.sbatch, que já prepara o ambiente e
# extrai o dataset no disco local; MODELO vem do ambiente):
#   MODELO=lstm sbatch --partition=gorgonas_dev --time=00:40:00 --job-name=nes-lstm-teste \
#       scripts/slurm/comum/job.sbatch scripts/slurm/comum/teste_dev.sh
# 1. test01, test02 e test03 (com os MIDIs extraídos no disco local do nó);
# 2. mede o tempo por época do modelo com lotes de 8, 16, 32 e 64.
set -euo pipefail
PY="$SLURM_SUBMIT_DIR/.venv/bin/python"
MODELO=${MODELO:-transformer}
source scripts/slurm/comum/midi_local.sh
midi_local_extrair

for teste in tests/test03_lstm_tests.py tests/test02_transformer_tests.py; do
    echo "[teste] $teste"
    "$PY" -u "$teste" || echo "[teste] $teste FALHOU"
done
echo "[teste] tests/test01_dataset_tests.py"
PYTHONPATH=scripts "$PY" -u tests/test01_dataset_tests.py || echo "[teste] test01 FALHOU"

"$PY" -c "import torch; print('[teste] cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'bf16', torch.cuda.is_bf16_supported())"
for lote in 8 16 32 64; do
    echo "[teste] benchmark $MODELO lote $lote"
    "$PY" -u "scripts/step03_train_$MODELO.py" --benchmark 50 --batch-size "$lote"
done
echo "[teste] fim"
