#!/bin/bash
# Prepara o dataset (step01) num nó (rodado pelo job.sbatch, que já prepara o ambiente):
#   sbatch --partition=gorgonas_dev --time=01:00:00 --job-name=nes-dados \
#       scripts/slurm/comum/job.sbatch scripts/slurm/comum/preparar_dados.sh
# Só precisa de numpy e pretty_midi. O step01 grava tudo no disco local do nó (MIDIs em
# midi_local.sh, .npz em dados_locais.sh) e o resultado vai para o storage num único arquivo,
# data/nesmdb_processed.tar. O step01 não é retomável: o aviso de fim de job (USR1) é ignorado.
set -euo pipefail
PY="$SLURM_SUBMIT_DIR/.venv/bin/python"
source scripts/slurm/comum/midi_local.sh
source scripts/slurm/comum/dados_locais.sh

if [ -f "$DADOS_TAR" ]; then
    echo "[preparar] $DADOS_TAR já existe; nada a fazer"
    exit 0
fi
trap 'echo "[preparar] USR1 ignorado: o step01 não é retomável e segue até o fim do job"' USR1

midi_local_preparar
rm -rf "$DADOS_LOCAL"
mkdir -p "$DADOS_LOCAL"
dados_locais_ligar
echo "[preparar] step01 $(date '+%T')"
time "$PY" -u scripts/step01_prepare_dataset.py
for split in train valid test; do
    echo "[preparar] $split: $(ls "$DADOS_LOCAL/nesmdb/tokens/$split" | wc -l) .npz"
done
tar -cf "$DADOS_TAR.tmp" -C "$DADOS_LOCAL" nesmdb
mv "$DADOS_TAR.tmp" "$DADOS_TAR"
touch "$DADOS_LOCAL/.extraido-$(stat -c %s-%Y "$DADOS_TAR")"
echo "[preparar] $(ls -l "$DADOS_TAR") | fim $(date '+%T')"
