#!/bin/bash
# Job do Slurm (cluster SONIC) do projeto. Baseado no modelo da skill slurm.
#
# Sem argumentos, roda o treino do modelo em $MODELO (transformer ou lstm; o sbatch repassa as
# variáveis de ambiente): scripts/step03_train_$MODELO.py, que começa do zero ou continua do
# runs/$MODELO/last.pt sozinho. Com argumentos, roda `python -u <argumentos>` ou, se o primeiro
# terminar em .sh, `bash <argumentos>`. Normalmente quem submete é o pipeline de cada modelo
# (scripts/slurm/<modelo>/pipeline.sh); para submeter à mão, a partir da raiz do projeto:
#   mkdir -p logs                      # o Slurm não cria a pasta do --output
#   squeue -u $USER                    # tem que vir vazio: um job por vez
#   MODELO=lstm sbatch --partition=gorgonas_dev --time=01:30:00 --job-name=nes-lstm scripts/slurm/comum/job.sbatch
#   sbatch --partition=gorgonas_dev --time=00:30:00 scripts/slurm/comum/job.sbatch \
#       scripts/step04_evaluate.py --model transformer
# Opções na linha de comando valem mais que as #SBATCH abaixo. Use --time de pelo menos 20 min
# (o aviso de fim chega 10 min antes). Comentários ficam em linhas próprias, nunca no fim de uma
# linha #SBATCH.
#
#SBATCH --job-name=nes
# gorgonas: sem limite de tempo, então --time de no máximo 8 h é obrigatório. Nunca --gres lá.
#SBATCH --partition=gorgonas
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append
# Aviso (USR1) ao bash 10 min antes do fim; ele repassa ao programa, que salva e sai.
#SBATCH --signal=B:USR1@600

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1

PY="$SLURM_SUBMIT_DIR/.venv/bin/python"
# Treino completo (o mesmo comando em todos os jobs da cadeia). Épocas e hiperparâmetros vêm do
# step03 de cada modelo; TREINO_ARGS (config.sh do modelo) acrescenta opções. Mudar lote, taxa
# ou semente no meio da rodada faz o step03 recusar a retomada; --epochs pode aumentar.
MODELO=${MODELO:-transformer}
read -r -a TREINO_EXTRA <<< "${TREINO_ARGS:-}"
TREINO=(scripts/step03_train_"$MODELO".py ${TREINO_EXTRA[@]+"${TREINO_EXTRA[@]}"})
if [ $# -gt 0 ] && [[ "$1" == *.sh ]]; then
    COMANDO=(bash "$@")
elif [ $# -gt 0 ]; then
    COMANDO=("$PY" -u "$@")
else
    COMANDO=("$PY" -u "${TREINO[@]}")
fi

# Ambiente: o $HOME da phocus4 não existe nos nós; caches dentro do projeto; nada de rede.
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache"
export HOME="$XDG_CACHE_HOME/home" TORCH_HOME="$XDG_CACHE_HOME/torch"
export HF_HOME="$XDG_CACHE_HOME/hf" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$HOME" "$TORCH_HOME" "$HF_HOME" "$TRITON_CACHE_DIR"

# Prazo interno: 10 min antes do fim real do job (acompanha o --time usado na submissão).
FIM_JOB="${SLURM_JOB_END_TIME:-}"
if [ -z "$FIM_JOB" ]; then
    FIM_TEXTO=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)
    [ -n "$FIM_TEXTO" ] && FIM_JOB=$(date -d "$FIM_TEXTO" +%s 2>/dev/null)
fi
[ -z "$FIM_JOB" ] && FIM_JOB=$(( $(date +%s) + 8 * 3600 ))
export PRAZO_EPOCH=$(( FIM_JOB - 10 * 60 ))

# Dataset processado: extraído do data/nesmdb_processed.tar no disco local do nó (dados_locais.sh).
source scripts/slurm/comum/dados_locais.sh
if dados_locais_extrair; then
    echo "[job] dados em $DADOS_LOCAL ($(ls "$DADOS_LOCAL/nesmdb/tokens/train" | wc -l) músicas de treino)"
else
    echo "[job] sem $DADOS_TAR (rode scripts/slurm/comum/preparar_dados.sh)"
fi

echo "[job] $SLURM_JOB_ID ($SLURM_JOB_NAME) em $(hostname) | partição $SLURM_JOB_PARTITION | início $(date '+%F %T')"
echo "[job] fim do job $(date -d "@$FIM_JOB" '+%F %T') | prazo interno $(date -d "@$PRAZO_EPOCH" '+%F %T')"
echo "[job] comando: ${COMANDO[*]}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || echo "[job] sem GPU visível"

# O programa roda em segundo plano para o bash poder repassar sinais: USR1 (aviso do Slurm) e
# TERM (scancel ou timeout; o SIGKILL vem 60 s depois) fazem o programa salvar e sair.
"${COMANDO[@]}" &
PID=$!
trap 'echo "[job] USR1: pedindo ao programa para salvar e sair"; kill -USR1 "$PID" 2>/dev/null' USR1
trap 'echo "[job] TERM: pedindo ao programa para salvar e sair"; kill -TERM "$PID" 2>/dev/null' TERM
RC=0
while true; do
    wait "$PID"; RC=$?
    kill -0 "$PID" 2>/dev/null || break   # o wait também volta quando chega um sinal
done
echo "[job] fim $(date '+%F %T') | código $RC"
exit "$RC"
