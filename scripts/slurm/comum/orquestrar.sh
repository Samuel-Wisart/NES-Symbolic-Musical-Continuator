#!/bin/bash
# Orquestrador (cluster SONIC): cadeia de jobs de treino de um modelo, UM job por vez. Baseado no
# modelo da skill slurm. Normalmente é chamado pelo pipeline do modelo; para rodar só a cadeia:
#
#   mkdir -p logs
#   setsid nohup bash scripts/slurm/comum/orquestrar.sh transformer > logs/orquestrador-transformer.out 2>&1 < /dev/null &
#
# Submete um job, espera ele terminar (sbatch --wait), confere se houve progresso, faz uma pausa
# para outros alunos pegarem a máquina e só então submete o próximo, até o treino gravar
# runs/<modelo>/CONCLUIDO. Partição, tempo por job e nome do job vêm de
# scripts/slurm/<modelo>/config.sh.
#
# Parar sem perder nada: `touch PARAR` na raiz do projeto (não submete o próximo; o job em
# andamento vai até o fim dele). Para encerrar também o job em andamento, salvando antes:
# `scancel --signal=USR1 --batch <id>`.
# Rodar o mesmo comando de novo retoma a cadeia (se já houver job seu na fila, espera ele sair).
#
# Garantias: um orquestrador por modelo (trava em logs/) e um job por vez somando todos os
# projetos do usuário (trava global em ~/.slurm_um_job_por_vez.lock + `squeue -u $USER` vazio).
set -uo pipefail
# Raiz do projeto (este arquivo em <raiz>/scripts/slurm/comum/).
cd "$(dirname "$(readlink -f "$0")")/../../.." || exit 1
MODELO=${1:?uso: orquestrar.sh <transformer|lstm>}
source "scripts/slurm/$MODELO/config.sh" || exit 1
export MODELO TREINO_ARGS

JOB_SCRIPT=scripts/slurm/comum/job.sbatch
# Opções extras do sbatch, ex.: (--partition=medusas_dev --gres=gpu:1 --cpus-per-task=8 --mem=32G)
SBATCH_ARGS=(--partition="$PARTICAO" --time="$TEMPO_JOB" --job-name="$JOB")
# Gravado pelo programa quando não há mais nada a fazer.
MARCA_FIM=runs/$MODELO/CONCLUIDO
# Muda a cada checkpoint; serve para detectar job que não avança.
PROGRESSO=runs/$MODELO/progresso.json
MAX_JOBS=${MAX_JOBS:-10}                    # teto de jobs desta cadeia
MAX_SEM_PROGRESSO=${MAX_SEM_PROGRESSO:-2}   # jobs seguidos sem progresso antes de desistir
PAUSA_ENTRE_JOBS=${PAUSA_ENTRE_JOBS:-300}   # folga entre um job e o próximo (vez dos outros)
INTERVALO_FILA=${INTERVALO_FILA:-600}       # espera quando já há job seu na fila

mkdir -p logs
LOG=logs/orquestrador-$MODELO.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

exec 9>"logs/orquestrador-$MODELO.lock"
flock -n 9 || { log "Já há um orquestrador do $MODELO rodando; saindo."; exit 1; }
exec 8>"$HOME/.slurm_um_job_por_vez.lock"

impressao() { if [ -f "$PROGRESSO" ]; then md5sum < "$PROGRESSO"; else echo ausente; fi; }

log "Início (pid $$, $(hostname)): $JOB_SCRIPT ${SBATCH_ARGS[*]-} | termina quando existir $MARCA_FIM"
enviados=0
sem_progresso=0
while true; do
    if [ -f "$MARCA_FIM" ]; then log "Concluído: $MARCA_FIM existe."; exit 0; fi
    if [ -f PARAR ]; then rm -f PARAR; log "PARAR encontrado: encerrando sem submeter."; exit 0; fi
    if [ "$enviados" -ge "$MAX_JOBS" ]; then
        log "Limite de $MAX_JOBS jobs atingido sem concluir: confira os logs e rode de novo."; exit 1
    fi

    flock 8   # espera outros orquestradores do usuário (um job por vez entre todos os projetos)
    fila=$(squeue -h -u "$USER" -o '%i %j %T %P' 2>&1)
    status_fila=$?
    if [ "$status_fila" -ne 0 ] || [ -n "$fila" ]; then
        flock -u 8
        log "Fila ocupada (ou squeue falhou); nova tentativa em ${INTERVALO_FILA}s: $(echo "$fila" | tr '\n' ';')"
        sleep "$INTERVALO_FILA"
        continue
    fi

    antes=$(impressao)
    enviados=$((enviados + 1))
    log "Submetendo job $enviados/$MAX_JOBS"
    sbatch --wait ${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"} "$JOB_SCRIPT" 2>&1 8>&- 9>&- | tee -a "$LOG" 8>&- 9>&-
    rc=${PIPESTATUS[0]}
    flock -u 8
    id=$(grep -oE 'Submitted batch job [0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+$')
    log "Job ${id:-?} terminou: código $rc | $(sacct -X -n -P -j "${id:-0}" --format=State,Elapsed,NodeList 2>/dev/null | head -1)"

    if [ -f "$MARCA_FIM" ]; then continue; fi
    if [ "$(impressao)" = "$antes" ]; then
        sem_progresso=$((sem_progresso + 1))
        log "Sem progresso neste job ($sem_progresso/$MAX_SEM_PROGRESSO): veja logs/*-${id:-?}.out"
        if [ "$sem_progresso" -ge "$MAX_SEM_PROGRESSO" ]; then
            log "Desistindo: corrija o problema e rode o orquestrador de novo."; exit 1
        fi
    else
        sem_progresso=0
    fi
    log "Pausa de ${PAUSA_ENTRE_JOBS}s antes do próximo job."
    sleep "$PAUSA_ENTRE_JOBS"
done
