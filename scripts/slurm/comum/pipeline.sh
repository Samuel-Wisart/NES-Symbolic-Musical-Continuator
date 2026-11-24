#!/bin/bash
# Pipeline completo de um modelo no cluster SONIC, sem supervisão (ex.: à noite). Roda na phocus4,
# destacado da sessão (sobrevive ao fechamento do terminal/VS Code). Use o ponto de entrada do
# modelo, que chama este arquivo:
#
#   mkdir -p logs
#   setsid nohup bash scripts/slurm/transformer/pipeline.sh > logs/pipeline-transformer.out 2>&1 < /dev/null &
#   tail -f logs/pipeline-transformer.log
#
# Etapas (cada uma é pulada se já estiver feita; rodar de novo continua de onde parou):
#   1. confere a venv (.venv);
#   2. prepara o dataset (step01) num job, se ainda não houver data/nesmdb_processed.tar;
#   3. teste de retomada na CPU (segundos, phocus4);
#   4. testes + benchmark do modelo num job;
#   5. teste curto do treino real (job de 20 min: tem que parar sozinho e salvar);
#   6. cadeia de treino (orquestrar.sh) até runs/<modelo>/CONCLUIDO;
#   7. avaliação no teste (step04) e geração do protocolo (step05), cada uma num job;
#   8. relatório do modelo (step06) e comparação entre os modelos (step07), leves, na phocus4.
# Um job por vez (trava global do usuário + squeue vazio). Parar: `touch PARAR` na raiz.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../../.." || exit 1
MODELO=${1:?uso: pipeline.sh <transformer|lstm>}
source "scripts/slurm/$MODELO/config.sh" || exit 1
export MODELO TREINO_ARGS
mkdir -p logs
LOG=logs/pipeline-$MODELO.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
falhar() { log "ERRO: $*"; exit 1; }

exec 9>"logs/pipeline-$MODELO.lock"
flock -n 9 || { log "Já há um pipeline do $MODELO rodando; saindo."; exit 1; }
exec 8>"$HOME/.slurm_um_job_por_vez.lock"

RUN=runs/$MODELO
JOB_SCRIPT=scripts/slurm/comum/job.sbatch
PY=.venv/bin/python
export TMPDIR=/snfs1/speed/$USER/cache/tmp PIP_CACHE_DIR=/snfs1/speed/$USER/cache/pip
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
unset PYTHONHOME PYTHONPATH

parar_se_pedido() {
    if [ -f PARAR ]; then rm -f PARAR; log "PARAR encontrado: encerrando."; exit 0; fi
}

# Submete um job (sbatch --wait) quando a fila do usuário estiver vazia; devolve o código do job.
submeter() {
    local id rc fila status_fila
    while true; do
        parar_se_pedido
        flock 8
        fila=$(squeue -h -u "$USER" -o '%i %j %T %P' 2>&1)
        status_fila=$?
        if [ "$status_fila" -eq 0 ] && [ -z "$fila" ]; then break; fi
        flock -u 8
        log "Fila ocupada (ou squeue falhou); nova tentativa em 600 s: $(echo "$fila" | tr '\n' ';')"
        sleep 600
    done
    log "sbatch $*"
    sbatch --wait "$@" 2>&1 8>&- 9>&- | tee -a "$LOG" 8>&- 9>&-
    rc=${PIPESTATUS[0]}
    flock -u 8
    id=$(grep -oE 'Submitted batch job [0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+$')
    ULTIMO_JOB=$id
    log "Job ${id:-?} terminou: código $rc | $(sacct -X -n -P -j "${id:-0}" --format=State,Elapsed,NodeList 2>/dev/null | head -1)"
    return "$rc"
}

log "Início do pipeline do $MODELO (pid $$, $(hostname)) | $PARTICAO, jobs de $TEMPO_JOB"

### 1. Venv (criação: ver README, seção "Cluster SONIC")
"$PY" -c "import torch, numpy, pretty_midi, mido, matplotlib" 2>> "$LOG" \
    || falhar "a .venv não importa torch/numpy/pretty_midi/mido/matplotlib"
log "Venv ok: $("$PY" -c 'import torch; print("torch", torch.__version__)')"

### 2. Dataset (data/nesmdb_processed.tar no storage; extraído no disco local de cada máquina)
source scripts/slurm/comum/dados_locais.sh
if [ ! -f "$DADOS_TAR" ]; then
    submeter --partition="$PARTICAO" --time=01:00:00 --job-name=nes-dados \
        "$JOB_SCRIPT" scripts/slurm/comum/preparar_dados.sh || falhar "preparação do dataset (logs/nes-dados-$ULTIMO_JOB.out)"
    [ -f "$DADOS_TAR" ] || falhar "step01 terminou sem $DADOS_TAR"
fi
dados_locais_extrair || falhar "extrair $DADOS_TAR na phocus4"
log "Dataset ok: $(($(wc -l < data/processed/nesmdb/token_index.csv) - 1)) músicas tokenizadas"

### 3. Retomada na CPU
if [ ! -f "logs/teste_retomada-$MODELO.ok" ]; then
    log "Teste de retomada na CPU"
    ARCH=$MODELO bash scripts/slurm/comum/teste_retomada.sh >> "$LOG" 2>&1 || falhar "teste de retomada (ver $LOG)"
    touch "logs/teste_retomada-$MODELO.ok"
fi

### 4. Testes e benchmark na GPU
if [ ! -f "logs/teste_dev-$MODELO.ok" ]; then
    submeter --partition="$PARTICAO" --time=00:40:00 --job-name="$JOB-teste" \
        "$JOB_SCRIPT" scripts/slurm/comum/teste_dev.sh || falhar "testes/benchmark (logs/$JOB-teste-$ULTIMO_JOB.out)"
    grep -h "passaram\|benchmark:\|estimativa\|FALHOU\|\[teste\] cuda" "logs/$JOB-teste-$ULTIMO_JOB.out" | tee -a "$LOG"
    grep -q "estimativa por época" "logs/$JOB-teste-$ULTIMO_JOB.out" || falhar "benchmark não rodou"
    touch "logs/teste_dev-$MODELO.ok"
fi

### 5. Teste curto do treino real (o mesmo comando da cadeia; o trecho treinado fica valendo)
if [ ! -f "$RUN/progresso.json" ] && [ ! -f "$RUN/CONCLUIDO" ]; then
    submeter --partition="$PARTICAO" --time=00:20:00 --job-name="$JOB" "$JOB_SCRIPT" \
        || falhar "teste curto do treino (logs/$JOB-$ULTIMO_JOB.out)"
    grep -q "\[parada\]\|Concluído" "logs/$JOB-$ULTIMO_JOB.out" || falhar "o treino não parou sozinho antes do fim do job"
    [ -f "$RUN/progresso.json" ] || [ -f "$RUN/CONCLUIDO" ] || falhar "o treino não gravou progresso.json"
    log "Teste curto ok: $(tr -d '\n ' < "$RUN/progresso.json" 2>/dev/null)"
    sleep 300
fi

### 6. Cadeia de treino
if [ ! -f "$RUN/CONCLUIDO" ]; then
    log "Iniciando a cadeia de treino (orquestrar.sh $MODELO)"
    bash scripts/slurm/comum/orquestrar.sh "$MODELO" 8>&- 9>&- || falhar "cadeia de treino (ver logs/orquestrador-$MODELO.log)"
    [ -f "$RUN/CONCLUIDO" ] || { log "Cadeia encerrada sem CONCLUIDO (PARAR?)."; exit 0; }
    sleep 300
fi
log "Treino concluído: $(tr -d '\n ' < "$RUN/CONCLUIDO")"

### 7. Avaliação e geração
if [ ! -f "resultados/$MODELO/test_metrics.json" ]; then
    submeter --partition="$PARTICAO" --time=00:40:00 --job-name="$JOB-avaliacao" \
        "$JOB_SCRIPT" scripts/step04_evaluate.py --model "$MODELO" || falhar "avaliação (logs/$JOB-avaliacao-$ULTIMO_JOB.out)"
    sleep 60
fi
log "Teste: $("$PY" -c "
import json; m = json.load(open('resultados/$MODELO/test_metrics.json'))
print(f\"perplexidade {m['perplexity']:.3f} | loss {m['loss']:.4f} | acurácia {m['accuracy']:.2%} | época {m['epoch']}\")")"
if [ ! -f "resultados/amostras/$MODELO/summary.json" ]; then
    submeter --partition="$PARTICAO" --time=00:40:00 --job-name="$JOB-geracao" \
        "$JOB_SCRIPT" scripts/step05_generate.py --model "$MODELO" || falhar "geração (logs/$JOB-geracao-$ULTIMO_JOB.out)"
fi
log "Geração: $(ls "resultados/amostras/$MODELO"/*.mid 2>/dev/null | wc -l) MIDI em resultados/amostras/$MODELO/"

### 8. Relatório e comparação (leves, na phocus4)
"$PY" scripts/step06_report.py --model "$MODELO" >> "$LOG" 2>&1 || falhar "relatório (step06)"
"$PY" scripts/step07_compare.py >> "$LOG" 2>&1 || falhar "comparação (step07)"
log "Pipeline do $MODELO concluído: resultados/$MODELO/README.md e resultados/comparacao.md."
