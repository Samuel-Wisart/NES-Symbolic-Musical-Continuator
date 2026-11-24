#!/bin/bash
# Pipeline completo do LSTM no cluster SONIC: treino (cadeia de jobs), avaliação, geração,
# relatório e comparação. Configuração em scripts/slurm/lstm/config.sh. Destacado da sessão:
#
#   mkdir -p logs
#   setsid nohup bash scripts/slurm/lstm/pipeline.sh > logs/pipeline-lstm.out 2>&1 < /dev/null &
#   tail -f logs/pipeline-lstm.log
exec bash "$(dirname "$(readlink -f "$0")")/../comum/pipeline.sh" lstm
