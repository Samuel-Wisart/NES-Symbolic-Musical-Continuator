#!/bin/bash
# Pipeline completo do Transformer no cluster SONIC: treino (cadeia de jobs), avaliação, geração,
# relatório e comparação. Configuração em scripts/slurm/transformer/config.sh. Destacado da sessão:
#
#   mkdir -p logs
#   setsid nohup bash scripts/slurm/transformer/pipeline.sh > logs/pipeline-transformer.out 2>&1 < /dev/null &
#   tail -f logs/pipeline-transformer.log
exec bash "$(dirname "$(readlink -f "$0")")/../comum/pipeline.sh" transformer
