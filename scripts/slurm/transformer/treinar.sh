#!/bin/bash
# Só a cadeia de treino do Transformer (sem testes, avaliação nem geração). Destacado da sessão:
#
#   mkdir -p logs
#   setsid nohup bash scripts/slurm/transformer/treinar.sh > logs/orquestrador-transformer.out 2>&1 < /dev/null &
#   tail -f logs/orquestrador-transformer.log
exec bash "$(dirname "$(readlink -f "$0")")/../comum/orquestrar.sh" transformer
