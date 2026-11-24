#!/bin/bash
# Só a cadeia de treino do LSTM (sem testes, avaliação nem geração). Destacado da sessão:
#
#   mkdir -p logs
#   setsid nohup bash scripts/slurm/lstm/treinar.sh > logs/orquestrador-lstm.out 2>&1 < /dev/null &
#   tail -f logs/orquestrador-lstm.log
exec bash "$(dirname "$(readlink -f "$0")")/../comum/orquestrar.sh" lstm
