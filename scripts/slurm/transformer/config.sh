#!/bin/bash
# Configuração do Transformer no cluster SONIC (incluída pelos scripts de scripts/slurm/comum/).
# Rodada de 2026-11-24: 100 épocas (padrão do step03), ~31 s/época numa RTX 4090; melhor época 40.
MODELO=transformer
# Nome dos jobs (logs em logs/<nome>-<id>.out).
JOB=nes-transformer
# Em 2026-11-23 as gorgonas travaram e tudo rodou na gorgonas_dev (máximo 1h30 por job). Na
# gorgonas: PARTICAO=gorgonas e TEMPO_JOB=08:00:00 (nunca mais de 8 h; nunca --gres).
PARTICAO=gorgonas_dev
TEMPO_JOB=01:30:00
# Opções extras do scripts/step03_train_transformer.py (ex.: "--epochs 120"). Lote, taxa e semente não
# podem mudar no meio de uma rodada (a retomada recusa); --epochs pode aumentar.
TREINO_ARGS=""
