#!/bin/bash
# Incluído (source) pelo job.sbatch, pelo pipeline.sh e pelo preparar_dados.sh (scripts/slurm/comum/).
#
# O /snfs1 está lento para criar arquivos (~0,3 s cada em 2026-11-24), e o step01 grava ~5 mil
# .npz. Por isso o dataset processado fica no storage num único arquivo, data/nesmdb_processed.tar,
# e cada máquina (nó do job ou phocus4) o extrai no próprio disco local, em DADOS_LOCAL. O link
# data/processed -> DADOS_LOCAL tem o mesmo destino em todas as máquinas, então o código Python
# (TOKENS_DIR = data/processed/nesmdb/tokens) e o step01 não mudam.
DADOS_TAR=data/nesmdb_processed.tar
DADOS_LOCAL="/tmp/${USER}-nes-processed"

# Garante o link data/processed -> DADOS_LOCAL (uma pasta real antiga é renomeada, não apagada).
dados_locais_ligar() {
    if [ -e data/processed ] && [ ! -L data/processed ]; then
        mv data/processed "data/processed.antigo.$(date +%s)"
    fi
    ln -sfn "$DADOS_LOCAL" data/processed
}

# Extrai o tar em DADOS_LOCAL se ainda não estiver extraído (ou se o tar mudou) e liga o link.
dados_locais_extrair() {
    [ -f "$DADOS_TAR" ] || return 1
    local marca
    marca="$DADOS_LOCAL/.extraido-$(stat -c %s-%Y "$DADOS_TAR")"
    if [ ! -f "$marca" ]; then
        rm -rf "$DADOS_LOCAL"
        mkdir -p "$DADOS_LOCAL"
        tar -xf "$DADOS_TAR" -C "$DADOS_LOCAL" || return 1
        touch "$marca"
    fi
    dados_locais_ligar
}
