#!/bin/bash
# Incluído (source) pelos scripts de teste e preparação. Os MIDIs extraídos do NES-MDB
# (data/raw/nesmdb_midi, ~5 mil arquivos) só servem ao step01 e aos testes: em vez de criá-los
# no storage compartilhado (lento para muitos arquivos pequenos), ficam no disco local do nó,
# por um link simbólico que é removido no fim do job. O step01 não muda: o rmtree dele falha em
# silêncio no link (ignore_errors) e a extração escreve através dele.
MIDI_LINK=data/raw/nesmdb_midi
MIDI_LOCAL="/tmp/${USER}-nesmdb-midi-${SLURM_JOB_ID:-local}"

midi_local_preparar() {
    if [ -e "$MIDI_LINK" ] && [ ! -L "$MIDI_LINK" ]; then
        echo "[midi] $MIDI_LINK já existe no storage; usando-o"
        return
    fi
    mkdir -p "$MIDI_LOCAL"
    ln -sfn "$MIDI_LOCAL" "$MIDI_LINK"
    trap midi_local_limpar EXIT
    echo "[midi] MIDIs no disco local do nó: $MIDI_LOCAL"
}

midi_local_limpar() {
    [ -L "$MIDI_LINK" ] && rm -f "$MIDI_LINK"
    rm -rf "$MIDI_LOCAL"
}

# Extrai os MIDIs (para os testes) sem rodar o step01 inteiro.
midi_local_extrair() {
    midi_local_preparar
    "$PY" -c "import sys; sys.path.insert(0, 'scripts'); import step01_prepare_dataset as s; s.extract_dataset()"
}
