"""
generate_markov

Carrega a matriz de transição do step02, amostra sequências de tokens
por cadeia de Markov de ordem 1 e escreve MIDIs com pretty_midi.

A amostragem é "restrita": como a cadeia de ordem 1 não tem memória de quais
notas estão ligadas, o gerador mantém esse estado e garante que a sequência
seja sempre válida (todo NOTE_OFF tem um NOTE_ON, cada canal é monofônico
como no NES, e nenhuma nota fica aberta no fim da música).
"""

import argparse
from pathlib import Path

import numpy as np
import pretty_midi

from prepareData_markov import decode, TO_ID, VOCAB

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "nesmdb"
MATRIX_PATH = PROCESSED / "markov_order1.npz"
OUTPUT_DIR = ROOT / "output" / "markov_samples"

PAD_ID = TO_ID["PAD"]
BOS_ID = TO_ID["BOS"]
EOS_ID = TO_ID["EOS"]

# Programas General MIDI razoáveis para os canais melódicos. Ajuste a gosto.
GM_PROGRAMS = {
    "p1": 80,   # Lead 1 (square wave)
    "p2": 80,
    "tr": 38,   # Synth Bass 1 (aproxima triângulo)
}

# O canal de ruído do NES usa pitches baixos que não correspondem a nenhum
# instrumento de bateria GM (35-81) e ficariam inaudíveis. Remapeamos para
# a faixa de percussão GM, mantendo a variedade original de pitches.
NOISE_DRUM_BASE = 35
NOISE_DRUM_RANGE = 47

# Metadados de cada token de nota: id -> (estado, canal, pitch)
TOKEN_INFO = {}
for _id, _tok in enumerate(VOCAB):
    if _tok.startswith("NOTE_"):
        _, _estado, _canal, _pitch = _tok.split("_")
        TOKEN_INFO[_id] = (_estado, _canal, int(_pitch))

OFF_IDS = np.array([i for i, (e, _, _) in TOKEN_INFO.items() if e == "OFF"])

# Tokens que nunca devem ser amostrados: PAD, BOS e todo NOTE_OFF
# (NOTE_OFF só é liberado quando a nota correspondente está aberta).
BASE_VALID = np.ones(len(VOCAB), dtype=bool)
BASE_VALID[[PAD_ID, BOS_ID]] = False
BASE_VALID[OFF_IDS] = False


def _off_id(canal, pitch):
    return TO_ID[f"NOTE_OFF_{canal}_{pitch}"]


def _sortear(pesos, valido, temperature, rng):
    """Sorteia um token entre os válidos. Retorna None se não houver massa."""
    pesos = np.where(valido, pesos, 0.0)
    if pesos.sum() <= 0:
        return None

    # Temperatura em espaço log: estável mesmo com T baixo (np.power
    # podia dar underflow para zero em todas as posições).
    with np.errstate(divide="ignore"):
        logits = np.log(pesos) / temperature
    logits -= logits.max()
    w = np.exp(logits)
    w /= w.sum()  # sempre renormaliza: probs vem em float32 e não soma 1 exato

    return int(rng.choice(len(w), p=w))


def amostrar(probs, global_probs, n_max=2000, min_tokens=0,
             temperature=1.0, rng=None):
    """Amostra uma sequência de tokens válida começando em BOS."""
    if temperature <= 0:
        raise ValueError("temperature deve ser > 0")

    rng = rng or np.random.default_rng()
    global_probs = global_probs.astype(np.float64)

    seq = [BOS_ID]
    abertas = {}  # canal -> pitch da nota atualmente ligada

    while len(seq) < n_max:
        valido = BASE_VALID.copy()
        for canal, pitch in abertas.items():
            valido[_off_id(canal, pitch)] = True
        if len(seq) < min_tokens:
            valido[EOS_ID] = False

        linha = probs[seq[-1]].astype(np.float64)
        prox = _sortear(linha, valido, temperature, rng)

        if prox is None:
            # Estado sem saída válida: cai na distribuição global
            prox = _sortear(global_probs, valido, temperature, rng)
        if prox is None:
            break

        if prox == EOS_ID:
            break

        info = TOKEN_INFO.get(prox)
        if info is not None:
            estado, canal, pitch = info
            if estado == "ON":
                # Canais do NES são monofônicos: fecha a nota anterior
                if canal in abertas:
                    seq.append(_off_id(canal, abertas.pop(canal)))
                abertas[canal] = pitch
            else:
                abertas.pop(canal)

        seq.append(prox)

    # Fecha o que sobrou aberto (EOS ou truncamento por n_max)
    if abertas:
        seq.append(TO_ID["TIME_SHIFT_16"])  # 0.25 s para as últimas notas soarem
        for canal, pitch in abertas.items():
            seq.append(_off_id(canal, pitch))

    seq.append(EOS_ID)
    return seq


def notas_para_midi(notes, output_path, tempo_bpm=150):
    """Converte a saída de decode() em um MIDI tocável.

    Os tempos já estão em segundos, então o tempo_bpm só afeta a grade
    de tics do arquivo, não a velocidade de reprodução.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)

    instrumentos = {
        canal: pretty_midi.Instrument(program=prog, name=canal)
        for canal, prog in GM_PROGRAMS.items()
    }
    instrumentos["no"] = pretty_midi.Instrument(program=0, is_drum=True, name="no")

    for channel, pitch, start, end in notes:
        if channel == "no":
            pitch = NOISE_DRUM_BASE + pitch % NOISE_DRUM_RANGE
        instrumentos[channel].notes.append(
            pretty_midi.Note(velocity=100, pitch=int(pitch),
                             start=float(start), end=float(end))
        )

    for inst in instrumentos.values():
        if inst.notes:
            midi.instruments.append(inst)

    midi.write(str(output_path))


def main(n_samples=5, temperature=1.0, n_max=2000, min_tokens=200, seed=None):
    rng = np.random.default_rng(seed)

    data = np.load(MATRIX_PATH)
    probs = data["probs"]
    global_probs = data["global_probs"]

    if int(data["vocab_size"]) != len(VOCAB):
        raise ValueError(
            f"Matriz treinada com vocab de {int(data['vocab_size'])} tokens, "
            f"mas o vocabulário atual tem {len(VOCAB)}. Retreine o modelo."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(n_samples):
        ids = amostrar(probs, global_probs, n_max=n_max,
                       min_tokens=min_tokens, temperature=temperature, rng=rng)
        notes = decode(ids)

        if not notes:
            print(f"[{i:03d}] sequência sem notas, pulando")
            continue

        out = OUTPUT_DIR / f"sample_T{temperature:.2f}_{i:03d}.mid"
        notas_para_midi(notes, out)
        print(f"[{i:03d}] {len(ids):4d} tokens → {len(notes):4d} notas → {out.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n-samples", type=int, default=10)
    # Comece com 1.0. Se estiver repetitivo, suba para 1.2; se caótico, 0.8.
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-max", type=int, default=2000)
    parser.add_argument("--min-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(n_samples=args.n_samples, temperature=args.temperature,
         n_max=args.n_max, min_tokens=args.min_tokens, seed=args.seed)
    