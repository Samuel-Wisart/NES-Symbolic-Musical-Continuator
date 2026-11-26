"""
midi_to_audio

Percorre a pasta de MIDIs gerados e converte cada um em áudio (.wav).

Em vez de depender de FluidSynth e de um soundfont, usa um sintetizador
chiptune próprio (só numpy) que imita os canais do NES:
    p1, p2 -> onda de pulso (duty 25% e 50%)
    tr     -> onda triangular com 16 níveis (4 bits, como no hardware)
    no     -> ruído binário com decaimento (percussão)

Uso:
    python midi_to_audio.py
    python midi_to_audio.py --input-dir caminho/midis --output-dir caminho/wavs
    python midi_to_audio.py --overwrite --mp3
"""

### Imports
import argparse
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pretty_midi

### Caminhos
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "output" / "markov_samples"
DEFAULT_OUTPUT = ROOT / "output" / "markov_audio"

### Configuração da síntese
ATTACK = 0.002   # s, evita clique no início da nota
RELEASE = 0.010  # s, evita clique no fim da nota
TAIL = 0.3       # s de silêncio no final do arquivo

# Mix entre canais (o triângulo do NES não tem controle de volume e soa mais baixo)
GANHO = {"p1": 0.25, "p2": 0.25, "tr": 0.35, "no": 0.15}
DUTY = {"p1": 0.25, "p2": 0.50}

NOISE_BASE_PITCH = 35     # menor pitch de percussão usado pelo gerador
NOISE_BASE_HZ = 1000.0    # taxa de troca do ruído no pitch base (sobe 1 oitava por 12 pitches)
NOISE_DECAY_S = 0.10      # constante de decaimento do ruído


### Seleção de voz
def tipo_voz(inst):
    """Decide qual voz sintetizar a partir do nome/tipo do instrumento."""
    nome = inst.name.lower().strip()
    if inst.is_drum or nome == "no":
        return "no"
    return nome if nome in ("p1", "p2", "tr") else "p2"

### Síntese de notas
def sintetizar_nota(voz, pitch, duracao, sr, rng):
    """Gera a forma de onda de uma nota (já com envelope)."""
    n_corpo = max(1, int(duracao * sr))
    n_rel = max(1, int(RELEASE * sr))
    n = n_corpo + n_rel
    t = np.arange(n) / sr

    if voz == "no":
        # Ruído amostrado-e-segurado. Pitch maior = troca mais rápida = mais agudo
        freq = min(NOISE_BASE_HZ * 2 ** ((pitch - NOISE_BASE_PITCH) / 12), sr / 2)
        valores = rng.choice([-1.0, 1.0], size=int(t[-1] * freq) + 2)
        onda = valores[(t * freq).astype(int)]
    else:
        freq = 440.0 * 2 ** ((pitch - 69) / 12)
        fase = (t * freq) % 1.0
        if voz == "tr":
            tri = 4 * np.abs(fase - 0.5) - 1
            onda = np.round((tri + 1) * 7.5) / 7.5 - 1  # 16 níveis
        else:
            onda = np.where(fase < DUTY[voz], 1.0, -1.0)

    env = np.ones(n)
    n_att = min(int(ATTACK * sr), n_corpo)
    if n_att:
        env[:n_att] = np.linspace(0, 1, n_att)
    env[n_corpo:] = np.linspace(1, 0, n - n_corpo)
    if voz == "no":
        env *= np.exp(-t / NOISE_DECAY_S)

    return onda * env

### Renderização de MIDI
def renderizar(midi_path, sr):
    """Renderiza um MIDI inteiro em um array mono float em [-1, 1]."""
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    total_notas = sum(len(inst.notes) for inst in midi.instruments)
    if total_notas == 0:
        return None

    buf = np.zeros(int((midi.get_end_time() + TAIL) * sr) + 1)
    rng = np.random.default_rng(0)  # ruído reprodutível

    for inst in midi.instruments:
        voz = tipo_voz(inst)
        for nota in inst.notes:
            duracao = nota.end - nota.start
            if duracao <= 0:
                continue
            onda = sintetizar_nota(voz, nota.pitch, duracao, sr, rng)
            ini = int(nota.start * sr)
            fim = min(ini + len(onda), len(buf))
            buf[ini:fim] += GANHO[voz] * (nota.velocity / 127) * onda[:fim - ini]

    pico = np.abs(buf).max()
    if pico > 0:
        buf *= 0.9 / pico  # normaliza para evitar clipping
    return buf

### Exportação de áudio
def salvar_wav(path, audio, sr):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())

def wav_para_mp3(wav_path):
    mp3_path = wav_path.with_suffix(".mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), str(mp3_path)],
        check=True,
    )

### Execução
def main(input_dir=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT, sample_rate=44100,
         overwrite=False, mp3=False):
    input_dir, output_dir = Path(input_dir), Path(output_dir)

    midis = sorted([*input_dir.glob("*.mid"), *input_dir.glob("*.midi")])
    if not midis:
        raise FileNotFoundError(f"Nenhum MIDI encontrado em {input_dir}")

    if mp3 and shutil.which("ffmpeg") is None:
        print("ffmpeg não encontrado: gerando apenas .wav")
        mp3 = False

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Convertendo {len(midis)} MIDI(s) de {input_dir}")

    ok = pulados = falhas = 0

    for midi_path in midis:
        wav_path = output_dir / f"{midi_path.stem}.wav"

        if wav_path.exists() and not overwrite:
            print(f"[pulado]  {wav_path.name} já existe (use --overwrite)")
            pulados += 1
            continue

        try:
            audio = renderizar(midi_path, sample_rate)
            if audio is None:
                print(f"[vazio]   {midi_path.name} não tem notas")
                falhas += 1
                continue

            salvar_wav(wav_path, audio, sample_rate)
            if mp3:
                wav_para_mp3(wav_path)

            print(f"[ok]      {midi_path.name} -> {wav_path.name} "
                  f"({len(audio) / sample_rate:.1f}s)")
            ok += 1
        except Exception as exc:
            print(f"[erro]    {midi_path.name}: {exc}")
            falhas += 1

    print(f"\nConcluído: {ok} convertido(s), {pulados} pulado(s), {falhas} falha(s)")
    print("Saída:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converte MIDIs em áudio chiptune.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--overwrite", action="store_true",
                        help="reconverte mesmo que o .wav já exista")
    parser.add_argument("--mp3", action="store_true",
                        help="também gera .mp3 (requer ffmpeg)")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.sample_rate,
         args.overwrite, args.mp3)
