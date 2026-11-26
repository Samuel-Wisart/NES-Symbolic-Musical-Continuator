"""
audio

Gera uma prévia chiptune de um MIDI exportado, sem banco de sons externo.
"""

### Imports
import math
import random
import sys
import wave
from array import array
from pathlib import Path

import mido

### Configuração
SAMPLE_RATE = 24_000
# Mapeia os canais MIDI às vozes p1, p2, tr e no.
CHANNEL_TO_VOICE = {0: 0, 1: 1, 2: 2, 9: 3}


### Renderização
def render_wav(midi_path: Path, wav_path: Path) -> None:
    midi = mido.MidiFile(midi_path)
    audio = array("f", [0.0]) * max(1, math.ceil(midi.length * SAMPLE_RATE))
    channels = CHANNEL_TO_VOICE
    active: dict[int, tuple[int, int]] = {}
    rng = random.Random(0)

    def mix(voice: int, pitch: int, start: int, end: int) -> None:
        frequency = 440 * 2 ** ((pitch - 69) / 12)
        if voice == 3:
            frequency = 7500 * 2 ** (-(pitch - 1) / 3)
        increment = frequency / SAMPLE_RATE
        phase = 0.0
        length = end - start
        noise = rng.choice((-1, 1))
        for offset in range(length):
            if voice < 2:
                duty = 0.25 if voice == 0 else 0.5
                value = 0.17 * (float(phase < duty) - duty)
            elif voice == 2:
                value = 0.12 * (1 - 4 * abs(phase - 0.5))
            else:
                value = 0.055 * noise
            envelope = min(1.0, offset / 24, (length - 1 - offset) / 60)
            audio[start + offset] += value * envelope
            phase += increment
            if phase >= 1.0:
                phase %= 1.0
                if voice == 3:
                    noise = rng.choice((-1, 1))

    elapsed = 0.0
    for message in midi:
        elapsed += message.time
        if message.type not in ("note_on", "note_off") or message.channel not in channels:
            continue
        voice = channels[message.channel]
        sample = min(len(audio), round(elapsed * SAMPLE_RATE))
        attacking = message.type == "note_on" and message.velocity > 0
        if voice in active and (attacking or active[voice][0] == message.note):
            pitch, start = active.pop(voice)
            mix(voice, pitch, start, sample)
        if attacking:
            active[voice] = (message.note, sample)

    for voice, (pitch, start) in active.items():
        mix(voice, pitch, start, len(audio))

    peak = max(map(abs, audio))
    gain = 0.8 / max(peak, 0.1)
    pcm = array("h", (round(sample * gain * 32767) for sample in audio))
    if sys.byteorder == "big":
        pcm.byteswap()
    with wave.open(str(wav_path), "wb") as output:
        output.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        output.writeframes(pcm.tobytes())
