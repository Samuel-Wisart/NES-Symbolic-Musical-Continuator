"""Tokenização de eventos para os quatro canais monofônicos do NES-MDB."""

from dataclasses import dataclass
from itertools import groupby
from math import isfinite
from pathlib import Path
from typing import Iterable, Iterator

import mido


@dataclass(frozen=True)
class Voice:
    name: str
    channel: int
    program: int
    pitches: range


VOICES = (
    Voice("p1", 0, 80, range(32, 109)),
    Voice("p2", 1, 81, range(32, 109)),
    Voice("tr", 2, 38, range(21, 109)),
    Voice("no", 9, 121, range(1, 17)),
)
FRAME_RATE = 120
MAX_SHIFT = 120
MIDI_TEMPO = 500_000
MIDI_TICKS_PER_BEAT = 480
TICKS_PER_FRAME = 8

PAD, BOS, EOS = range(3)
# Eventos são (tipo, número da voz, valor). Para SHIFT, o valor é em quadros.
Event = tuple[str, int, int]
EVENTS: tuple[Event, ...] = (
    (("PAD", -1, 0), ("BOS", -1, 0), ("EOS", -1, 0))
    + tuple(("SHIFT", -1, frames) for frames in range(1, MAX_SHIFT + 1))
    + tuple(("OFF", voice_id, 0) for voice_id in range(len(VOICES)))
    + tuple(("ON", voice_id, pitch)
            for voice_id, voice in enumerate(VOICES) for pitch in voice.pitches)
)
TOKEN_ID = {event: index for index, event in enumerate(EVENTS)}
VOCAB_SIZE = len(EVENTS)


def _shift(frames: int) -> Iterator[int]:
    whole, remainder = divmod(frames, MAX_SHIFT)
    for _ in range(whole):
        yield TOKEN_ID["SHIFT", -1, MAX_SHIFT]
    if remainder:
        yield TOKEN_ID["SHIFT", -1, remainder]


def _read_notes(path: str | Path) -> Iterator[tuple[int, int, str, int]]:
    midi = mido.MidiFile(path)
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError(f"{path}: esperado um MIDI síncrono com resolução em PPQ")

    voice_ids = {voice.name: index for index, voice in enumerate(VOICES)}
    timed_events = []
    for track in midi.tracks:
        voice_id = voice_ids.get(track.name.strip().lower())
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "set_tempo":
                timed_events.append((tick, -1, "TEMPO", message.tempo))
            elif message.type in ("note_on", "note_off"):
                if voice_id is None:
                    raise ValueError(f"{path}: trilha de notas desconhecida {track.name!r}")
                voice = VOICES[voice_id]
                if message.channel != voice.channel or message.note not in voice.pitches:
                    raise ValueError(
                        f"{path}: nota {message.note} ou canal {message.channel} "
                        f"incompatível com a voz {voice.name}"
                    )
                kind = "ON" if message.type == "note_on" and message.velocity else "OFF"
                timed_events.append((tick, voice_id, kind, message.note))

    # Ordenação estável preserva a ordem original dentro de cada voz no mesmo tick.
    timed_events.sort(key=lambda event: (event[0], event[1]))
    tempo = MIDI_TEMPO
    elapsed = previous_tick = 0
    denominator = midi.ticks_per_beat * 1_000_000
    for tick, voice_id, kind, value in timed_events:
        elapsed += (tick - previous_tick) * tempo
        previous_tick = tick
        if kind == "TEMPO":
            tempo = value
        else:
            frame = (elapsed * FRAME_RATE + denominator // 2) // denominator
            yield frame, voice_id, kind, value


def _encode(path: str | Path, stop_frame: int | None = None) -> list[int]:
    tokens = [BOS]
    active: list[int | None] = [None] * len(VOICES)
    last_frame = 0
    for frame, messages in groupby(_read_notes(path), key=lambda event: event[0]):
        if stop_frame is not None and frame >= stop_frame:
            break

        next_active = active.copy()
        attacks = set()
        for _, voice, kind, pitch in messages:
            if kind == "ON":
                next_active[voice] = pitch
                attacks.add(voice)
            elif next_active[voice] == pitch:
                next_active[voice] = None

        changes = []
        for voice, pitch in enumerate(next_active):
            if pitch is not None and voice in attacks:
                changes.append(TOKEN_ID["ON", voice, pitch])
            elif pitch is None and active[voice] is not None:
                changes.append(TOKEN_ID["OFF", voice, 0])

        if changes:
            tokens.extend(_shift(frame - last_frame))
            tokens.extend(changes)
            last_frame = frame
        active = next_active

    if stop_frame is None:
        tokens.append(EOS)
    else:
        tokens.extend(_shift(stop_frame - last_frame))
    return tokens


def encode_midi(path: str | Path) -> list[int]:
    """Codifica uma música inteira, incluindo BOS e EOS."""
    return _encode(path)


def encode_prompt(path: str | Path, seconds: float | None = None) -> list[int]:
    """Codifica o prefixo sem EOS, mantendo notas ativas no ponto de corte."""
    if seconds is None:
        tokens = encode_midi(path)[:-1]
    else:
        if not isfinite(seconds) or seconds <= 0:
            raise ValueError("seconds deve ser finito e positivo")
        tokens = _encode(path, int(seconds * FRAME_RATE + 0.5))

    if not any(EVENTS[token][0] == "ON" for token in tokens):
        raise ValueError("O trecho inicial deve conter pelo menos um ataque de nota")
    return tokens


def decode_midi(tokens: Iterable[int], path: str | Path) -> None:
    """Exporta a sequência completa, incluindo o prompt, para MIDI NES-MDB."""
    midi = mido.MidiFile(type=1, ticks_per_beat=MIDI_TICKS_PER_BEAT)
    tempo_track = midi.add_track()
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=MIDI_TEMPO))
    tracks = []
    for voice in VOICES:
        track = midi.add_track(voice.name)
        track.append(mido.Message("program_change", channel=voice.channel, program=voice.program))
        tracks.append(track)

    frame = 0
    last_ticks = [0] * len(VOICES)
    active: list[int | None] = [None] * len(VOICES)
    pending: dict[int, int | None] = {}

    def emit(voice_id: int, kind: str, pitch: int, at_frame: int) -> None:
        tick = at_frame * TICKS_PER_FRAME
        tracks[voice_id].append(mido.Message(
            kind, channel=VOICES[voice_id].channel, note=pitch,
            velocity=12 if kind == "note_on" else 0, time=tick - last_ticks[voice_id],
        ))
        last_ticks[voice_id] = tick

    def flush() -> None:
        for voice_id, pitch in sorted(pending.items()):
            previous_pitch = active[voice_id]
            if previous_pitch is not None:
                emit(voice_id, "note_off", previous_pitch, frame)
            if pitch is not None:
                emit(voice_id, "note_on", pitch, frame)
            active[voice_id] = pitch
        pending.clear()

    for index, token in enumerate(tokens):
        if not 0 <= token < VOCAB_SIZE:
            raise ValueError(f"Token inválido no índice {index}: {token}")
        kind, voice, value = EVENTS[token]
        if index == 0 and kind == "BOS":
            continue
        if kind == "EOS":
            break
        if kind == "SHIFT":
            flush()
            frame += value
        elif kind in ("ON", "OFF"):
            pending[voice] = value if kind == "ON" else None
        else:
            raise ValueError(f"{kind} fora da posição esperada no índice {index}")

    # Se o sufixo terminar em um ataque, dá a essa nota ao menos um quadro.
    boundary_attacks = {voice for voice, pitch in pending.items() if pitch is not None}
    flush()
    for voice_id, pitch in enumerate(active):
        if pitch is not None:
            end = frame + int(voice_id in boundary_attacks)
            emit(voice_id, "note_off", pitch, end)
    end_tick = max(frame * TICKS_PER_FRAME, *last_ticks)
    for track, last_tick in zip(tracks, last_ticks):
        track.append(mido.MetaMessage("end_of_track", time=end_tick - last_tick))
    midi.save(path)
