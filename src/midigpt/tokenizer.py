"""MIDI <-> event tokens.

Encode: performance MIDI -> [VELOCITY / NOTE_ON / NOTE_OFF / TIME_SHIFT] ids.
Sustain pedal (CC64) is folded into note durations here — a note released
while the pedal is down sounds until pedal release — and never tokenized.

Timing: absolute note times are quantized to the 10ms grid FIRST, then
consecutive grid positions are diffed into TIME_SHIFT tokens. (Diffing raw
times and quantizing the diffs would accumulate drift over a piece.)

Decode is defensive by design — it must survive model-generated token soup:
orphan NOTE_OFFs are dropped, a re-struck pitch closes its previous note,
and every open note is closed at EOS/end-of-stream. Output is a tick-based
Score at tpq=500 / qpm=120, i.e. exactly 1ms per tick.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from pathlib import Path

import numpy as np
from symusic import Note, Score, Tempo, Track

from . import vocab as V

_DECODE_TPQ = 500
_DECODE_QPM = 120.0                    # 500 ticks/quarter at 120qpm -> 1 tick = 1ms
_TICKS_PER_GRID = V.TIME_STEP_MS       # 10 ticks per grid unit


def _quantize(seconds: float) -> int:
    """Seconds -> grid units, rounding half UP (deterministic; Python's
    round() half-to-even makes ties depend on float noise)."""
    return int(math.floor(seconds * V.GRID_PER_SEC + 0.5))


def _pedal_spans(track) -> list[tuple[float, float]]:
    """Merged (start, end) sustain spans in seconds. Uses symusic's paired
    pedal events; falls back to raw CC64 if a file has none paired."""
    spans = [(p.time, p.end) for p in track.pedals]
    if not spans:
        down = None
        for c in sorted((c for c in track.controls if c.number == 64),
                        key=lambda c: c.time):
            if c.value >= 64:
                if down is None:
                    down = c.time
            elif down is not None:
                spans.append((down, c.time))
                down = None
        if down is not None:
            spans.append((down, float("inf")))
    spans.sort()
    merged: list[tuple[float, float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _sustained_notes(score: Score) -> list[tuple[int, int, int, int]]:
    """All non-drum notes as (on_grid, off_grid, pitch, velocity), with pedal
    folded into durations and pedal-extended overlaps of the same pitch cut
    at the re-strike."""
    raw: list[tuple[float, float, int, int]] = []
    for tr in score.tracks:
        if tr.is_drum:
            continue
        spans = _pedal_spans(tr)
        starts = [s for s, _ in spans]
        for n in tr.notes:
            if not (V.PITCH_MIN <= n.pitch <= V.PITCH_MAX):
                continue  # outside the 88-key range; MAESTRO never hits this
            end = n.end
            i = bisect_right(starts, end) - 1
            if i >= 0 and end < spans[i][1]:
                end = spans[i][1]  # pedal down at release -> ring until pedal up
            raw.append((n.time, end, n.pitch, min(max(int(n.velocity), 1), 127)))

    # same-pitch overlap (pedal extension can run past a re-strike): cut at re-strike
    raw.sort(key=lambda r: (r[2], r[0]))
    out: list[tuple[int, int, int, int]] = []
    for j, (t, end, p, v) in enumerate(raw):
        if j + 1 < len(raw) and raw[j + 1][2] == p:
            end = min(end, raw[j + 1][0])
        on_g = _quantize(t)
        off_g = max(_quantize(end), on_g + 1)
        out.append((on_g, off_g, p, v))
    return out


def encode_score(score: Score) -> np.ndarray:
    """Event token ids (uint16) for one piece. No COMPOSER/BOS/EOS framing —
    prepare.py adds that."""
    notes = _sustained_notes(score.to("second"))

    # (grid, is_on, pitch, velocity); note-offs sort before note-ons at the
    # same grid position so a re-strike is off-then-on
    events: list[tuple[int, int, int, int]] = []
    for on_g, off_g, p, v in notes:
        events.append((on_g, 1, p, v))
        events.append((off_g, 0, p, 0))
    events.sort()

    out: list[int] = []
    cur = 0
    cur_bin = -1
    for g, is_on, p, v in events:
        d = g - cur
        while d > 0:
            step = min(d, V.N_SHIFT)
            out.append(V.time_shift(step))
            d -= step
        cur = g
        if is_on:
            if v // 4 != cur_bin:
                cur_bin = v // 4
                out.append(V.velocity(v))
            out.append(V.note_on(p))
        else:
            out.append(V.note_off(p))
    return np.array(out, dtype=np.uint16)


def encode_file(path: str | Path) -> np.ndarray:
    return encode_score(Score(str(path)))


def decode(ids) -> Score:
    """Token ids -> tick Score (1ms ticks). Robust to arbitrary streams."""
    cur = 0
    cur_vel = V.velocity_bin_to_midi(V.N_VELOCITY // 2)
    active: dict[int, tuple[int, int]] = {}   # pitch -> (start_grid, velocity)
    notes: list[tuple[int, int, int, int]] = []

    def close(pitch: int, end_grid: int) -> None:
        start, vel = active.pop(pitch)
        notes.append((start, max(end_grid - start, 1), pitch, vel))

    for tok in np.asarray(ids).tolist():
        k = V.kind(tok)
        if k == "shift":
            cur += V.value(tok)
        elif k == "velocity":
            cur_vel = V.velocity_bin_to_midi(V.value(tok))
        elif k == "note_on":
            p = V.value(tok)
            if p in active:
                close(p, cur)
            active[p] = (cur, cur_vel)
        elif k == "note_off":
            p = V.value(tok)
            if p in active:
                close(p, cur)
        elif k == "eos":
            break
        # pad / bos / composer / invalid: ignored

    for p in sorted(active):
        close(p, cur)
    notes.sort()

    score = Score(_DECODE_TPQ)
    score.tempos.append(Tempo(0, _DECODE_QPM))
    track = Track("Piano", 0, False)
    for start, dur, pitch, vel in notes:
        track.notes.append(Note(start * _TICKS_PER_GRID, dur * _TICKS_PER_GRID,
                                pitch, vel))
    score.tracks.append(track)
    return score


def duration_seconds(ids) -> float:
    """Total musical time covered by a token stream."""
    total = 0
    for tok in np.asarray(ids).tolist():
        if V.kind(tok) == "shift":
            total += V.value(tok)
    return total / V.GRID_PER_SEC


def cut_at_seconds(ids: np.ndarray, seconds: float) -> np.ndarray:
    """Prefix of `ids` covering at most the first `seconds` of music.
    Cuts before the TIME_SHIFT that would cross the limit."""
    limit = int(seconds * V.GRID_PER_SEC)
    cur = 0
    for i, tok in enumerate(np.asarray(ids).tolist()):
        if V.kind(tok) == "shift":
            cur += V.value(tok)
            if cur > limit:
                return np.asarray(ids)[:i]
    return np.asarray(ids)
