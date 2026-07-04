"""Tokenizer correctness — round-trip fidelity, sustain folding, robustness.

Fixtures are built at tpq=500 / qpm=120 so 1 tick = exactly 1ms, matching the
decoder's output convention. All comparisons happen on the 10ms grid.
"""

from pathlib import Path

import numpy as np
import pytest
from symusic import ControlChange, Note, Pedal, Score, Tempo, Track

from midigpt import vocab as V
from midigpt.tokenizer import (cut_at_seconds, decode, duration_seconds,
                               encode_file, encode_score)

MAESTRO = Path("data/maestro/maestro-v3.0.0")


def make_score(notes, pedals=()):
    """notes: (onset_ms, dur_ms, pitch, velocity); pedals: (onset_ms, dur_ms)."""
    s = Score(500)
    s.tempos.append(Tempo(0, 120.0))
    tr = Track("Piano", 0, False)
    for t, d, p, v in notes:
        tr.notes.append(Note(t, d, p, v))
    for t, d in pedals:
        tr.pedals.append(Pedal(t, d))
    s.tracks.append(tr)
    return s


def decoded_notes(ids):
    """Decode to (onset_grid, dur_grid, pitch, velocity) tuples."""
    out = decode(ids)
    return sorted((n.time // 10, n.duration // 10, n.pitch, n.velocity)
                  for n in out.tracks[0].notes)


def test_round_trip_two_notes_and_chord():
    ids = encode_score(make_score([
        (0, 500, 60, 80),
        (500, 250, 64, 90),
        (500, 250, 67, 90),      # chord with the previous
        (1000, 495, 72, 40),     # 495ms rounds to 50 grid units
    ]))
    assert decoded_notes(ids) == [
        (0, 50, 60, V.velocity_bin_to_midi(80 // 4)),
        (50, 25, 64, V.velocity_bin_to_midi(90 // 4)),
        (50, 25, 67, V.velocity_bin_to_midi(90 // 4)),
        (100, 50, 72, V.velocity_bin_to_midi(40 // 4)),
    ]


def test_velocity_token_only_on_bin_change():
    ids = encode_score(make_score([
        (0, 100, 60, 80), (200, 100, 62, 81), (400, 100, 64, 20),
    ]))
    vel_tokens = [t for t in ids.tolist() if V.kind(t) == "velocity"]
    assert len(vel_tokens) == 2  # 80 and 81 share a bin; 20 differs


def test_sustain_pedal_extends_note():
    # note released at 100ms while pedal (50..500ms) is down -> rings to 500ms
    ids = encode_score(make_score([(0, 100, 60, 80)], pedals=[(50, 450)]))
    assert decoded_notes(ids)[0][:2] == (0, 50)


def test_pedal_up_before_release_no_extension():
    ids = encode_score(make_score([(0, 300, 60, 80)], pedals=[(0, 100)]))
    assert decoded_notes(ids)[0][:2] == (0, 30)


def test_restrike_under_pedal_cuts_previous():
    # both notes' releases are pedal-extended, but the first must stop
    # ringing when the same pitch is struck again at 300ms
    ids = encode_score(make_score(
        [(0, 100, 60, 80), (300, 100, 60, 80)], pedals=[(0, 1000)]))
    assert decoded_notes(ids) == [
        (0, 30, 60, V.velocity_bin_to_midi(80 // 4)),
        (30, 70, 60, V.velocity_bin_to_midi(80 // 4)),
    ]


def test_cc64_fallback_matches_pedal_events():
    a = make_score([(0, 100, 60, 80)], pedals=[(50, 450)])
    b = make_score([(0, 100, 60, 80)])
    b.tracks[0].controls.append(ControlChange(50, 64, 100))
    b.tracks[0].controls.append(ControlChange(500, 64, 0))
    assert encode_score(a).tolist() == encode_score(b).tolist()


def test_long_gap_multiple_shift_tokens():
    ids = encode_score(make_score([(0, 100, 60, 80), (3500, 100, 62, 80)]))
    shifts = [V.value(t) for t in ids.tolist() if V.kind(t) == "shift"]
    assert max(shifts) == V.N_SHIFT  # 1s chunks needed for the 3.4s gap
    assert decoded_notes(ids)[1][0] == 350


def test_no_timing_drift():
    # 200 onsets at multiples of 34ms: fractional grid positions .0/.4/.8/.2/.6,
    # never exactly on a rounding boundary, so expected values are unambiguous.
    # Each decoded onset must equal the quantized *absolute* time — a constant
    # 34ms gap diff-then-quantized to round(3.4)=3 would drift ~40ms/second.
    notes = [(34 * i, 20, 60 + (i % 12), 80) for i in range(200)]
    got = decoded_notes(encode_score(make_score(notes)))
    onsets = sorted((34 * i + 5) // 10 for i in range(200))
    assert [n[0] for n in got] == onsets


def test_zero_length_note_survives():
    ids = encode_score(make_score([(0, 0, 60, 80)]))
    assert decoded_notes(ids) == [(0, 1, 60, V.velocity_bin_to_midi(20))]


def test_decode_survives_garbage():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, V.VOCAB_SIZE, size=5000, dtype=np.uint16)
    score = decode(ids)  # must not raise
    for n in score.tracks[0].notes:
        assert n.duration >= 10  # every note closed, min one grid unit


def test_orphan_note_off_dropped_and_open_notes_closed():
    ids = np.array([V.note_off(60), V.time_shift(10), V.note_on(64)],
                   dtype=np.uint16)
    assert decoded_notes(ids) == [(10, 1, 64, V.velocity_bin_to_midi(16))]


def test_eos_stops_decoding():
    ids = np.array([V.note_on(60), V.time_shift(50), V.EOS,
                    V.time_shift(50), V.note_on(64)], dtype=np.uint16)
    assert decoded_notes(ids) == [(0, 50, 60, V.velocity_bin_to_midi(16))]


def test_cut_at_seconds():
    ids = encode_score(make_score([(i * 1000, 100, 60, 80) for i in range(10)]))
    cut = cut_at_seconds(ids, 3.0)
    assert duration_seconds(cut) <= 3.0
    assert len(decoded_notes(cut)) == 4  # onsets at 0,1,2,3s


@pytest.mark.skipif(not MAESTRO.exists(), reason="MAESTRO not downloaded")
def test_real_maestro_file_round_trip_and_idempotency():
    path = sorted(MAESTRO.rglob("*.mid*"))[0]
    ids = encode_score(Score(str(path)))
    assert int(ids.max()) < V.VOCAB_SIZE
    assert int(ids.min()) >= V.NOTE_ON_OFF  # events only: no specials/composers

    src = Score(str(path)).to("second")
    n_in_range = sum(1 for tr in src.tracks for n in tr.notes
                     if V.PITCH_MIN <= n.pitch <= V.PITCH_MAX)
    assert len(decode(ids).tracks[0].notes) == n_in_range

    # encode(decode(x)) is a fixed point: grid times, folded sustain, and
    # binned velocities all survive a second pass exactly
    again = encode_score(decode(ids))
    assert again.tolist() == ids.tolist()
