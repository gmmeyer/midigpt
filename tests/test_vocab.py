"""Vocab layout invariants that other code depends on."""

import numpy as np

from midigpt import vocab as V


def test_layout_is_contiguous_and_fits_uint16():
    assert V.COMPOSER_OFF == 3
    assert V.NOTE_ON_OFF == V.COMPOSER_OFF + V.N_COMPOSER
    # train.py's transpose augmentation needs NOTE_ON/NOTE_OFF adjacent
    assert V.NOTE_OFF_OFF == V.NOTE_ON_OFF + V.N_PITCH
    assert V.TIME_SHIFT_OFF == V.NOTE_OFF_OFF + V.N_PITCH
    assert V.VELOCITY_OFF == V.TIME_SHIFT_OFF + V.N_SHIFT
    assert V.VOCAB_SIZE == V.VELOCITY_OFF + V.N_VELOCITY
    assert V.VOCAB_SIZE < np.iinfo(np.uint16).max


def test_every_id_classifies():
    kinds = {V.kind(t) for t in range(V.VOCAB_SIZE)}
    assert "invalid" not in kinds
    assert V.kind(V.VOCAB_SIZE) == "invalid"


def test_constructors_round_trip():
    for p in (V.PITCH_MIN, 60, V.PITCH_MAX):
        assert V.kind(V.note_on(p)) == "note_on" and V.value(V.note_on(p)) == p
        assert V.kind(V.note_off(p)) == "note_off" and V.value(V.note_off(p)) == p
    for s in (1, 50, V.N_SHIFT):
        assert V.kind(V.time_shift(s)) == "shift" and V.value(V.time_shift(s)) == s
    for vel in (1, 64, 127):
        tok = V.velocity(vel)
        assert V.kind(tok) == "velocity"
        assert abs(V.velocity_bin_to_midi(V.value(tok)) - vel) <= 2
    assert V.value(V.composer(5)) == 5
