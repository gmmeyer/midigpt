"""Token vocabulary — the single source of truth for id layout.

Event-based encoding (Music Transformer style) for performance MIDI:
NOTE_ON/NOTE_OFF over the 88-key piano range, TIME_SHIFT on a 10ms grid,
VELOCITY in 32 bins. Composer ids are reserved up front so conditioning
(Phase 5) never invalidates tokenized data.

Layout invariants relied on elsewhere (tested in test_vocab.py):
- NOTE_ON and NOTE_OFF are adjacent 88-wide blocks (train.py's transpose
  augmentation shifts both with one mask).
- Everything fits in uint16.
"""

from __future__ import annotations

PAD = 0
BOS = 1
EOS = 2

N_COMPOSER = 64
N_PITCH = 88          # piano range A0..C8
N_SHIFT = 100         # 10ms .. 1000ms
N_VELOCITY = 32       # 128 / 4

COMPOSER_OFF = 3
NOTE_ON_OFF = COMPOSER_OFF + N_COMPOSER          # 67
NOTE_OFF_OFF = NOTE_ON_OFF + N_PITCH             # 155
TIME_SHIFT_OFF = NOTE_OFF_OFF + N_PITCH          # 243
VELOCITY_OFF = TIME_SHIFT_OFF + N_SHIFT          # 343
VOCAB_SIZE = VELOCITY_OFF + N_VELOCITY           # 375

PITCH_MIN = 21        # A0
PITCH_MAX = PITCH_MIN + N_PITCH - 1              # 108, C8
TIME_STEP_MS = 10     # grid resolution; one TIME_SHIFT unit
GRID_PER_SEC = 1000 // TIME_STEP_MS              # 100


def composer(idx: int) -> int:
    assert 0 <= idx < N_COMPOSER, f"composer index {idx} out of range"
    return COMPOSER_OFF + idx


def note_on(pitch: int) -> int:
    assert PITCH_MIN <= pitch <= PITCH_MAX, f"pitch {pitch} outside piano range"
    return NOTE_ON_OFF + (pitch - PITCH_MIN)


def note_off(pitch: int) -> int:
    assert PITCH_MIN <= pitch <= PITCH_MAX, f"pitch {pitch} outside piano range"
    return NOTE_OFF_OFF + (pitch - PITCH_MIN)


def time_shift(units: int) -> int:
    """One shift token for a gap of `units` grid steps, 1..N_SHIFT."""
    assert 1 <= units <= N_SHIFT, f"shift {units} out of range"
    return TIME_SHIFT_OFF + (units - 1)


def velocity(vel: int) -> int:
    """MIDI velocity 1..127 -> one of 32 bins."""
    assert 1 <= vel <= 127, f"velocity {vel} out of range"
    return VELOCITY_OFF + (vel // 4)


def velocity_bin_to_midi(b: int) -> int:
    """Bin midpoint; bin 0 -> 2 ... bin 31 -> 126."""
    return b * 4 + 2


def kind(tok: int) -> str:
    """Classify a token id: pad/bos/eos/composer/note_on/note_off/shift/velocity."""
    if tok == PAD:
        return "pad"
    if tok == BOS:
        return "bos"
    if tok == EOS:
        return "eos"
    if COMPOSER_OFF <= tok < NOTE_ON_OFF:
        return "composer"
    if NOTE_ON_OFF <= tok < NOTE_OFF_OFF:
        return "note_on"
    if NOTE_OFF_OFF <= tok < TIME_SHIFT_OFF:
        return "note_off"
    if TIME_SHIFT_OFF <= tok < VELOCITY_OFF:
        return "shift"
    if VELOCITY_OFF <= tok < VOCAB_SIZE:
        return "velocity"
    return "invalid"


def value(tok: int) -> int:
    """The within-block value: pitch for notes, grid units for shifts,
    bin for velocities, index for composers."""
    k = kind(tok)
    if k == "note_on":
        return tok - NOTE_ON_OFF + PITCH_MIN
    if k == "note_off":
        return tok - NOTE_OFF_OFF + PITCH_MIN
    if k == "shift":
        return tok - TIME_SHIFT_OFF + 1
    if k == "velocity":
        return tok - VELOCITY_OFF
    if k == "composer":
        return tok - COMPOSER_OFF
    raise ValueError(f"token {tok} ({k}) has no value")
