# CLAUDE.md — working guidelines for midigpt

An autoregressive transformer for expressive solo piano, trained on MAESTRO.
`PLAN.md` holds the full design rationale and phase roadmap — read it before
making design changes. This file is the operational playbook.

## Environment & tooling

- **Native Windows + uv.** No global pip; everything is `uv run` / `uv sync`.
  Python is pinned to 3.12 (`.python-version`); torch comes from the cu128
  index in `pyproject.toml` — required for the RTX 5090 (Blackwell), don't
  swap it for the default PyPI torch.
- **The GPU is shared with gaming.** Ask before launching anything on CUDA.
  Everything through Phase 1 (tokenizer, tests, dataset prep) is CPU-only;
  `--device cpu` works everywhere for smoke tests.
- **No WSL needed.** The sibling Latin-LLM project (`OneDrive\Desktop\train a
  model`) trained via WSL and its playbook documents that pain; this project
  deliberately stays native. `torch.compile` is off the table because of that
  (Triton needs Linux) — a 27M model doesn't need it.
- Port lineage: `model.py`/`train.py` derive from cicero's
  `train_token_gpt_v2.py`; packaging/tests follow `work\gptbird`.

## Commands

```
uv sync                                    # install deps
uv run pytest                              # full test suite (CPU, fast)
uv run python -m midigpt.prepare --download    # fetch + tokenize MAESTRO -> data/tokens/
uv run python -m midigpt.prepare --single path\to\x.mid --out data/tokens-overfit
uv run python -m midigpt.train --config configs/maestro-small.json --out-dir checkpoints/runN
uv run python -m midigpt.sample --ckpt checkpoints/runN/checkpoint_best.pt --out out.mid
uv run python -m midigpt.render out.mid    # -> out.wav (needs fluidsynth + soundfont)
```

Rendering: `winget install FluidSynth.FluidSynth`, then point the
`MIDIGPT_SOUNDFONT` env var at a piano `.sf2` (or drop one in `data/soundfonts/`).

## Conventions

- **`vocab.py` is the single source of truth** for token ids. Never hard-code
  id ranges elsewhere; the transpose augmentation in `train.py` additionally
  relies on NOTE_ON/NOTE_OFF being adjacent 88-wide blocks (tested).
- Token streams are flat **uint16** `.bin` files + a `manifest.json`
  (vocab layout, composer map, split stats). Pieces are framed
  `<COMPOSER> BOS ... EOS`. Regenerating `.bin`s after any vocab change is
  mandatory — the manifest records the layout so `train.py` can refuse stale data.
- **Every training run**: a config JSON in `configs/`, an out-dir with
  `resolved_config.json`, `metrics.jsonl`, periodic sample `.mid`/`.wav`s, and
  checkpoints. For A/Bs: fix the recipe, change exactly one variable.
- `data/` and `checkpoints/` are gitignored — code and configs only in git.

## Gotchas

- **MAESTRO is performance MIDI** — constant nominal tempo, no beat grid.
  Bar/Position (REMI-style) tokens are meaningless here; time is modeled with
  10ms TIME_SHIFT events on purpose (see PLAN.md for the full argument).
- **Sustain pedal is folded into note durations** at tokenize time (CC64 ≥ 64
  extends note-offs to pedal release); CC events are never tokenized. Piano
  samples sound wrong without this — don't "simplify" it away.
- **Quantize absolute times to the 10ms grid, then diff** to get TIME_SHIFTs.
  Diff-then-quantize accumulates drift over a piece; there's a test for this.
- Decoding must survive arbitrary token soup: orphan NOTE_OFFs are dropped,
  re-struck pitches close the previous note, everything closes at EOS.
- Sampling temperature is make-or-break for music: too low → stuck loops,
  too high → noise. Render a sweep (0.8 / 0.95 / 1.1) before judging a model.
- symusic for all MIDI I/O (fast C++ parser). Encode reads scores converted
  `.to("second")`; decode writes ticks at 1ms resolution (tpq=500, qpm=120).
