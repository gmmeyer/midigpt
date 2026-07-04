# MIDI Music Model — Plan

An autoregressive transformer that generates expressive solo piano, trained on MAESTRO.

**Port sources** (both live on this machine, both proven on the RTX 5090):
- **cicero** — the Latin LLM at `C:\Users\gmeyp\OneDrive\Desktop\train a model\`. Its
  `scripts/train_token_gpt_v2.py` is the primary port: a modern decoder substrate
  (RoPE + RMSNorm + SwiGLU + optional GQA, SDPA flash attention) plus a mature training
  harness — JSON run configs, grad accumulation, full resume (optimizer + RNG state),
  `metrics.jsonl` logging, periodic sampling to files, memmap token loading via a
  manifest. It trained a 100M model at block 2048 on this card.
- **gptbird** — `C:\Users\gmeyp\work\gptbird`. Contributes the packaging pattern
  (uv + Python 3.12 pin + the cu128 torch index the 5090 needs on native Windows) and
  the test-suite conventions.

Only the tokenizer, the MIDI-native sampling/eval story, and thin glue are new code.

## Decisions (and why)

### Dataset: MAESTRO v3.0.0
~1,276 performances / ~200 hours / ~7M notes of competition-grade piano, recorded on
Disklaviers. MIDI-only download is ~60MB. Ships with official train/validation/test
splits and composer/title metadata (CSV). Clean enough that zero data triage is needed —
the opposite of Lakh. Lakh stays on the shelf for a later multi-instrument phase.

- Download: `https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip`
- Use the official splits (split by piece — avoids near-duplicate leakage between train and val).

### Tokenization: event-based (Music Transformer style), NOT REMI
The chat-brainstorm plan suggested REMI, but REMI's Bar/Position tokens require a
meaningful beat grid. MAESTRO is **performance MIDI**: tempo metadata is a constant
default and there is no downbeat annotation, so REMI bars would be arbitrary fixed-width
windows with no musical meaning. Event-based tokenization is what Music Transformer used
on this exact dataset, and it preserves the expressive human timing that makes samples
sound alive. REMI becomes a Phase-6 experiment on score-like MIDI instead.

Vocabulary (~372 tokens, fits uint16 with room to spare):

| block            | count | notes                                             |
|------------------|-------|---------------------------------------------------|
| specials         | 3     | PAD, BOS, EOS                                     |
| composer ids     | 64    | reserved NOW so Phase-5 doesn't invalidate .bins  |
| NOTE_ON pitch    | 88    | piano range A0(21)–C8(108)                        |
| NOTE_OFF pitch   | 88    | 〃                                                 |
| TIME_SHIFT       | 100   | 10ms steps, 10ms–1s; longer gaps = multiple tokens|
| VELOCITY         | 32    | 128/4 bins; sets velocity for subsequent NOTE_ONs |

Preprocessing rules:
- **Sustain pedal**: don't model CC64. Extend each note's off-time until the pedal
  releases (CC64 < 64) or the same pitch re-strikes, then drop pedal events. This is the
  Performance RNN / Music Transformer standard and matters a lot for piano quality.
- MIDI parsing via **symusic** (fast C++ parser; exposes notes + pedal events; can give
  timing in seconds directly). Whole dataset tokenizes in seconds.
- Output: `train.bin` / `val.bin` / `test.bin` — flat uint16 streams, pieces separated as
  `<COMPOSER_i> BOS ...events... EOS`, plus a manifest JSON (vocab layout, composer map,
  piece start offsets) in the same spirit as cicero's `tokenized_manifest.json`.
- ~7M notes × ~4–5 tokens/note ≈ **30M train tokens**. Small — plan for overfitting
  (dropout + augmentation), not for data starvation.

### Architecture: cicero's v2 substrate, as-is
`TokenGPTV2` ports with almost no changes: RoPE (already implemented, cached cos/sin),
RMSNorm with f32 compute, SwiGLU (ratio 3.0), SDPA flash, weight tying. Two config-level
choices:
- **GQA off** (`n_kv_head = n_head`): GQA only shrinks the KV cache, which is trivial at
  this scale. The code supports it; the config just doesn't ask for it.
- `block_size` 2048 (≈ 1–2 minutes of performance) — same as cicero, known-good.
- One addition: an optional per-token **loss mask** in the forward (5 lines, pattern in
  gptbird's model) so PAD/conditioning tokens can be excluded later.

Tiers (SwiGLU-3x params):

| tier   | layers | d_model | heads | params | use                     |
|--------|--------|---------|-------|--------|-------------------------|
| nano   | 4      | 256     | 4     | ~3M    | smoke tests, overfit-1  |
| small  | 8      | 512     | 8     | ~27M   | main model              |
| medium | 12     | 512     | 8     | ~40M   | if small underfits      |

### Training: cicero's harness, MIDI-flavored
Port `train_token_gpt_v2.py`'s main loop wholesale — JSON run configs (same schema as
`configs/cicero-substrate-100m.json`), cosine schedule with warmup, bf16 autocast, grad
clip, resume with RNG state, `metrics.jsonl`, `resolved_config.json` per run. Changes:
- **Batch**: 64 × 2048 ≈ 131k tokens/step, no grad accum needed (cicero needed 16×8 for
  a 100M model on this card; 27M fits with room).
- **lr 3e-4, warmup 500** (total run is only ~5–10k steps).
- **Val loss** on the official validation split; checkpoint best-val. With 30M train
  tokens, expect best-val in the 10–30 epoch range — early stopping matters.
- **Augmentation, on the fly in token space**: per-window pitch transpose k ∈ [-3, +3]
  (shift NOTE_ON/OFF ids by k; clamp k per window so no note leaves the 88-key range).
  Exact, ~free, and the single most effective anti-overfit trick on MAESTRO.
  Time-stretch ±5% (remap TIME_SHIFT bins) is optional, slightly lossy — only if val
  loss says we need it.
- **Sample hook**: cicero writes `sample_step_NNNNNN.txt`; ours writes `.mid` and shells
  out to `render.py` for a `.wav` — every run leaves behind listenable checkpoints.
  Render a small temperature sweep (0.8 / 0.95 / 1.1) each time; temperature is
  make-or-break for music.
- Budget estimate: ~1.5–3 steps/s → a full 5–10k-step run in **under ~2 hours** on the
  5090. Multiple experiments per evening are realistic.

### Environment: native Windows uv (gptbird pattern), not WSL
Both paths are proven on this machine — cicero trained in WSL, gptbird natively. Native
wins for a fresh project: no CRLF/`wsl bash` boundary pain (cicero's CLAUDE.md documents
plenty), and uv + the cu128 index is a two-line pyproject block. The one thing WSL would
buy is `torch.compile` (Triton), which a 27M model doesn't need. Revisit only if
training feels slow. fluidsynth installs natively via `winget install FluidSynth.FluidSynth`.

Data convention: MAESTRO is ~60MB raw / ~60MB tokenized — small enough to live in
repo-local `data/` (gitignored), no need for the `E:` data-drive convention cicero uses.

### Evaluation: ears first, numbers second
- `sample.py` — unconditional generation (temperature, top-p as flags), and
  **continuation mode**: tokenize the first ~30s of a real MIDI file, let the model
  finish it, write both to disk. Decoder must be robust: track active notes, drop orphan
  NOTE_OFFs, close everything at EOS.
- `render.py` — MIDI → WAV via the fluidsynth CLI + a piano soundfont (FluidR3_GM works;
  a dedicated piano .sf2 sounds better).
- Borrow cicero's experimental discipline where it counts: every run gets a config JSON
  and a `metrics.jsonl`; for any A/B (tokenizer variants, aug on/off), fix the recipe,
  change one variable, and decide on held-out val loss + blind listening.

## Repo layout

```
midi-music-model/
  pyproject.toml            # uv; copy gptbird's cu128 index block; Python 3.12 pin
  PLAN.md
  configs/
    overfit-one.json        # nano tier, single piece (Phase 2)
    maestro-small.json      # the main recipe (Phase 3)
  data/
    maestro/                # extracted dataset (gitignored)
    tokens/                 # train/val/test.bin + manifest.json (gitignored)
  src/midigpt/
    vocab.py                # token id layout, single source of truth
    tokenizer.py            # MIDI <-> events <-> ids; sustain handling
    prepare.py              # download, verify, tokenize per official split -> .bin
    model.py                # TokenGPTV2 port (+ loss-mask arg)
    train.py                # cicero v2 loop + aug + .mid/.wav sample hook
    sample.py               # unconditional + continuation -> .mid
    render.py               # .mid -> .wav (fluidsynth)
  tests/
    test_tokenizer.py       # THE critical tests — see Phase 1
    test_model.py           # shapes, causality, tiny-overfit smoke
  checkpoints/              # (gitignored)
```

## Phases

**Phase 0 — scaffold.** `git init`, `uv init` (Python 3.12 pin), pyproject with cu128
torch index, port `model.py`/`train.py` from cicero v2, empty test harness. ~30 min.

**Phase 1 — tokenizer + tests.** This is where all the silent bugs live, so it gets the
test budget: (a) round-trip — MIDI → tokens → MIDI preserves every note's pitch/velocity
bin/onset within 10ms quantization; (b) sustain-pedal extension against a hand-built
fixture; (c) vocab bounds — every MAESTRO file tokenizes with zero out-of-range ids;
(d) decode robustness on garbage token streams. Then `prepare.py` over the full dataset.
(symusic's pedal-event API needs a 5-minute check here; fallback is a mido pass.)

**Phase 2 — overfit one piece.** Nano tier, one performance, loss → near zero, sampled
output audibly *is* the piece. Proves model + data path + decode end-to-end. Same-day
milestone; can run while the GPU is only lightly gamed-on.

**Phase 3 — real training.** Small tier on full MAESTRO train split. Watch val loss,
listen to periodic renders. Deliverable: checkpoints + a pile of WAVs, and a first
verdict on temperature settings. This is the "it makes music now" milestone.

**Phase 4 — continuation sampling.** Prime with 30s of held-out test pieces. This is the
best demo mode and also the most honest eval (exposes drift and incoherence fast).

**Phase 5 — composer conditioning.** Ids are already reserved and baked into the streams.
To make the signal actually reach the model (random windows rarely include position 0 of
a piece), fine-tune with a fraction of windows anchored at piece starts. Then
`sample.py --composer chopin`.

**Phase 6 — stretch, pick by mood.**
- **Browser demo**: cicero already has the full pipeline — `export_to_onnx.py`,
  `convert_tokenizer_for_browser.py`, and a deployed `web/` app. Swap text decode for a
  Web-MIDI/Tone.js player and the piano model runs on a web page.
- REMI on score-like MIDI (Lakh clean subset / POP909) as the tokenizer A/B the chat
  plan wanted.
- Multi-instrument; velocity-humanizer head.

## Open items / risks
- MAESTRO's occasional >1s silences become TIME_SHIFT token runs — fine, but confirm the
  tokenizer handles multi-token gaps in round-trip tests.
- GPU is shared with gaming — training runs are launch-and-walk-away; coordinate before
  Phase 2+.
