# midigpt

An autoregressive transformer that generates expressive solo piano, trained on
[MAESTRO](https://magenta.tensorflow.org/datasets/maestro). Event-based tokenization
(Music Transformer style), ~27M-param decoder (RoPE + RMSNorm + SwiGLU), trained on a
single RTX 5090.

See `PLAN.md` for the full design and roadmap, `CLAUDE.md` for the operational playbook.

## Quickstart

```
uv sync
uv run pytest
uv run python -m midigpt.prepare --download          # fetch + tokenize MAESTRO
uv run python -m midigpt.train --config configs/maestro-small.json --out-dir checkpoints/run0
uv run python -m midigpt.sample --ckpt checkpoints/run0/checkpoint_best.pt --out sample.mid
```
