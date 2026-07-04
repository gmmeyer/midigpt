"""Train a MusicGPT on tokenized MAESTRO.

Ported from cicero's train_token_gpt_v2.py: JSON run configs, cosine LR with
warmup, bf16 autocast, grad accumulation/clip, full resume (optimizer + RNG),
metrics.jsonl. MIDI-specific additions:

- on-the-fly pitch-transpose augmentation in token space (the main anti-overfit
  lever on MAESTRO's ~23M tokens);
- best-val checkpointing;
- a sample hook that writes real .mid files (and optionally renders .wav) every
  sample_interval steps, at a small temperature sweep.

    uv run python -m midigpt.train --config configs/maestro-small.json --out-dir checkpoints/run0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from . import vocab as V
from .model import GPTConfig, MusicGPT
from .sample import generate_tokens
from .tokenizer import decode


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None)}


def restore_rng_state(s: dict | None) -> None:
    if not s:
        return
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["torch_cuda"])


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_split(tokens_dir: Path, split: str) -> np.memmap:
    return np.memmap(tokens_dir / f"{split}.bin", dtype=np.uint16, mode="r")


def transpose_windows(x: torch.Tensor, semitones: int, gen: torch.Generator) -> torch.Tensor:
    """Shift every NOTE_ON/NOTE_OFF id in each row by a random k per row,
    k in [-semitones, +semitones]. k is clamped per row so no note leaves the
    88-key range; non-note tokens are untouched. Relies on NOTE_ON and NOTE_OFF
    being one contiguous 2*N_PITCH block (asserted in vocab tests)."""
    if semitones <= 0:
        return x
    B = x.shape[0]
    k = torch.randint(-semitones, semitones + 1, (B, 1), generator=gen,
                      device=x.device)
    is_note = (x >= V.NOTE_ON_OFF) & (x < V.TIME_SHIFT_OFF)
    # within-octave-block position of each note (0..87 for on, 0..87 for off)
    on = is_note & (x < V.NOTE_OFF_OFF)
    off = is_note & (x >= V.NOTE_OFF_OFF)
    pitch_pos = torch.where(on, x - V.NOTE_ON_OFF, x - V.NOTE_OFF_OFF)
    new_pos = pitch_pos + k
    # clamp k per row so the extreme pitches used in that row stay in [0, 87]
    lo = torch.where(is_note, pitch_pos, torch.full_like(x, V.N_PITCH)).amin(1, keepdim=True)
    hi = torch.where(is_note, pitch_pos, torch.full_like(x, -1)).amax(1, keepdim=True)
    k = k.clamp(min=-lo, max=(V.N_PITCH - 1) - hi)
    new_pos = pitch_pos + k
    shifted = torch.where(on, V.NOTE_ON_OFF + new_pos,
                          torch.where(off, V.NOTE_OFF_OFF + new_pos, x))
    return torch.where(is_note, shifted, x)


def get_batch(data: np.memmap, bs: int, block: int, device: str,
              transpose: int, gen: torch.Generator,
              anchors: np.ndarray | None = None,
              anchor_frac: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample bs windows of length block+1. When `anchors` (piece-start offsets)
    is given, a fraction of the windows begin exactly at a piece boundary — so
    the leading <COMPOSER> BOS tokens are in context and the composer signal can
    actually reach the events. Anchors within block of the stream end fall back
    to uniform sampling."""
    ix = np.random.randint(0, len(data) - block - 1, size=bs)
    if anchors is not None and anchor_frac > 0:
        n_anchor = int(round(bs * anchor_frac))
        if n_anchor:
            valid = anchors[anchors < len(data) - block - 1]
            if len(valid):
                ix[:n_anchor] = np.random.choice(valid, size=n_anchor)
    batch = np.stack([np.asarray(data[i:i + block + 1], dtype=np.int64) for i in ix])
    t = torch.from_numpy(batch).to(device, non_blocking=True)
    if transpose:
        t = transpose_windows(t, transpose, gen)
    return t[:, :-1], t[:, 1:]


@torch.no_grad()
def estimate_loss(model, splits, bs, block, device, dtype_ctx, eval_iters) -> dict:
    was_training = model.training
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(data, bs, block, device, 0, None)
            with dtype_ctx():
                _, loss = model(xb, targets=yb)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    if was_training:
        model.train()
    return out


def learning_rate(step: int, max_steps: int, cfg: dict) -> float:
    warmup = int(cfg.get("warmup_steps", 100))
    lr = float(cfg["lr"])
    min_lr = float(cfg.get("min_lr", lr * 0.1))
    if step < warmup:
        return lr * step / max(1, warmup)
    if step > max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def write_samples(model, out_dir: Path, step: int, device: str, temps, n_tokens: int,
                  render: bool) -> None:
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    for temp in temps:
        ids = generate_tokens(model, [V.composer(0), V.BOS], n_tokens,
                              temperature=temp, top_p=0.95, device=device, seed=step)
        mid = samples_dir / f"step{step:06d}_t{temp:.2f}.mid"
        decode(ids).dump_midi(str(mid))
        if render:
            try:
                from .render import midi_to_wav
                midi_to_wav(mid, mid.with_suffix(".wav"))
            except Exception as e:  # rendering is best-effort; never kill training
                print(f"  (render skipped: {e})")
    model.train()


def save_checkpoint(path: Path, model, optimizer, step, model_config, run_config,
                    train_cfg, losses) -> None:
    torch.save({"step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": asdict(model_config), "run_config": run_config,
                "train_config": train_cfg, "losses": losses,
                "rng_state": rng_state()}, path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a MusicGPT.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tokens-dir", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--resume", type=Path, default=None,
                    help="continue a run: restore weights+optimizer+step+RNG")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="fine-tune: load weights only, fresh optimizer/schedule "
                         "at step 0 (uses the base checkpoint's model architecture)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--render-samples", action="store_true",
                    help="also render sample .wavs (needs fluidsynth)")
    args = ap.parse_args()

    run_config = json.loads(args.config.read_text(encoding="utf-8"))
    train_cfg = dict(run_config["train"])
    model_cfg = dict(run_config["model"])
    tokens_dir = args.tokens_dir or Path(run_config["tokens_dir"])
    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    seed = int(train_cfg.get("seed", 20260703))
    set_seed(seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    autocast = device_type == "cuda" and train_cfg.get("precision") == "bf16"

    def dtype_ctx():
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                              enabled=autocast)

    # vocab guard: refuse token streams built under a different layout
    manifest = json.loads((tokens_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["vocab_size"] != V.VOCAB_SIZE:
        raise SystemExit(f"stale tokens: manifest vocab {manifest['vocab_size']} "
                         f"!= current {V.VOCAB_SIZE}; re-run prepare.py")

    if args.resume and args.init_from:
        raise SystemExit("--resume and --init-from are mutually exclusive")

    checkpoint = None
    start_step = 0
    last_losses = None
    best_val = float("inf")
    # both paths must build the model with the base checkpoint's architecture
    base = args.resume or args.init_from
    if base:
        checkpoint = torch.load(base, map_location=device, weights_only=False)
        model_cfg = {k: v for k, v in checkpoint["model_config"].items()
                     if k != "vocab_size"}
        if args.resume:  # continue: also restore step/loss/best-val
            start_step = int(checkpoint.get("step", 0))
            last_losses = checkpoint.get("losses")
            best_val = float(checkpoint.get("best_val", best_val))

    model_config = GPTConfig(vocab_size=V.VOCAB_SIZE, **model_cfg)
    model = MusicGPT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["lr"]),
        betas=(0.9, 0.95), weight_decay=float(train_cfg.get("weight_decay", 0.1)))
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        if args.resume:  # fine-tune (init-from) starts with a fresh optimizer + RNG
            optimizer.load_state_dict(checkpoint["optimizer"])
            restore_rng_state(checkpoint.get("rng_state"))
        else:
            print(f"fine-tuning from {args.init_from} (weights only, fresh schedule)")

    splits = {"train": load_split(tokens_dir, "train"),
              "validation": load_split(tokens_dir, "validation")}
    max_steps = int(train_cfg["max_steps"])
    bs = int(train_cfg["batch_size"])
    grad_accum = int(train_cfg.get("grad_accum", 1))
    block = int(model_cfg["block_size"])
    eval_interval = int(train_cfg.get("eval_interval", 250))
    sample_interval = int(train_cfg.get("sample_interval", 500))
    ckpt_interval = int(train_cfg.get("checkpoint_interval", max_steps))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    transpose = int(train_cfg.get("transpose_semitones", 0))
    temps = train_cfg.get("sample_temperatures", [0.8, 0.95, 1.1])
    sample_tokens = int(train_cfg.get("sample_tokens", 1024))

    # composer conditioning: anchor a fraction of windows at piece starts so the
    # leading <COMPOSER> BOS is in context (random windows almost never hit it)
    anchor_frac = float(train_cfg.get("anchor_frac", 0.0))
    anchors = None
    if anchor_frac > 0:
        anchors = np.array([p["offset"] for p in manifest["splits"]["train"]["pieces"]],
                           dtype=np.int64)
        print(f"composer conditioning: anchor_frac={anchor_frac} "
              f"over {len(anchors)} piece starts")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "resolved_config.json").write_text(json.dumps({
        "run_config": run_config, "train": train_cfg, "model": asdict(model_config),
        "tokens_dir": str(tokens_dir), "device": device,
        "resume": str(args.resume) if args.resume else None}, indent=2),
        encoding="utf-8")

    print(f"device={device} params={model.num_params()/1e6:.2f}M vocab={V.VOCAB_SIZE} "
          f"block={block} train_tokens={len(splits['train']):,} "
          f"transpose=±{transpose} autocast={autocast}")

    aug_gen = torch.Generator(device=device).manual_seed(seed + 1)
    metrics_path = args.out_dir / "metrics.jsonl"
    model.train()
    start = time.time()
    for step in range(start_step + 1, max_steps + 1):
        lr = learning_rate(step, max_steps, train_cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(grad_accum):
            xb, yb = get_batch(splits["train"], bs, block, device, transpose, aug_gen,
                               anchors=anchors, anchor_frac=anchor_frac)
            with dtype_ctx():
                _, loss = model(xb, targets=yb)
                loss = loss / grad_accum
            loss.backward()
            total += loss.item()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step == 1 or step % eval_interval == 0 or step == max_steps:
            last_losses = estimate_loss(model, splits, bs, block, device, dtype_ctx,
                                        args.eval_iters)
            elapsed = time.time() - start
            improved = last_losses["validation"] < best_val
            best_val = min(best_val, last_losses["validation"])
            print(f"step={step} lr={lr:.2e} train={last_losses['train']:.4f} "
                  f"val={last_losses['validation']:.4f}"
                  f"{' *best' if improved else ''} | {elapsed:.0f}s")
            append_jsonl(metrics_path, {
                "step": step, "lr": lr, "train_loss": last_losses["train"],
                "validation_loss": last_losses["validation"], "best_val": best_val,
                "elapsed_sec": round(elapsed, 1),
                "tokens_seen": step * bs * block * grad_accum})
            if improved:
                save_checkpoint(args.out_dir / "checkpoint_best.pt", model, optimizer,
                                step, model_config, run_config, train_cfg, last_losses)

        if step % sample_interval == 0 or step == max_steps:
            write_samples(model, args.out_dir, step, device, temps, sample_tokens,
                          args.render_samples)

        if step % ckpt_interval == 0 or step == max_steps:
            save_checkpoint(args.out_dir / f"checkpoint_step_{step:06d}.pt", model,
                            optimizer, step, model_config, run_config, train_cfg,
                            last_losses)

    print(f"done. best_val={best_val:.4f}  wrote {args.out_dir}")


if __name__ == "__main__":
    main()
