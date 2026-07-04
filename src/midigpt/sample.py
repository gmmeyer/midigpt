"""Generate MIDI from a checkpoint.

Unconditional (primed with a composer token + BOS) or continuation mode
(primed with the first N seconds of a real MIDI file). Cacheless generation —
fine at this model size.

    uv run python -m midigpt.sample --ckpt checkpoints/run0/checkpoint_best.pt --out out.mid
    uv run python -m midigpt.sample --ckpt ... --continue-from piece.mid --prime-seconds 30 --out cont.mid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import vocab as V
from .model import GPTConfig, MusicGPT
from .tokenizer import cut_at_seconds, decode, duration_seconds, encode_file


@torch.no_grad()
def generate_tokens(model: MusicGPT, prime: list[int], max_new: int,
                    temperature: float = 0.95, top_p: float = 0.95,
                    device: str = "cpu", seed: int | None = None) -> list[int]:
    was_training = model.training
    model.eval()
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    x = torch.tensor([prime], dtype=torch.long, device=device)
    out = list(prime)
    for _ in range(max_new):
        x_cond = x if x.size(1) <= model.cfg.block_size else x[:, -model.cfg.block_size:]
        logits, _ = model(x_cond)
        logits = logits[0, -1] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_p, sorted_ix = torch.sort(probs, descending=True)
            keep = torch.cumsum(sorted_p, dim=-1) - sorted_p < top_p
            keep[0] = True  # always keep the top token
            probs = torch.zeros_like(probs).scatter(0, sorted_ix[keep], sorted_p[keep])
            probs /= probs.sum()
        nxt = int(torch.multinomial(probs, 1, generator=gen).item())
        out.append(nxt)
        if nxt == V.EOS:
            break
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    if was_training:
        model.train()
    return out


def load_model(ckpt_path: Path, device: str) -> tuple[MusicGPT, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MusicGPT(GPTConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _composer_token(name: str | None, tokens_dir: Path, rng: np.random.Generator) -> int:
    manifest_path = tokens_dir / "manifest.json"
    if manifest_path.exists():
        composers: dict[str, int] = json.loads(
            manifest_path.read_text(encoding="utf-8"))["composers"]
        if name:
            matches = [i for c, i in composers.items() if name.lower() in c.lower()]
            if not matches:
                raise SystemExit(f"no composer matching {name!r}; "
                                 f"known: {sorted(composers)}")
            return V.composer(matches[0])
        return V.composer(int(rng.choice(sorted(composers.values()))))
    if name:
        raise SystemExit(f"--composer needs {manifest_path} for the name->id map")
    return V.composer(0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample MIDI from a checkpoint.")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("sample.mid"))
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--composer", default=None,
                    help="substring match against manifest composer names")
    ap.add_argument("--tokens-dir", type=Path, default=Path("data/tokens"))
    ap.add_argument("--continue-from", type=Path, default=None,
                    help="MIDI file to continue")
    ap.add_argument("--prime-seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--render", action="store_true", help="also render a .wav")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.ckpt, device)
    rng = np.random.default_rng(args.seed)

    if args.continue_from:
        events = cut_at_seconds(encode_file(args.continue_from), args.prime_seconds)
        prime = [_composer_token(args.composer, args.tokens_dir, rng), V.BOS,
                 *events.tolist()]
    else:
        prime = [_composer_token(args.composer, args.tokens_dir, rng), V.BOS]

    ids = generate_tokens(model, prime, args.tokens, args.temperature,
                          args.top_p, device, args.seed)
    decode(ids).dump_midi(str(args.out))
    print(f"{args.out}: {len(ids)} tokens, {duration_seconds(ids):.1f}s of music "
          f"(prime {len(prime)} tokens)")

    if args.render:
        from .render import midi_to_wav
        wav = args.out.with_suffix(".wav")
        midi_to_wav(args.out, wav)
        print(f"rendered {wav}")


if __name__ == "__main__":
    main()
