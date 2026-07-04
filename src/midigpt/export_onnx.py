"""Export a trained MusicGPT to ONNX for in-browser inference.

Produces web/model.onnx (dynamic sequence length, returns last-position logits)
and web/config.json (vocab layout + composer map) so the JavaScript side can run
the autoregressive loop, decode event tokens to notes, and play them. Validates
the exported graph against PyTorch (argmax must agree — that's what sampling
depends on).

    uv run --with onnx --with onnxruntime python -m midigpt.export_onnx \
        --checkpoint checkpoints/run0/checkpoint_best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import vocab as V
from .model import GPTConfig, MusicGPT


class _LastLogits(torch.nn.Module):
    """Return only the last position's logits — all the JS decode loop needs."""

    def __init__(self, model: MusicGPT):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(idx)
        return logits[:, -1, :]


class _ExplicitRMSNorm(torch.nn.Module):
    """Drop-in for our RMSNorm using primitive ops (opset 17's legacy exporter
    can't emit aten::rms_norm). Same math given the same weight and eps."""

    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.detach().clone())
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


def _swap_rmsnorm(module: torch.nn.Module) -> None:
    from .model import RMSNorm
    for name, child in list(module.named_children()):
        if isinstance(child, RMSNorm):
            setattr(module, name, _ExplicitRMSNorm(child.weight.data, child.eps))
        else:
            _swap_rmsnorm(child)


def build_web_config(manifest_path: Path | None, block_size: int,
                     web_context_cap: int) -> dict:
    cfg = {
        "vocab_size": V.VOCAB_SIZE,
        "specials": {"PAD": V.PAD, "BOS": V.BOS, "EOS": V.EOS},
        "layout": {
            "COMPOSER_OFF": V.COMPOSER_OFF, "NOTE_ON_OFF": V.NOTE_ON_OFF,
            "NOTE_OFF_OFF": V.NOTE_OFF_OFF, "TIME_SHIFT_OFF": V.TIME_SHIFT_OFF,
            "VELOCITY_OFF": V.VELOCITY_OFF,
        },
        "n_pitch": V.N_PITCH, "n_shift": V.N_SHIFT, "n_velocity": V.N_VELOCITY,
        "pitch_min": V.PITCH_MIN, "time_step_ms": V.TIME_STEP_MS,
        "block_size": block_size,
        "web_context_cap": web_context_cap,  # JS sliding-window cap (speed vs coherence)
        "composers": {},
    }
    if manifest_path and manifest_path.exists():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg["composers"] = man.get("composers", {})
    return cfg


def export(checkpoint: str, out_dir: str = "web", opset: int = 17,
           web_context_cap: int = 512, manifest: str | None = None) -> dict:
    ck = torch.load(checkpoint, weights_only=False, map_location="cpu")
    model_cfg = {k: v for k, v in ck["model_config"].items() if k != "vocab_size"}
    model = MusicGPT(GPTConfig(vocab_size=V.VOCAB_SIZE, **model_cfg))
    model.load_state_dict(ck["model"])
    model.eval()
    # build the RoPE cache eagerly on CPU/f32 so it exports as a constant initializer
    model._ensure_rope(torch.device("cpu"), torch.float32)
    _swap_rmsnorm(model)
    wrap = _LastLogits(model).eval()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "model.onnx"

    dummy = torch.tensor([[V.composer(0), V.BOS, V.velocity(80), V.note_on(60),
                           V.time_shift(10), V.note_off(60)]], dtype=torch.long)
    torch.onnx.export(
        wrap, (dummy,), str(onnx_path), input_names=["idx"], output_names=["logits"],
        dynamic_axes={"idx": {1: "T"}, "logits": {0: "B"}}, opset_version=opset,
        dynamo=False)

    man_path = Path(manifest) if manifest else (Path("data/tokens/manifest.json"))
    web_cfg = build_web_config(man_path, model_cfg["block_size"], web_context_cap)
    (out / "config.json").write_text(json.dumps(web_cfg, indent=2), encoding="utf-8")

    # ---- validate against PyTorch on variable-length contexts ----
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    max_diff = 0.0
    for T in (6, 19, 64, 200):
        ctx = [V.composer(0), V.BOS]
        for _ in range(T):
            ctx.append(int(rng.integers(V.NOTE_ON_OFF, V.VOCAB_SIZE)))
        idx = torch.tensor([ctx], dtype=torch.long)
        with torch.no_grad():
            ref = wrap(idx).numpy()
        got = sess.run(None, {"idx": idx.numpy()})[0]
        max_diff = max(max_diff, float(np.abs(ref - got).max()))
        assert int(ref.argmax()) == int(got.argmax()), f"argmax mismatch at T={T}"

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"exported {onnx_path} ({size_mb:.1f} MB), vocab={V.VOCAB_SIZE}, "
          f"block_size={web_cfg['block_size']}, composers={len(web_cfg['composers'])}")
    print(f"onnx-vs-torch max logit diff = {max_diff:.2e} (argmax agrees on all lengths)")
    return {"onnx": str(onnx_path), "max_diff": max_diff, "size_mb": size_mb}


def _main() -> None:
    ap = argparse.ArgumentParser(description="Export MusicGPT to ONNX for the web demo.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="web")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--web-context-cap", type=int, default=512)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()
    export(args.checkpoint, out_dir=args.out_dir, opset=args.opset,
           web_context_cap=args.web_context_cap, manifest=args.manifest)


if __name__ == "__main__":
    _main()
