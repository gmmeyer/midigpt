"""Tokenize MAESTRO into flat uint16 streams.

Reads the official CSV (train/validation/test split by piece), frames each
performance as `<COMPOSER> BOS ...events... EOS`, and concatenates per split:

    data/tokens/{train,validation,test}.bin  +  manifest.json

The manifest records the vocab layout, composer map, and per-piece offsets
(so later phases can anchor windows at piece starts or sample continuations
from held-out pieces).

    uv run python -m midigpt.prepare --download
    uv run python -m midigpt.prepare --single path\\to\\x.mid --out data/tokens-overfit
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import vocab as V
from .tokenizer import encode_file

MAESTRO_URL = ("https://storage.googleapis.com/magentadata/datasets/maestro/"
               "v3.0.0/maestro-v3.0.0-midi.zip")


def download(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "maestro-v3.0.0-midi.zip"
    if not zip_path.exists():
        print(f"downloading {MAESTRO_URL} ...")
        urllib.request.urlretrieve(MAESTRO_URL, zip_path)
    if not (root / "maestro-v3.0.0").exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root)


def _manifest_base() -> dict:
    return {
        "dtype": "uint16",
        "vocab_size": V.VOCAB_SIZE,
        "time_step_ms": V.TIME_STEP_MS,
        "n_shift": V.N_SHIFT,
        "n_velocity": V.N_VELOCITY,
        "pitch_range": [V.PITCH_MIN, V.PITCH_MAX],
        "layout": {
            "COMPOSER_OFF": V.COMPOSER_OFF,
            "NOTE_ON_OFF": V.NOTE_ON_OFF,
            "NOTE_OFF_OFF": V.NOTE_OFF_OFF,
            "TIME_SHIFT_OFF": V.TIME_SHIFT_OFF,
            "VELOCITY_OFF": V.VELOCITY_OFF,
        },
    }


def prepare_maestro(root: Path, out_dir: Path) -> None:
    csv_path = root / "maestro-v3.0.0" / "maestro-v3.0.0.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

    composers = sorted({r["canonical_composer"] for r in rows})
    assert len(composers) <= V.N_COMPOSER, (
        f"{len(composers)} composers > {V.N_COMPOSER} reserved ids")
    comp_id = {name: i for i, name in enumerate(composers)}

    streams: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    pieces: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    offsets = {"train": 0, "validation": 0, "test": 0}

    for r in tqdm(rows, desc="tokenizing"):
        split = r["split"]
        events = encode_file(root / "maestro-v3.0.0" / r["midi_filename"])
        frame = np.concatenate([
            np.array([V.composer(comp_id[r["canonical_composer"]]), V.BOS],
                     dtype=np.uint16),
            events,
            np.array([V.EOS], dtype=np.uint16),
        ])
        assert int(frame.max()) < V.VOCAB_SIZE
        streams[split].append(frame)
        pieces[split].append({
            "offset": offsets[split],
            "length": int(len(frame)),
            "composer": r["canonical_composer"],
            "title": r["canonical_title"],
            "midi_filename": r["midi_filename"],
        })
        offsets[split] += len(frame)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_base()
    manifest["composers"] = comp_id
    manifest["splits"] = {}
    for split, chunks in streams.items():
        tokens = np.concatenate(chunks)
        path = out_dir / f"{split}.bin"
        tokens.tofile(path)
        manifest["splits"][split] = {
            "path": path.name,
            "tokens": int(len(tokens)),
            "pieces": pieces[split],
        }
        print(f"{split}: {len(chunks)} pieces, {len(tokens):,} tokens -> {path}")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    print(f"wrote {out_dir / 'manifest.json'}")


def prepare_single(midi: Path, out_dir: Path) -> None:
    """One piece as both train and validation — the Phase-2 overfit set."""
    events = encode_file(midi)
    frame = np.concatenate([
        np.array([V.composer(0), V.BOS], dtype=np.uint16),
        events,
        np.array([V.EOS], dtype=np.uint16),
    ])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_base()
    manifest["composers"] = {"overfit": 0}
    manifest["splits"] = {}
    for split in ("train", "validation"):
        (out_dir / f"{split}.bin").write_bytes(frame.tobytes())
        manifest["splits"][split] = {
            "path": f"{split}.bin", "tokens": int(len(frame)),
            "pieces": [{"offset": 0, "length": int(len(frame)),
                        "composer": "overfit", "title": midi.name,
                        "midi_filename": str(midi)}],
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    print(f"{midi.name}: {len(frame):,} tokens -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tokenize MAESTRO -> uint16 .bin streams.")
    ap.add_argument("--root", type=Path, default=Path("data/maestro"))
    ap.add_argument("--out", type=Path, default=Path("data/tokens"))
    ap.add_argument("--download", action="store_true",
                    help="fetch + extract MAESTRO first if missing")
    ap.add_argument("--single", type=Path, default=None,
                    help="tokenize one MIDI file as an overfit set instead")
    args = ap.parse_args()

    if args.single:
        prepare_single(args.single, args.out)
        return
    if args.download:
        download(args.root)
    prepare_maestro(args.root, args.out)


if __name__ == "__main__":
    main()
