"""Render MIDI to WAV via the fluidsynth CLI.

Install: `winget install FluidSynth.FluidSynth`, then set MIDIGPT_SOUNDFONT
to a piano .sf2, or drop one in data/soundfonts/.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

_SF_DIRS = (Path("data/soundfonts"), Path("C:/soundfonts"),
            Path(os.environ.get("ProgramData", "C:/ProgramData")) / "soundfonts")


def find_soundfont() -> Path:
    env = os.environ.get("MIDIGPT_SOUNDFONT")
    if env and Path(env).exists():
        return Path(env)
    for d in _SF_DIRS:
        if d.is_dir():
            fonts = sorted(d.glob("*.sf2")) + sorted(d.glob("*.sf3"))
            if fonts:
                return fonts[0]
    raise FileNotFoundError(
        "no soundfont found — set MIDIGPT_SOUNDFONT or put a .sf2 in data/soundfonts/")


def find_fluidsynth() -> str:
    # prefer a project-bundled portable build, then fall back to PATH
    local = sorted(Path("tools").glob("fluidsynth*/**/bin/fluidsynth.exe"))
    if local:
        return str(local[0])
    exe = shutil.which("fluidsynth")
    if not exe:
        raise FileNotFoundError(
            "fluidsynth not found — expected a portable build under tools/ or "
            "fluidsynth on PATH")
    return exe


def midi_to_wav(midi: Path, wav: Path, soundfont: Path | None = None,
                sample_rate: int = 44100) -> None:
    sf = soundfont or find_soundfont()
    # ALL options must precede the positional soundfont/midi args — fluidsynth
    # rejects a flag that appears after them (and exits 0 anyway, so we can't
    # rely on the return code; we check the output file exists below).
    proc = subprocess.run(
        [find_fluidsynth(), "-ni", "-F", str(wav), "-r", str(sample_rate),
         str(sf), str(midi)],
        check=True, capture_output=True, text=True)
    if not wav.exists() or wav.stat().st_size == 0:
        raise RuntimeError(
            f"fluidsynth produced no audio for {midi}\n{proc.stdout}\n{proc.stderr}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MIDI -> WAV via fluidsynth.")
    ap.add_argument("midi", type=Path)
    ap.add_argument("wav", type=Path, nargs="?", default=None)
    ap.add_argument("--soundfont", type=Path, default=None)
    args = ap.parse_args()
    wav = args.wav or args.midi.with_suffix(".wav")
    midi_to_wav(args.midi, wav, args.soundfont)
    print(f"wrote {wav}")


if __name__ == "__main__":
    main()
