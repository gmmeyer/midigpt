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
    exe = shutil.which("fluidsynth")
    if not exe:
        raise FileNotFoundError(
            "fluidsynth not on PATH — `winget install FluidSynth.FluidSynth`")
    return exe


def midi_to_wav(midi: Path, wav: Path, soundfont: Path | None = None,
                sample_rate: int = 44100) -> None:
    sf = soundfont or find_soundfont()
    subprocess.run(
        [find_fluidsynth(), "-ni", str(sf), str(midi), "-F", str(wav),
         "-r", str(sample_rate)],
        check=True, capture_output=True)


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
