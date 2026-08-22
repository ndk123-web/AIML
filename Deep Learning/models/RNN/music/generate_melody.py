from pathlib import Path
import random
import shutil

from midi2audio import FluidSynth
from music21 import note, stream

melody = stream.Stream()
base_dir = Path(__file__).resolve().parent
sound_font = base_dir / "FluidR3_GM.sf2"
mid_path = base_dir / "melody.mid"
wav_path = base_dir / "melody.wav"

for _ in range(20):
    n = random.choice([60, 62, 64, 65, 67, 69])
    melody.append(note.Note(n, quarterLength=0.5))

melody.write("midi", fp=str(mid_path))

if not sound_font.exists():
    raise FileNotFoundError(f"SoundFont not found: {sound_font}")

if shutil.which("fluidsynth") is None:
    print(f"Saved MIDI to {mid_path}")
    print(
        "Skipping WAV rendering because the `fluidsynth` executable is not on PATH. "
        "Install Fluidsynth and rerun to create melody.wav."
    )
else:
    fs = FluidSynth(sound_font=str(sound_font))
    fs.midi_to_audio(str(mid_path), str(wav_path))
    print(f"Saved MIDI to {mid_path}")
    print(f"Saved WAV to {wav_path}")