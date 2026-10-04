"""Record your dataset: shows one sentence at a time, records you reading it, and saves a 16 kHz WAV.

Run it from the whisper-finetune folder:

    uv run python record.py                  # record every sentence that isn't recorded yet
    uv run python record.py --redo s042      # record sentence s042 again (replaces the old take)
    uv run python record.py --take 2         # a second pass over every sentence, saved as take 2
    uv run python record.py --list-devices   # list the microphones on this computer
    uv run python record.py --device 3       # record with microphone number 3 from that list

Progress is saved after every sentence, so you can stop at any time (q or Ctrl+C) and run the same
command later to continue where you left off.
"""

import argparse
import csv
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16_000
ROOT = Path(os.environ.get("WHISPER_FT_ROOT", Path(__file__).resolve().parent))
DATA_DIR = ROOT / "data"
METADATA_FIELDS = ["file", "sentence_id", "take", "category", "text", "duration_s", "peak", "recorded_at"]


def load_sentences(data_dir):
    with open(data_dir / "sentences.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_metadata(data_dir):
    path = data_dir / "metadata.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_recording(data_dir, sentence, take, audio):
    """Writes the WAV file and adds (or replaces) its row in metadata.csv. Returns the row."""
    file = f"recordings/{sentence['sentence_id']}_take{take}.wav"
    path = data_dir / file
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")

    row = {
        "file": file,
        "sentence_id": sentence["sentence_id"],
        "take": str(take),
        "category": sentence["category"],
        "text": sentence["text"],
        "duration_s": f"{len(audio) / SAMPLE_RATE:.2f}",
        "peak": f"{peak_level(audio):.3f}",
        "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    rows = {r["file"]: r for r in load_metadata(data_dir)}
    rows[file] = row
    tmp = data_dir / "metadata.csv.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda r: (r["sentence_id"], int(r["take"]))))
    os.replace(tmp, data_dir / "metadata.csv")
    return row


def peak_level(audio):
    return float(np.abs(audio).max()) if len(audio) else 0.0


def problems_with(audio):
    seconds = len(audio) / SAMPLE_RATE
    peak = peak_level(audio)
    problems = []
    if seconds < 0.5:
        problems.append("Very short. Did it record anything?")
    if seconds > 29.5:
        problems.append("Too long: Whisper only hears the first 30 seconds.")
    if peak < 0.03:
        problems.append("Very quiet: move closer to the microphone or raise its input level.")
    if peak >= 0.99:
        problems.append("Clipped (too loud): move back a little or lower the input level.")
    return problems


def record_until_enter(device):
    chunks = []
    dropped = []

    def on_audio(indata, frames, time, status):
        if status:
            dropped.append(str(status))
        chunks.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device, callback=on_audio):
        input()
    if dropped:
        print("  ! Some audio may have been dropped while recording. Consider redoing this one.")
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def ask(prompt, allowed):
    while True:
        answer = input(prompt).strip().lower()
        if answer in allowed:
            return answer
        print("  Type one of: " + ", ".join(a or "Enter" for a in sorted(allowed)))


def record_one(sentence, take, position, total, device, data_dir):
    """Records one sentence. Returns "saved", "skipped", or "quit"."""
    print()
    print(f"[{position}/{total}]  {sentence['sentence_id']}  ({sentence['category']}, take {take})")
    print()
    print(f"    {sentence['text']}")
    print()
    answer = ask("  Enter = start recording, s = skip, q = quit: ", {"", "s", "q"})
    if answer == "s":
        return "skipped"
    if answer == "q":
        return "quit"

    while True:
        print("  >> Recording... press Enter when you have finished the sentence.")
        audio = record_until_enter(device)
        print(f"  {len(audio) / SAMPLE_RATE:.1f} s, peak level {peak_level(audio):.2f}")
        for problem in problems_with(audio):
            print(f"  ! {problem}")

        answer = ask("  Enter = keep, p = play back, r = redo, q = quit without saving: ", {"", "p", "r", "q"})
        while answer == "p":
            sd.play(audio, SAMPLE_RATE)
            sd.wait()
            answer = ask("  Enter = keep, p = play back, r = redo, q = quit without saving: ", {"", "p", "r", "q"})
        if answer == "r":
            continue
        if answer == "q":
            return "quit"
        save_recording(data_dir, sentence, take, audio)
        return "saved"


def main():
    parser = argparse.ArgumentParser(description="Record yourself reading the sentences in data/sentences.csv.")
    parser.add_argument("--take", type=int, default=1, help="which take to record (default: 1)")
    parser.add_argument("--redo", metavar="SENTENCE_ID", help="record this sentence again, for example s042")
    parser.add_argument("--device", help="microphone number or part of its name (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    args = parser.parse_args()

    sys.stdout.reconfigure(errors="replace")
    if args.list_devices:
        print(sd.query_devices())
        print("\n'>' marks the default input (microphone), '<' the default output.")
        return

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    try:
        sd.check_input_settings(device=device, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    except Exception as error:
        sys.exit(f"This microphone can't record at 16 kHz ({error}).\n"
                 "Run with --list-devices and pick a device whose host API is MME.")

    sentences = load_sentences(DATA_DIR)
    if args.redo:
        todo = [s for s in sentences if s["sentence_id"] == args.redo]
        if not todo:
            sys.exit(f"There is no sentence {args.redo} in data/sentences.csv.")
    else:
        done = {(r["sentence_id"], r["take"]) for r in load_metadata(DATA_DIR)}
        todo = [s for s in sentences if (s["sentence_id"], str(args.take)) not in done]

    placeholders = [s for s in todo if "{" in s["text"]]
    if placeholders:
        print(f"Skipping {len(placeholders)} sentence(s) that still contain placeholders like {{PERSON_1}}.")
        print("Replace them in data/sentences.csv (notebook 02, Part 1), then run this again to record them.")
        todo = [s for s in todo if "{" not in s["text"]]

    print(f"Microphone: {sd.query_devices(device, kind='input')['name']}")
    print(f"{len(todo)} sentence(s) to record. Leave a short pause before and after each sentence.")
    saved = 0
    try:
        for position, sentence in enumerate(todo, start=1):
            result = record_one(sentence, args.take, position, len(todo), device, DATA_DIR)
            if result == "quit":
                break
            saved += result == "saved"
    except KeyboardInterrupt:
        print()
    print(f"\nSaved {saved} recording(s) this session. {len(load_metadata(DATA_DIR))} recordings in total.")
    print("Run the same command again to continue.")


if __name__ == "__main__":
    main()
