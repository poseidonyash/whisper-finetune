"""Shared code for notebooks 02 to 04.

The top half is provided: folder paths, loaders for the files in data/, and small helpers.
The bottom half is yours. At the end of notebooks 02 and 03 you paste the functions you wrote
into it, so later notebooks can import them instead of copying them.

Every path below is relative to this folder. Set the WHISPER_FT_ROOT environment variable to
use a different folder, for example a copy of the project on Colab.
"""

import csv
import io
import json
import os
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperProcessor

SAMPLE_RATE = 16_000
MODEL_NAME = "openai/whisper-tiny.en"

ROOT = Path(os.environ.get("WHISPER_FT_ROOT", Path(__file__).resolve().parent))
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
SOT = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
NO_TIMESTAMPS = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
EOT = processor.tokenizer.eos_token_id


def load_sentences():
    """data/sentences.csv as a list of dicts with the keys sentence_id, category, and text (all str)."""
    with open(DATA_DIR / "sentences.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_vocabulary():
    """data/vocabulary.txt as a list of str, one key term per line."""
    lines = (DATA_DIR / "vocabulary.txt").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def load_metadata():
    """data/metadata.csv (written by record.py) as a list of dicts, one per recording.

    A CSV file stores everything as text, so this converts the number columns: take becomes an int,
    duration_s and peak become floats. It also adds "path", the full path of the WAV file as a str.
    """
    path = DATA_DIR / "metadata.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["take"] = int(row["take"])
        row["duration_s"] = float(row["duration_s"])
        row["peak"] = float(row["peak"])
        row["path"] = str(DATA_DIR / row["file"])
    return rows


def read_wav(path):
    """Reads a mono 16 kHz WAV file. Returns the waveform: a 1-D float32 NumPy array with values from -1 to 1."""
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError(f"{path} should be mono at {SAMPLE_RATE} Hz, but it's {sample_rate} Hz with shape {audio.shape}")
    return audio


def load_splits():
    """data/splits.json (written in notebook 02) as a dict. Its "train", "val", and "test" keys each hold a list of sentence ids."""
    return load_json(DATA_DIR / "splits.json")


def rows_for_split(rows, splits, name):
    """The rows whose sentence_id belongs to split `name` ("train", "val", or "test")."""
    ids = set(splits[name])
    return [row for row in rows if row["sentence_id"] in ids]


def contains_term(text, term):
    """True if `term` appears in `text` as whole words, ignoring case and punctuation.

    Both strings go through Whisper's text normalizer first, so "LoRA," matches the term "lora",
    and the term "WER" doesn't match the word "power".
    """
    normalize = processor.tokenizer.normalize
    return f" {normalize(term)} " in f" {normalize(text)} "


def load_librispeech():
    """The 73 LibriSpeech clips from notebook 01, as a list of (waveform, transcript) tuples."""
    from datasets import Audio, load_dataset

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    clips = []
    for example in ds:
        audio, _ = sf.read(io.BytesIO(example["audio"]["bytes"]), dtype="float32")
        clips.append((audio, example["text"]))
    return clips


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------
# Your code. Replace each stub with the function you wrote, when its notebook tells you to.
# ---------------------------------------------------------------------------------------------

# From notebook 02, Part 5.


def transcribe(model, audios, batch_size=8):
    results = []
    for start in range(0, len(audios), batch_size):
        batch = audios[start:start+batch_size]
        inputs = processor.feature_extractor(batch, sampling_rate = 16000, return_tensors="pt")
        with torch.no_grad():
            predicted_ids = model.generate(inputs.input_features)
        decoded_strings = processor.batch_decode(predicted_ids, skip_special_tokens = True)
        results.extend(decoded_strings)
    return results

def corpus_wer(references, hypotheses):
    corpus_wer = jiwer.wer(
        [processor.tokenizer.normalize(r) for r in references],
        [processor.tokenizer.normalize(h) for h in hypotheses],
    )
    return corpus_wer

def term_recall(references, hypotheses, terms):
    hits = 0
    total = 0
    misses = []
    for ref, hyp in zip(references, hypotheses):
        for term in terms:
            if contains_term(ref, term):
                total+=1 
                if contains_term(hyp, term):
                    hits +=1 
                else:
                    misses.append((term, ref, hyp))
        recall = hits / total if total > 0 else None
    return {"hits": hits, "total": total, "recall": recall, "misses": misses}

# From notebook 03.


def prepare_example(row):
    output = {}
    audio = read_wav(row["path"])
    output["input_features"] = processor.feature_extractor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="np"
    ).input_features[0]
    text = row["text"]
    output["labels"] = processor.tokenizer(" " + text).input_ids
    return output


def collate(batch):
    output = {}
    features = np.stack([ex["input_features"] for ex in batch])
    output["input_features"] = torch.from_numpy(features)
    labels = processor.tokenizer.pad(
        [{"input_ids": ex["labels"][1:]} for ex in batch], return_tensors="pt"
    )
    output["labels"] = labels["input_ids"].masked_fill(labels["attention_mask"] == 0, -100)
    return output


def train_one_epoch(model, loader, optimizer):
    losses = []
    model.train()
    for batch in loader:
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(outputs.loss.item())
    return sum(losses) / len(losses)