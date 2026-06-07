"""Publish the curated dataset to the HuggingFace Hub as a public dataset.

Usage:
    huggingface-cli login                 # one-time, stores token
    python src/publish_hf.py --repo your-username/sarvam-tts-dataset

Optional flags:
    --private             create the repo as private (default: public)
    --no-push             build the dataset locally but skip the push
    --commit-message TXT  override the default commit message
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import Audio, Dataset, Features, Value
from huggingface_hub import HfApi

from config import CLIPS_DIR, MANIFEST_CSV, setup_logging

log = setup_logging("publish_hf")


KEEP_COLUMNS = [
    "clip_id",
    "audio",
    "transcription",
    "language",
    "emotion",
    "speaker_id",
    "duration_sec",
    "video_id",
    "source_url",
    "start_sec",
    "end_sec",
]


def build_dataset() -> Dataset:
    if not MANIFEST_CSV.exists():
        log.error("manifest not found at %s; run load_manifest.py first", MANIFEST_CSV)
        sys.exit(1)

    df = pd.read_csv(MANIFEST_CSV)
    log.info("manifest rows: %d", len(df))

    df["transcription"] = df.get("transcription", "").astype("object").fillna("")
    df["transcribed"] = df.get("transcribed", "").astype("object").fillna("")
    df["selected"] = df.get("selected", "").astype("object").fillna("")

    df = df[df["selected"].astype(str).str.upper() == "Y"]
    df = df[df["transcribed"].astype(str).str.upper() == "Y"]
    df = df[df["transcription"].str.strip() != ""]
    log.info("rows selected & transcribed: %d", len(df))

    audio_paths = []
    keep_mask = []
    for _, r in df.iterrows():
        p = CLIPS_DIR / f"{r['clip_id']}.wav"
        if p.exists() and p.stat().st_size > 0:
            audio_paths.append(str(p))
            keep_mask.append(True)
        else:
            log.warning("missing audio for %s, dropping", r["clip_id"])
            keep_mask.append(False)

    df = df.loc[keep_mask].reset_index(drop=True)
    df["audio"] = audio_paths
    log.info("rows with audio on disk: %d", len(df))

    df = df[KEEP_COLUMNS]
    df["duration_sec"] = df["duration_sec"].astype("int32")
    df["start_sec"] = df["start_sec"].astype("int32")
    df["end_sec"] = df["end_sec"].astype("int32")

    features = Features({
        "clip_id": Value("string"),
        "audio": Audio(sampling_rate=16000),
        "transcription": Value("string"),
        "language": Value("string"),
        "emotion": Value("string"),
        "speaker_id": Value("string"),
        "duration_sec": Value("int32"),
        "video_id": Value("string"),
        "source_url": Value("string"),
        "start_sec": Value("int32"),
        "end_sec": Value("int32"),
    })

    ds = Dataset.from_pandas(df, features=features, preserve_index=False)

    # Stats
    total_sec = int(df["duration_sec"].sum())
    log.info("total audio: %ds (%.1f min)", total_sec, total_sec / 60)
    for lang in sorted(df["language"].unique()):
        sub = df[df["language"] == lang]
        log.info("  %s: %d clips, %ds (%.1f min)",
                 lang, len(sub), int(sub["duration_sec"].sum()),
                 sub["duration_sec"].sum() / 60)
    return ds


DATASET_CARD = """---
license: cc-by-4.0
language:
  - en
  - hi
size_categories:
  - n<1K
task_categories:
  - text-to-speech
  - automatic-speech-recognition
tags:
  - tts
  - indian-english
  - hindi
  - emotion
  - speech
---

# Indian English + Hindi TTS Mini Dataset

A small (~60 min) speech corpus assembled for TTS training, sourced from
public YouTube content and transcribed with the Sarvam `saaras:v3` model.

## Summary

| Field | Value |
|---|---|
| Total duration | ~60 min |
| Indian English (en-IN) | ~30 min |
| Hindi (hi-IN) | ~30 min |
| Audio | mono, 16 kHz, WAV |
| Per-clip duration | ≤ 30 s |
| Emotion tags | Angry, Calm, Excited, Formal, Happy, Instruction, Motivation, Narration, Neutral, News, Sad, Surprised, Whisper |

## Schema

Each row contains:

- `clip_id` — deterministic id, `{video_id}_{start_sec}_{end_sec}`
- `audio` — 16 kHz mono WAV (HuggingFace `Audio` feature)
- `transcription` — Sarvam `saaras:v3` transcript, lightly normalized
- `language` — `en-IN` or `hi-IN`
- `emotion` — one of the categories above
- `speaker_id` — set to source `video_id` (one speaker per source video)
- `duration_sec`, `start_sec`, `end_sec` — integers, seconds
- `video_id`, `source_url` — YouTube provenance

## Notes & limitations

- Clips longer than 30 s in the source spec were split into back-to-back
  ≤30 s pieces so they fit Sarvam's sync ASR limit.
- Transcripts come from an ASR model; not all are perfect — manual review
  on a subset corrected obvious errors but the rest are model output.
- `speaker_id` is approximated as the YouTube `video_id`. Most rows are
  single-speaker monologues; we did not run diarization to verify across
  the entire set.
- Emotion tags come from the curator's labelling of each source clip.

## License

Audio is sourced from YouTube; rights remain with their original creators.
Transcriptions and metadata are released under CC-BY-4.0. Please consult
the original videos for redistribution constraints.
"""


def write_dataset_card(repo_id: str, api: HfApi):
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    log.info("dataset card uploaded to %s", repo_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HF dataset repo, e.g. user/sarvam-tts")
    parser.add_argument("--private", action="store_true", help="create as private")
    parser.add_argument("--no-push", action="store_true", help="build but skip push")
    parser.add_argument("--commit-message", default="Initial dataset upload")
    args = parser.parse_args()

    ds = build_dataset()

    if args.no_push:
        log.info("--no-push: dataset built (%d rows) but not pushed", len(ds))
        return

    api = HfApi()
    log.info("creating/verifying repo: %s (private=%s)", args.repo, args.private)
    api.create_repo(repo_id=args.repo, repo_type="dataset",
                    private=args.private, exist_ok=True)

    log.info("pushing dataset to %s ...", args.repo)
    ds.push_to_hub(args.repo, private=args.private,
                   commit_message=args.commit_message)

    write_dataset_card(args.repo, api)

    log.info("DONE. https://huggingface.co/datasets/%s", args.repo)


if __name__ == "__main__":
    main()
