# Indian English + Hindi TTS Mini Dataset

A reproducible pipeline that assembles a ~60-minute, emotion-tagged,
single-speaker TTS training dataset from public YouTube content and
transcribes it with Sarvam's `saaras:v3` speech-to-text model.

The end product is:

- **~60 min** of clean, mono 16 kHz WAV clips
- **~30 min Indian English (en-IN)** and **~30 min Hindi (hi-IN)**
- **12 emotion / style categories** balanced across both languages
- **143 clips**, each **≤ 30 s** (so they fit the Sarvam sync ASR limit)
- A `manifest.csv` row per clip with transcript, language, emotion,
  speaker id, source URL and timestamps
- A small web UI for listening to clips and editing transcripts

It is published as a public HuggingFace dataset; this repository is the
code, configuration, and curation tracker behind it.

---

## Pipeline at a glance

```
data/sources.csv               ┐
  (clip tracker spreadsheet)   │
                               ▼
 src/load_manifest.py   ──►  data/manifest.csv
                               │  (URL parse, mm:ss → sec,
                               │   chunk to ≤30s,
                               │   60-min budget selection,
                               │   per-language / per-emotion balance)
                               ▼
 src/download.py        ──►  data/clips/*.wav
                               │  (yt-dlp full audio + ffmpeg trim
                               │   to mono 16 kHz at exact start..end)
                               ▼
 src/transcribe.py      ──►  data/manifest.csv (transcription column)
                               │  (Sarvam saaras:v3, mode=transcribe,
                               │   per-row language code,
                               │   retries + backoff, idempotent)
                               ▼
 src/web.py             ──►  http://127.0.0.1:5050
                               │  (review UI: audio + editable transcript,
                               │   filter by language/emotion/reviewed)
                               ▼
 src/publish_hf.py      ──►  https://huggingface.co/datasets/<your-repo>
```

Every script is **idempotent** — re-running picks up where you left off.
You can edit the source CSV, rerun `load_manifest.py`, and prior
transcriptions are carried forward automatically.

---

## Repository layout

```
.
├── README.md                  ← you are here
├── requirements.txt
├── .env.example               ← copy to .env and add SARVAM_API_KEY
├── .gitignore
├── data/
│   ├── sources.csv            ← input: clip tracker (one row per source span)
│   ├── manifest.csv           ← generated: chunked, selected, transcribed
│   ├── clips/                 ← generated: WAVs (gitignored)
│   └── raw/                   ← generated: temp downloads (gitignored)
└── src/
    ├── config.py              ← env loading, paths, shared logging
    ├── load_manifest.py       ← build + chunk + select + carry-forward
    ├── download.py            ← yt-dlp + ffmpeg trim
    ├── transcribe.py          ← Sarvam STT
    ├── web.py                 ← Flask review UI
    └── publish_hf.py          ← publish to HuggingFace Hub
```

---

## Source CSV schema (`data/sources.csv`)

The pipeline auto-detects the header row, so the spreadsheet may have
preamble rows above the column names. Column names may end in
`(mm:ss)` etc.; whitespace and casing are tolerated.

| Column | Required | Notes |
|---|---|---|
| `Clip ID` | no | Free-form; not used downstream |
| `Language` | yes | `English` or `Hindi` (mapped to `en-IN` / `hi-IN`) |
| `Emotion` | yes | Free-form tag; `surprise` is normalized to `Surprised`, others title-cased |
| `Source URL` | yes | YouTube `watch?v=`, `/shorts/`, or `watch?v=...&list=PL...` |
| `Start (mm:ss)` | yes | `0:13`, `1:00:29`, or raw seconds |
| `End (mm:ss)` | yes | Same formats |
| `Duration (sec)` | no | Ignored — recomputed from start/end |

Any rows with bad URLs, unparseable times, or `end ≤ start` are skipped
with a warning. Long spans are automatically split into back-to-back
≤30 s chunks; trailing fragments under 5 s are dropped.

---

## Setup

Requirements: macOS or Linux, Python 3.9+ (3.10+ recommended), `ffmpeg`
on PATH, and a Sarvam API key.

```bash
# 1. clone, enter the directory
git clone <this-repo>
cd <this-repo>

# 2. virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. ffmpeg
brew install ffmpeg          # macOS
# or: sudo apt-get install ffmpeg

# 4. Sarvam API key
cp .env.example .env
# then edit .env and paste your SARVAM_API_KEY=sk_...
```

---

## Running the pipeline

```bash
# Step 1 — build the manifest (chunking + budget selection)
python src/load_manifest.py

# Step 2 — download audio (idempotent; skips clips already present)
python src/download.py

# Step 3 — transcribe via Sarvam (idempotent; skips already-transcribed)
python src/transcribe.py
```

Common flags:

- `LOG_LEVEL=DEBUG python src/download.py` — print yt-dlp internals,
  per-clip ffmpeg commands, request IDs, etc.
- `python src/download.py --limit 5` — try only the first 5 selected
  clips (good for smoke-testing source changes).
- `python src/transcribe.py --limit 5` — same idea for STT.

Outputs:

- `data/manifest.csv` — full schema; one row per chunk.
- `data/clips/*.wav` — mono 16 kHz, exactly `start..end`.

---

## Reviewing & editing transcripts

```bash
python src/web.py
# then open http://127.0.0.1:5050
```

Per clip the UI shows: audio player, editable transcript textarea, save
button, "reviewed" checkbox. Filters: language, emotion, reviewed/
unreviewed, full-text search. Saves write back to `data/manifest.csv`
under a serialised file lock. `⌘S` / `Ctrl+S` saves the focused clip.

The `reviewed` column is added on first save and persisted thereafter.

---

## Publishing to HuggingFace

```bash
# one-time
huggingface-cli login

# build + push (dataset is public by default)
python src/publish_hf.py --repo <your-username>/sarvam-tts-mini
```

Flags:

- `--private` — create as private instead of public.
- `--no-push` — only build the in-memory dataset (sanity check rows
  + audio paths) without uploading.
- `--commit-message "..."` — override the default commit message.

The script auto-uploads a dataset card with schema, stats, and license
notes after the data push completes.

---

## Design choices worth knowing

**Chunking at 30 s.** Sarvam's sync STT cap is 30 s. Anything longer in
the source CSV is split into back-to-back 30 s segments
(`clip_id = {video_id}_{start}_{end}`). Trailing remainders shorter than
5 s are dropped — they're usually mid-word and not useful for TTS.

**Budget selection.** Targets are 30 min per language across 12 emotions
(~150 s per cell). The selector picks earliest chunks first for stability,
then top-up round-robin if a language is under target (e.g. English
`Surprised` only has ~30 s of source available — leftover budget is
redistributed among other emotions in that language).

**ffmpeg trim, not yt-dlp section cut.** yt-dlp's `download_sections`
silently degraded when YouTube's web-client formats were SABR-gated,
producing full-length files instead of trimmed slices. We now download
`bestaudio` end-to-end and trim with a direct ffmpeg call
(`-ss start -to end -ac 1 -ar 16000`). Slower per clip but correct.

**Resumability.** Every script is restartable:

- `load_manifest.py` carries forward existing `transcription` /
  `transcribed` values when re-run after a tracker edit.
- `download.py` skips any `data/clips/{id}.wav` that already exists with
  a non-zero size.
- `transcribe.py` skips any row where `transcribed == "Y"` and
  per-row-writes `manifest.csv` so a crash mid-run loses at most one
  row's worth of work.

**Logging.** Each script uses a shared `setup_logging` helper with
timestamps and per-record levels. Set `LOG_LEVEL=DEBUG` for verbose
output. yt-dlp's own messages are routed through the same logger so
warnings and 403s land in your log file when redirected.

**Idempotent re-runs after source edits.** Swap a URL or extend a
window in `data/sources.csv` and re-run `load_manifest.py` → the merge
step preserves transcripts for clips whose `clip_id` didn't change;
only the new chunks get downloaded and transcribed.

---

## Known limitations

- **PO-token / SABR warnings.** YouTube fights yt-dlp regularly; the
  pipeline retries across `android`, `ios`, `tv`, and `web` player
  clients and falls back to cookies from Chrome for age-gated content.
  The verbose warnings are normal — what matters is the final
  `OK clip ...` line per clip.
- **Single speaker by curation, not by model.** A separate diarization
  model was not used. Each source URL/timestamp was hand-picked to
  contain one speaker, and every clip was listened to during the manual
  review passes; any clip that drifted into multi-speaker territory was
  re-cut or replaced. `speaker_id` is set to the YouTube `video_id`.
- **Transcripts: ASR + iterative manual review.** Sarvam `saaras:v3` is
  the starting point; transcripts were then corrected through several
  passes in the web UI before publishing.
- **>30 s clips.** This pipeline targets the sync ASR endpoint. For
  uncut long-form transcription, switch to Sarvam's batch API
  (out of scope here).

---

## License

Code in this repository: MIT.
Audio clips: derived from public YouTube content; rights remain with the
original creators. Transcriptions and metadata are released under
CC-BY-4.0. The HuggingFace dataset card reiterates this.

---

## Acknowledgements

- **Sarvam AI** — `saaras:v3` STT model and Python SDK
- **yt-dlp** — robust YouTube fetching across format and auth quirks
- **ffmpeg** — the audio plumbing
- All the YouTube creators whose voices made it into the corpus
