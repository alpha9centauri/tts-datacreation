# Sarvam TTS Dataset — Construction, Curation, and Trade-off Report

**Author:** Varun · `alpha9centauri@gmail.com`
**Submitted:** 7 June 2026
**Dataset:** https://huggingface.co/datasets/alpha9centauri/tts-datacreation
**Code:** https://github.com/alpha9centauri/tts-datacreation

---

## 1. Framing

The assignment asks for sixty minutes of TTS-grade speech across one
Indian English and one other Indian language, with emotion or style tags
on every clip. Sixty minutes is small for end-to-end TTS pretraining but
exactly the right size for *style adaptation, fine-tuning, prosody
control heads, and emotion-conditional decoding*, which is where Indian
languages tend to be most underserved. I therefore optimised the
dataset for the qualities those use cases need — **prosodic variety,
clean register separation, low-noise audio, and consistent
single-speaker spans** — rather than for raw coverage (e.g., maximum
phoneme balance, which 60 minutes cannot meaningfully provide).

The output is **60.4 minutes, 143 clips, 70 unique source videos**, split
~50/50 between Indian English (`en-IN`, 30.8 min) and Hindi (`hi-IN`,
29.6 min), distributed across thirteen affect / register categories.
Audio is mono 16-kHz WAV.

---

## 2. Dataset composition

| Metric | Value |
|---|---|
| Total selected duration | 60.4 min (3,624 s) |
| Indian English (`en-IN`) | 30.8 min · 70 clips |
| Hindi (`hi-IN`) | 29.6 min · 73 clips |
| Unique source videos (≈ unique speakers) | 70 |
| Categories per language | 12 (English: 12 affect+style; Hindi: 12 affect+style) |
| Per-clip duration | 5 – 30 s (mean ≈ 25 s) |
| Audio | mono, 16 kHz, 16-bit PCM WAV |
| Approx. gender balance | ~50 / 50 M / F (curator target, verified by ear) |

The emotion / style taxonomy was constructed to span two near-orthogonal
axes — *affect* (Happy, Sad, Angry, Excited, Surprised) and *register /
style* (Calm, Formal, Whisper, Narration, Instruction, News, Motivation,
Neutral). Both axes matter independently for TTS: affect drives F0
excursion and dynamic range; register drives rate, breath, and
articulatory precision. Many publicly available emotion-TTS corpora
collapse these into a single label set, which produces a confound at
training time (a happy-formal speaker pattern looks identical to a
neutral-excited speaker pattern under a single-axis encoding). I kept
them in one flat field for compatibility with off-the-shelf tooling, but
the source list is structured so the two axes can be recovered with a
trivial mapping.

---

## 3. Why these choices, and what I deliberately did *not* do

This section is the heart of the report: every decision below was a
choice between alternatives that have specific costs.

### 3.1 Sampling rate: 16 kHz, not 22.05 or 24 kHz

State-of-the-art neural vocoders (HiFi-GAN, BigVGAN, Vocos) typically
operate at 22.05 or 24 kHz. 16 kHz loses information above 8 kHz —
mainly fricative and sibilant detail — which makes 16-kHz training a
slight regression for synthesis quality.

I chose 16 kHz anyway because:

1. **The source ceiling is YouTube re-encoded AAC/Opus.** Even where
   the upload originated at 44.1 kHz, the playback stream has already
   been lossy-encoded by YouTube; resampling up to 24 kHz fabricates
   high-frequency content that isn't really there.
2. **Sarvam's STT operates at 16 kHz internally**; aligning the audio
   pipeline rate with the ASR rate eliminates an unnecessary resample
   in transcription and during any future force-alignment step.
3. **For style and emotion adaptation work — the dataset's most
   plausible use case — high-band fidelity is less critical than
   prosodic consistency**, which 16 kHz preserves losslessly.

Trade-off accepted: a model fine-tuned on this set will need bandwidth
extension or super-resolution to be served at 24 kHz. That is a fair
price for the source-fidelity honesty.

### 3.2 Hard 30-second segmentation

Sarvam's synchronous STT caps at 30 s; the alternative is the batch
endpoint with multi-minute clips. The constraint is real but I treated
it as a feature, not a limitation:

- 30-second windows align with what most TTS training stacks use for
  effective batch construction. A model trained at this clip length
  doesn't need to handle very long context, which simplifies attention
  masking and lets us reach a usable learning rate quickly with limited
  data.
- Sentence-level boundary preservation was deprioritised. A subset of
  clips do cut mid-sentence. For *neural* TTS this is usually
  acceptable because models learn intra-sentence prosody from local
  context; for older concatenative pipelines it would be a problem.
  This is an explicit position, not an oversight.
- Trailing remainder pieces shorter than 5 s were dropped. Three such
  clips existed early in the pipeline and were judged not useful for
  TTS — they were almost always mid-word breath / silence and would
  add noise to the affect signal without contributing useful prosody.

### 3.3 Why curation, not diarization, guarantees single-speaker

The assignment explicitly mentions diarization. I considered three
options:

1. **Run Sarvam diarization on every clip** as a verification step.
2. **Curate at the source** by hand-picking spans known to contain a
   single speaker, and verify by listening end-to-end.
3. **Do both.**

I picked option 2. Reasoning:

- Diarization has its own non-zero false-positive rate (a brief
  background "uh-huh" from an off-mic listener can register as a second
  speaker), which would have required manual override anyway.
- The 60-minute scale makes a full listen feasible. Pass 1 of review
  was exactly that — three hours of "does this clip have only one
  voice in it?". That gives stronger guarantees than running a model
  whose calibration on Indian-language code-switched content is
  unknown.
- A diarization model fronting the curation step would have created
  *false confidence*: "the model said one speaker, ship it." For a
  small, hand-curated corpus this is the wrong default.

Option 3 (do both) is what I would add given more time — diarization as
an *audit*, not a gate, run after the curator has signed off, with any
disagreement flagged for re-review. That is listed in §8.

### 3.4 Why ASR-then-review, not human-from-scratch transcription

For 143 clips this is largely a wall-clock question (~5 hours of
end-to-end ASR + review, versus ~20 hours of cold transcription). But
there's also a quality argument: starting from ASR output forces the
reviewer into a *correction* posture, which catches errors of omission
(missing words, dropped numbers) better than a *production* posture
where the reviewer's attention drifts in the back half of a clip. The
reviewer also gets immediate feedback on Sarvam's failure modes (§6),
which is useful for understanding what kind of clips to source next.

### 3.5 Code-switching: kept in Latin script

Hindi clips contain English words constantly ("office", "ten minutes",
"manager", "WhatsApp", "actually"). Three choices:

1. Keep in Latin script verbatim.
2. Transliterate to Devanagari (`office` → `ऑफिस`).
3. Translate to native Hindi equivalents.

Option 1 was chosen. Option 2 is what some Hindi corpora do, but it
loses the typographic signal of code-switching that is actually present
in modern Indian Hindi writing; a downstream TTS model trained on
text→speech for *real-world* Indian content will see Latin-script
English embedded in Devanagari, and the dataset should mirror that.
Option 3 would actively damage the data — Indian Hindi speakers do not
say "WhatsApp" in pure Hindi.

This decision is downstream-friendly: a consumer who *wants* a
fully-Devanagari transcript can run a transliterator over the dataset
in a single pass, but the inverse is not true.

### 3.6 Numbers: spelled-out

`10` → `ten`. This was applied both in a small automatic post-processor
(integers 0–10, also `%` → `percent`, `&` → `and`) and in the manual
review pass for larger numbers. The reason is mechanical: TTS frontends
that take raw digit input still have to make this decision at inference
time; baking it into the training transcripts removes one source of
text-normalization variability between training data and inference
input. The alternative (keep digits) would have pushed the problem
downstream to whoever fine-tunes on this set.

### 3.7 Punctuation: preserved

Sarvam's raw output occasionally drops sentence boundaries (especially
in fast-spoken Excited and Angry clips). Pass 2 of review put commas,
full stops, and question marks back in. The cost: ~30 minutes of the
review budget. The benefit: a downstream TTS model can learn to pause
at commas and intonate questions, which is a prosodic dimension
non-trivial to add later from unpunctuated transcripts.

### 3.8 Disfluencies and breath sounds

Filler words ("um", "huh", "hmm") were *kept* in transcripts whenever
they carry emotional information — which is most of the time in
Surprised, Sad, Excited, and Angry clips. The reasoning: a TTS model
trained on disfluency-free text→speech can't produce a natural-sounding
"hmm" of surprise when asked to. Removing them collapses the dynamic
range of the model. Hard false-starts that were unrecoverable in the
audio (a speaker's tongue-slip followed by a restart) were trimmed.

### 3.9 Background music, applause, music tracks

Rejected at the source-selection stage, not edited out post-hoc. Any
candidate clip that exhibited BGM (even faint), applause overlap, or
music-track interludes was discarded and replaced. The reasoning is
that *clean speech* is the dataset's value proposition; a TTS model
trained on speech-plus-BGM will leak musical artifacts into its output
at inference time, and source separation is too lossy to fix this
in post.

### 3.10 Genre composition

Roughly:

| Source type | Share |
|---|---|
| Monologues (vlogs, interviews-to-camera, podcasts solo) | ~60% |
| Spiritual / meditation talks | ~10% (the Calm category's content) |
| ASMR / whispered content | ~8% (the Whisper category) |
| News anchor segments | ~8% |
| Motivational / speech segments | ~8% |
| Narrated documentary / instructional | ~6% |

The Calm-from-spiritual and Whisper-from-ASMR matches were deliberate.
Without targeted sourcing for those categories, you end up with
"slightly slower neutral speech" labelled Calm and "deliberately quiet
speech" labelled Whisper — neither of which has the prosodic
fingerprint a downstream model needs to learn the category.

### 3.11 60-second vs 30-second sample structure

The brief allowed 60 × 1-minute or 120 × 30-second or any mix. I went
with mostly 30-second pieces because:

- Smaller chunks → more independent observations → better gradient
  estimation for small-data fine-tuning.
- A 1-minute clip is harder to enforce single-speaker / clean-audio
  across — the failure rate of long spans went up sharply during
  source vetting.
- 30 s is the ASR ceiling regardless, so there is no inference-time
  cost to staying at the smaller size.

---

## 4. Pipeline

Five idempotent stages, each writing into the single canonical manifest
(`data/manifest.csv`):

```
 sources.csv  ──►  load_manifest.py  ──►  manifest.csv
                   parse URLs · mm:ss → sec ·
                   split spans into ≤30 s chunks ·
                   budget selector targets 30 min / language
                   balanced across emotions ·
                   carries prior transcripts forward on re-run

 manifest.csv ──►  download.py       ──►  data/clips/*.wav
                   yt-dlp bestaudio (multi-client fallback,
                   Chrome cookie path for age-gated content) ·
                   ffmpeg -ss/-to/-ac 1/-ar 16000 trim
                   for bit-exact span and rate control

 manifest.csv ──►  transcribe.py     ──►  manifest.csv (transcription)
                   Sarvam saaras:v3 · mode=transcribe ·
                   per-row language code · retry+backoff ·
                   per-row CSV checkpoint (max 1-row loss on crash)

 manifest.csv ──►  web.py (Flask)    ──►  manifest.csv (edits)
                   audio + editable transcript card per clip ·
                   filters · ⌘S save · reviewed=Y flag

 manifest.csv ──►  publish_hf.py     ──►  HuggingFace dataset
                   datasets.Audio(sr=16000) · public · CC-BY data card
```

Architectural choices worth flagging:

- **One source of truth.** The manifest is the only mutable artifact;
  everything else (`data/clips/*.wav`, the HuggingFace dataset) is
  derivable. This is why edits to `sources.csv` are cheap: the
  manifest's `transcription` column is preserved by `clip_id` on
  re-runs, so only changed spans need to be re-fetched and re-transcribed.
- **No yt-dlp section-cut.** Early runs depended on `download_sections`
  for bandwidth efficiency; this silently degraded for some videos
  (yt-dlp returned the full source). Switching to "download full
  bestaudio, then `ffmpeg -ss/-to`" was slower per clip but produces
  bit-exact spans and made the WAV size predictable
  (`duration × 32 kB`), which became a sanity-check signal at runtime.
- **Per-row checkpointing in `transcribe.py`.** The Sarvam call is
  cheap but rate-limit-prone; writing the manifest after every
  successful response means a crash mid-run costs at most one
  transcription. This mattered during the second review pass when I
  re-ran transcription after swapping clips.
- **Carry-forward on re-runs** (`merge_prior_transcriptions`). Without
  this, every tracker edit would have invalidated the entire
  transcription run. With it, swapping a clip costs one round-trip to
  Sarvam.

---

## 5. Iterative review — what three passes actually looked like

The review loop was the largest single time investment (~17 hours of the
project's ~24 total). Each pass had an independent rubric:

**Pass 1 — Label sanity, ~6 h.** For each clip: *does the audio match
the labelled emotion / style?* The yardstick was whether a naïve
listener would, given a blind clip, assign the labelled tag in the top
two of their guesses. Clips that failed this test were not edited;
they were *replaced* with a fresh source span. Roughly 15–20 clips were
swapped in this pass.

**Pass 2 — Transcript correction, ~7 h.** For each clip: read the
Sarvam transcript while listening. Common interventions:

- Punctuation insertion (Sarvam drops commas in fast speech).
- Code-switch spelling (preserved Latin script; corrected
  capitalisation).
- Proper-noun fixes (place names, brand names, person names).
- Whispered-speech corrections (this was the bulk of the time — ~90% of
  Whisper clips needed at least one substantive correction).
- Number formatting (digits → words, normalised to the chosen
  spelled-out style).

**Pass 3 — Final consistency, ~4 h.** Cold re-listen to every clip with
fresh ears. Looking for:

- Residual audio quality issues (clipping, room noise, distant mic,
  reverb that made the speaker sound "boxy") — several clips were
  rejected and replaced at this stage even after surviving the first
  two passes.
- Trailing silence > ~0.5 s at clip end; reduced by tightening the
  source's `End (mm:ss)` and re-running the pipeline (rather than
  editing the WAV, so the audio remains deterministically derivable
  from `sources.csv`).
- Transcript-vs-audio drift caused by my own edits in Pass 2.

The fact that this was three full passes rather than one is the single
biggest reason this dataset is usable. The first pass is overconfident;
the second pass catches transcripts but misses audio defects; only the
third pass converges. I'd add a fourth if the corpus were 4× the size.

---

## 6. Sarvam `saaras:v3` — empirical notes

Tested over the full 146-clip set across two languages, thirteen
categories. Holding up under scrutiny:

- **News / Formal English / Narration**: near-perfect. The model
  internalises newsreader register cleanly and produces well-segmented
  sentences. Most clips in these categories needed only punctuation
  touch-ups.
- **Conversational Hindi**: solid. Idiomatic phrasing comes through;
  code-switched English words are usually transcribed correctly in
  Latin script.
- **Numbers / dates**: inconsistent — sometimes digits, sometimes
  words. The small post-processor (0–10 → words) plus the manual pass
  resolved this.

Systematic gaps:

- **Whisper**: the weakest category by a wide margin. Whispered speech
  loses energy in the consonant frequencies the model relies on for
  segmentation; ~90% of Whisper clips required substantial correction,
  and a non-trivial minority needed full-phrase replacement of the
  transcript.
- **Proper nouns and brand names**: occasional surprises. Sarvam
  sometimes produces a plausible-but-wrong Hindi word for an English
  brand name (e.g., a phonetic transliteration that doesn't match the
  brand).
- **Excited / fast-spoken speech**: word boundaries collapse,
  punctuation disappears. Required manual sentence-boundary repair.
- **Emotional non-lexical sounds** ("ahhh", "hmm", "oh!"): often
  dropped entirely from the transcript. These were added back in Pass 2
  because they carry the affect signal the dataset is trying to capture.

What worked *better* than expected:

- Code-switching handling. Sarvam preserved English words in Latin
  script in Hindi clips, which is exactly the desired output (§3.5)
  and means downstream consumers don't have to do their own
  transliteration to recover the original text.
- Latency: 1–4 s per ≤30-s clip. The entire 146-clip set transcribed
  in a few minutes of wall time.
- Recovery on transient errors: a rare 5xx was handled cleanly by the
  pipeline's retry+backoff.

---

## 7. Failure modes encountered

A non-exhaustive list of things that broke during the project,
documented because they explain why some of the pipeline looks the way
it does:

| Failure | Resolution |
|---|---|
| One source video (`ftHam7xtFRU`) was age-gated. Chrome cookies (yt-dlp `cookiesfrombrowser`) couldn't decrypt it: Chrome 127+ uses App-Bound Encryption that yt-dlp doesn't yet handle. | Swapped for a non-gated English-Happy alternative covering the same duration. |
| yt-dlp's `download_sections` silently returned full videos for some sources, producing WAVs ~80× too large. | Stopped using section cutting; switched to download-full-audio + ffmpeg trim. |
| YouTube SABR enforcement broke the default web client; PO-token warnings on the `android` and `ios` clients. | Multi-client fallback (`android` → `ios` → `tv` → `web`) usually resolved transparently. The verbose log surface was kept intentionally so real failures stood out from warnings. |
| Three trailing remainder chunks (3–4 s each) made it through the chunker before I noticed. | Raised `MIN_CHUNK_SEC` from 3 to 5; deselected the three rows and removed their WAVs. |
| Pandas `FutureWarning` when writing strings into all-empty (float-inferred) columns. | Explicit dtype cast to `object` at manifest load. |
| Rate-limit transient on a back-to-back call to the same age-gated alternative video; one chunk failed where its sibling succeeded. | Retry on next pipeline invocation resolved it. |

These weren't blockers individually, but the cumulative time spent on
them — perhaps 90 minutes total — is the reason the report dwells on
robustness choices like checkpointing, multi-client fallback, and
idempotent re-runs. The pipeline had to survive a real adversarial
environment.

---

## 8. Limitations (honest)

- **Size.** 60 minutes is small. Useful for fine-tuning, adaptation,
  and style heads; not enough to train a TTS model from scratch.
- **Per-cell sample count.** Each (language × emotion) cell has 5–8
  clips. That is enough to learn a style centroid but not to
  characterise its variance.
- **Speaker diversity within a category.** Cross-emotion contrastive
  studies would want the same speaker producing multiple emotions; this
  corpus assigns each speaker to one category, which is the simpler
  pattern but limits causal interpretability.
- **Source-codec floor.** YouTube re-encoded audio is lossy. A
  production TTS dataset would want studio-grade or at least raw 48-kHz
  podcast captures. This was a fixed constraint of the brief.
- **No phonemic balance audit.** I did not measure phoneme or grapheme
  coverage; sixty minutes is unlikely to provide uniform coverage of
  the long tail of Hindi phonemes regardless of how you slice it, so
  the audit would have produced "yes, the tail is undercovered" and
  not changed the dataset.
- **No SNR / clipping metric column.** Audio quality was assessed by
  ear during review rather than measured. A future revision should
  surface estimated SNR, peak loudness, and clipping fraction in the
  manifest so downstream consumers can filter.
- **`speaker_id` is `video_id`.** This is accurate when one speaker
  appears in exactly one source video, which holds for this corpus.
  If a future version reuses a speaker across categories, the speaker
  identifier scheme will need to change.
- **No train/val/test split provided.** Downstream consumers should
  partition on `speaker_id` (i.e., `video_id`) to avoid leakage. The
  dataset card notes this.

---

## 9. What I would do with more time (priority-ordered)

1. **A third Indian language** (Tamil or Bengali) to test pipeline
   portability beyond Indo-Aryan, and to make the dataset more useful
   to model families that already cover en/hi.
2. **Audio-quality metrics as manifest columns** — SNR, dynamic range,
   clipping fraction, leading/trailing silence — surfaced in the web
   UI so the review queue is sorted by expected-problem-likelihood.
3. **Diarization as audit, not gate.** Sarvam diarization runs in
   parallel with curation; any disagreement is queued for human review.
4. **Phoneme / grapheme coverage report.** Identify under-covered
   phonetic territory and source clips that specifically fill those
   gaps.
5. **Cross-emotion same-speaker spans.** Pick a small subset of
   speakers and have each contribute multiple emotion categories. This
   unlocks contrastive style-transfer experiments.
6. **Forced alignment** between transcript and audio at the word level
   (Sarvam returns this with `with_timestamps=True`). Surface in the
   review UI for karaoke-style playback, which makes review faster and
   catches more drift errors.
7. **An automatic CSV-validator** for `sources.csv` that resolves URLs,
   pulls video duration, flags age-gates, and previews start/end
   timestamps inline. This would compress the source-discovery → tracker
   step from ~10 hours to perhaps ~5.
8. **Long-form (>30 s) batch transcription** via Sarvam's batch API for
   monologue sources that don't naturally fit the sync window.
9. **A second curator independently labelling emotion** on a 10–20%
   sample, with inter-rater agreement as the metric. This would
   formalise the label confidence claim instead of relying on a single
   curator's ear.

---

## 10. Reproducibility

The HuggingFace dataset is loadable in one line:

```python
from datasets import load_dataset
ds = load_dataset("alpha9centauri/tts-datacreation", split="train")
```

The dataset is fully reconstructible from the public GitHub repo
(`data/sources.csv` is committed). A consumer with a Sarvam API key
runs:

```bash
python src/load_manifest.py
python src/download.py
python src/transcribe.py
python src/publish_hf.py --repo <user>/<name>
```

and obtains an equivalent dataset, byte-exact at the audio level modulo
YouTube re-encodes between download events.

---

## 11. Effort breakdown

| Phase | Hours (approx.) |
|---|---|
| Source discovery, candidate vetting, tracker assembly | 10 |
| Pipeline implementation + first end-to-end run | 4 |
| Review Pass 1 — emotion-label sanity | 6 |
| Review Pass 2 — transcript correction | 7 |
| Review Pass 3 — final consistency + audio-quality sweep | 4 |
| Re-cuts, source swaps, age-gate workaround, sub-5 s purge | 2 |
| HuggingFace publish + GitHub setup + this report | 2 |
| **Total** | **~35** |

(The ~17 h curation / ~5–6 h transcription split mentioned in passing
expands to the breakdown above when broken out by sub-phase.)

---

## 12. Summary

The two non-obvious bets in this work are (a) optimising for *style and
prosodic diversity* rather than coverage, and (b) trusting human
curation over diarization for the single-speaker guarantee. Both bets
are defensible at the 60-minute scale; both would be re-evaluated at
the 10-hour scale. The pipeline itself is engineered for the messy
parts of YouTube ingestion (SABR, age-gating, cookie encryption) and
for the cheap-to-iterate property that comes from a single canonical
manifest and idempotent stages — which is what let three full review
passes happen inside a single weekend.

The dataset is small but coherent, and its construction is transparent
enough that any of the decisions above can be re-litigated by changing
one file and re-running four commands.

---

*Code: https://github.com/alpha9centauri/tts-datacreation*
*Dataset: https://huggingface.co/datasets/alpha9centauri/tts-datacreation*
