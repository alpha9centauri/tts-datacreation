import argparse
import re
import sys
import time
import traceback

import pandas as pd
from sarvamai import SarvamAI

from config import (
    MANIFEST_CSV,
    CLIPS_DIR,
    SARVAM_MAX_SYNC_SECONDS,
    require_sarvam_key,
    setup_logging,
)

log = setup_logging("transcribe")


MAX_RETRIES = 5
BASE_BACKOFF = 2.0


NUM_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten",
}

SYMBOL_WORDS = {
    "%": " percent ",
    "&": " and ",
    "@": " at ",
    "+": " plus ",
    "=": " equals ",
}


def normalize_spoken(text: str) -> str:
    if not text:
        return text
    out = text
    for sym, word in SYMBOL_WORDS.items():
        out = out.replace(sym, word)

    def _num(m):
        n = m.group(0)
        return NUM_WORDS.get(n, n)

    out = re.sub(r"\b\d+\b", _num, out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def transcribe_one(client: SarvamAI, audio_path, language_code: str, clip_id: str) -> str:
    last_err = None
    size = audio_path.stat().st_size
    log.debug("%s: opening %s (%d bytes) lang=%s",
              clip_id, audio_path, size, language_code)
    for attempt in range(MAX_RETRIES):
        t0 = time.time()
        try:
            with open(audio_path, "rb") as f:
                resp = client.speech_to_text.transcribe(
                    file=f,
                    model="saaras:v3",
                    language_code=language_code,
                )
            dt = time.time() - t0
            text = getattr(resp, "transcript", None)
            if text is None and isinstance(resp, dict):
                text = resp.get("transcript", "")
            req_id = getattr(resp, "request_id", None)
            log.debug("%s: API ok in %.2fs (request_id=%s, %d chars)",
                      clip_id, dt, req_id, len(text or ""))
            return text or ""
        except Exception as e:
            dt = time.time() - t0
            last_err = e
            msg = str(e).lower()
            is_rate = "429" in msg or "rate" in msg or "too many" in msg
            if attempt == MAX_RETRIES - 1:
                log.error("%s: giving up after %d attempts (last %.2fs): %s",
                          clip_id, MAX_RETRIES, dt, e)
                break
            sleep_for = BASE_BACKOFF * (2 ** attempt)
            if is_rate:
                sleep_for *= 2
            log.warning("%s: attempt %d/%d failed in %.2fs (%s%s); sleeping %.1fs",
                        clip_id, attempt + 1, MAX_RETRIES, dt,
                        "rate-limit, " if is_rate else "", e, sleep_for)
            time.sleep(sleep_for)
    raise RuntimeError(f"transcription failed after {MAX_RETRIES} attempts: {last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process first N clips")
    args = parser.parse_args()

    log.info("manifest: %s", MANIFEST_CSV)
    log.info("clips dir: %s", CLIPS_DIR)
    if args.limit is not None:
        log.info("--limit %d", args.limit)

    if not MANIFEST_CSV.exists():
        log.error("%s not found. Run load_manifest.py first.", MANIFEST_CSV)
        sys.exit(1)

    api_key = require_sarvam_key()
    log.info("Sarvam API key loaded (...%s)", api_key[-4:])
    client = SarvamAI(api_subscription_key=api_key)
    log.info("Sarvam client initialised, model=saaras:v3, mode=transcribe")

    df = pd.read_csv(MANIFEST_CSV)
    log.info("loaded manifest: %d rows", len(df))
    if "transcription" not in df.columns:
        df["transcription"] = ""
    if "transcribed" not in df.columns:
        df["transcribed"] = ""
    # Force string dtype so we don't get FutureWarnings when writing into
    # columns that pandas inferred as float64 (all-NaN -> float).
    df["transcription"] = df["transcription"].astype("object").fillna("")
    df["transcribed"] = df["transcribed"].astype("object").fillna("")

    if "selected" in df.columns:
        sel_count = (df["selected"].astype(str).str.upper() == "Y").sum()
        log.info("%d rows marked selected=Y", sel_count)

    processed = 0
    skipped = 0
    failed = 0
    over_limit = 0
    not_selected = 0
    missing_audio = 0

    total_targets = (
        (df["selected"].astype(str).str.upper() == "Y").sum()
        if "selected" in df.columns else len(df)
    )

    started = time.time()
    progress = 0

    for idx, row in df.iterrows():
        if args.limit is not None and processed >= args.limit:
            log.info("hit --limit %d, stopping", args.limit)
            break

        if "selected" in df.columns and str(row.get("selected", "")).strip().upper() != "Y":
            not_selected += 1
            continue

        clip_id = row["clip_id"]
        progress += 1

        if str(row.get("transcribed", "")).strip().upper() == "Y":
            log.info("[%d/%d] SKIP  %s (already transcribed)",
                     progress, total_targets, clip_id)
            skipped += 1
            continue

        duration = int(row["duration_sec"])
        if duration > SARVAM_MAX_SYNC_SECONDS:
            log.warning("[%d/%d] SKIP  %s: %ss > %ss sync limit",
                        progress, total_targets, clip_id,
                        duration, SARVAM_MAX_SYNC_SECONDS)
            over_limit += 1
            continue

        audio_path = CLIPS_DIR / f"{clip_id}.wav"
        if not audio_path.exists():
            log.error("[%d/%d] MISS  %s: audio missing at %s",
                      progress, total_targets, clip_id, audio_path)
            missing_audio += 1
            failed += 1
            continue

        log.info("[%d/%d] STT   %s  lang=%s  %ss  %d bytes",
                 progress, total_targets, clip_id, row["language"],
                 duration, audio_path.stat().st_size)
        t0 = time.time()
        try:
            text = transcribe_one(client, audio_path, row["language"], clip_id)
            text = normalize_spoken(text)
            df.at[idx, "transcription"] = text
            df.at[idx, "transcribed"] = "Y"
            df.to_csv(MANIFEST_CSV, index=False)
            processed += 1
            dt = time.time() - t0
            preview = (text[:80] + "…") if len(text) > 80 else text
            log.info("       OK  %s in %.2fs (%d chars) :: %s",
                     clip_id, dt, len(text), preview)
        except Exception as e:
            failed += 1
            dt = time.time() - t0
            log.error("       FAIL %s after %.2fs: %s", clip_id, dt, e)
            log.debug("traceback:\n%s", traceback.format_exc())

    elapsed = time.time() - started
    log.info(
        "Summary: transcribed=%d already_done=%d over_limit=%d "
        "missing_audio=%d failed=%d not_selected=%d  (%.1fs)",
        processed, skipped, over_limit, missing_audio, failed, not_selected, elapsed,
    )


if __name__ == "__main__":
    main()
