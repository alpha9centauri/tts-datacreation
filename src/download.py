import argparse
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
from yt_dlp import YoutubeDL

from config import MANIFEST_CSV, CLIPS_DIR, setup_logging

log = setup_logging("download")


class _YdlLogger:
    """Bridge yt-dlp's messages into our logger."""
    def debug(self, msg):
        if msg.startswith("[debug] "):
            log.debug("yt-dlp: %s", msg[8:])
        else:
            log.debug("yt-dlp: %s", msg)

    def info(self, msg):
        log.debug("yt-dlp: %s", msg)

    def warning(self, msg):
        log.warning("yt-dlp: %s", msg)

    def error(self, msg):
        log.error("yt-dlp: %s", msg)


def ydl_opts_for(out_template: str) -> dict:
    """Download the full bestaudio stream — no in-yt-dlp trimming, no audio
    postprocessor. We do the trim + resample ourselves with ffmpeg so we can
    guarantee the output is exactly start..end at mono 16 kHz."""
    return {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "logger": _YdlLogger(),
        # YouTube's web client is SABR-gated; android/ios/tv player clients still
        # serve plain progressive/HLS streams.
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "tv", "web"],
            },
        },
        "retries": 5,
        "fragment_retries": 5,
        # Use the signed-in Chrome session so age-gated videos can be fetched.
        "cookiesfrombrowser": ("chrome",),
    }


def _ffmpeg_trim(src: Path, dst: Path, start_sec: int, end_sec: int, clip_id: str) -> bool:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", str(src),
        "-ac", "1", "-ar", "16000",
        "-vn",
        str(dst),
    ]
    log.debug("%s: ffmpeg: %s", clip_id, " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("%s: ffmpeg failed (rc=%d): %s", clip_id, r.returncode, r.stderr.strip())
        return False
    return True


def download_clip(url: str, start_sec: int, end_sec: int, clip_id: str) -> bool:
    out_path = CLIPS_DIR / f"{clip_id}.wav"
    if out_path.exists() and out_path.stat().st_size > 0:
        log.debug("%s: already present (%d bytes)", clip_id, out_path.stat().st_size)
        return True

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: yt-dlp -> full bestaudio in original container, named by clip_id
    src_template = str(CLIPS_DIR / f"{clip_id}.src.%(ext)s")
    opts = ydl_opts_for(src_template)
    log.debug("%s: yt-dlp fetch full audio from %s", clip_id, url)
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        src_path = Path(ydl.prepare_filename(info))

    if not src_path.exists():
        # Some PPs (or container choice) may have changed extension; glob.
        candidates = list(CLIPS_DIR.glob(f"{clip_id}.src.*"))
        if not candidates:
            log.error("%s: yt-dlp produced no file matching %s.src.*", clip_id, clip_id)
            return False
        src_path = candidates[0]
        log.debug("%s: resolved downloaded file to %s", clip_id, src_path)

    src_size = src_path.stat().st_size
    log.debug("%s: source audio %d bytes -> trimming %ss..%ss",
              clip_id, src_size, start_sec, end_sec)

    # Step 2: ffmpeg trim + convert to mono 16 kHz WAV
    ok = _ffmpeg_trim(src_path, out_path, start_sec, end_sec, clip_id)

    # Step 3: clean up the intermediate
    try:
        src_path.unlink()
    except Exception as e:
        log.debug("%s: could not unlink %s: %s", clip_id, src_path, e)

    if not ok:
        return False
    if not out_path.exists() or out_path.stat().st_size == 0:
        log.error("%s: output missing/empty at %s", clip_id, out_path)
        return False
    log.debug("%s: wrote %d bytes -> %s", clip_id, out_path.stat().st_size, out_path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only download first N selected clips")
    args = parser.parse_args()

    log.info("manifest: %s", MANIFEST_CSV)
    log.info("clips dir: %s", CLIPS_DIR)
    if args.limit is not None:
        log.info("--limit %d", args.limit)
    if shutil.which("ffmpeg") is None:
        log.error("ffmpeg not found on PATH; install it (brew install ffmpeg)")
        sys.exit(1)
    if not MANIFEST_CSV.exists():
        log.error("%s not found. Run load_manifest.py first.", MANIFEST_CSV)
        sys.exit(1)

    df = pd.read_csv(MANIFEST_CSV)
    log.info("loaded manifest: %d rows, columns=%s", len(df), list(df.columns))

    if "selected" in df.columns:
        sel_total = (df["selected"].astype(str).str.upper() == "Y").sum()
        log.info("%d rows marked selected=Y", sel_total)

    done = 0
    skipped = 0
    failed = 0
    not_selected = 0
    seen = set()
    started = time.time()

    total_targets = (
        (df["selected"].astype(str).str.upper() == "Y").sum()
        if "selected" in df.columns else len(df)
    )

    progress = 0
    attempted = 0
    for _, row in df.iterrows():
        if "selected" in df.columns and str(row.get("selected", "")).strip().upper() != "Y":
            not_selected += 1
            continue

        clip_id = row["clip_id"]
        if clip_id in seen:
            log.debug("%s: duplicate row in manifest, skipping", clip_id)
            continue
        seen.add(clip_id)
        progress += 1

        if args.limit is not None and attempted >= args.limit:
            log.info("hit --limit %d, stopping", args.limit)
            break
        attempted += 1

        out_path = CLIPS_DIR / f"{clip_id}.wav"
        if out_path.exists() and out_path.stat().st_size > 0:
            log.info("[%d/%d] SKIP  %s (already %d bytes)",
                     progress, total_targets, clip_id, out_path.stat().st_size)
            skipped += 1
            continue

        url = row["source_url"]
        s, e = int(row["start_sec"]), int(row["end_sec"])
        log.info("[%d/%d] DL    %s  %s  %ss..%ss (%ss)",
                 progress, total_targets, clip_id, url, s, e, e - s)
        t0 = time.time()
        try:
            ok = download_clip(url=url, start_sec=s, end_sec=e, clip_id=clip_id)
            dt = time.time() - t0
            if ok:
                done += 1
                log.info("       OK  %s in %.1fs (%d bytes)",
                         clip_id, dt, out_path.stat().st_size)
            else:
                failed += 1
                log.error("       FAIL %s after %.1fs (no output)", clip_id, dt)
        except Exception as e:
            failed += 1
            dt = time.time() - t0
            log.error("       FAIL %s after %.1fs: %s", clip_id, dt, e)
            log.debug("traceback:\n%s", traceback.format_exc())

    elapsed = time.time() - started
    log.info("Summary: downloaded=%d already_present=%d failed=%d not_selected=%d  (%.1fs)",
             done, skipped, failed, not_selected, elapsed)


if __name__ == "__main__":
    main()
