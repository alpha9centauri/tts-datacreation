import math
import re
import sys
from urllib.parse import urlparse, parse_qs

import pandas as pd

from config import SOURCES_CSV, MANIFEST_CSV, SARVAM_MAX_SYNC_SECONDS, setup_logging

log = setup_logging("load_manifest")


# Budget targets (seconds)
TOTAL_TARGET_SEC = 60 * 60      # 60 min total
PER_LANG_TARGET_SEC = 30 * 60   # 30 min per language

# Chunking
CHUNK_SEC = SARVAM_MAX_SYNC_SECONDS  # 30
MIN_CHUNK_SEC = 5                    # drop trailing crumbs shorter than this


EMOTION_OVERRIDES = {
    "surprise": "Surprised",
}

LANG_MAP = {
    "english": "en-IN",
    "en": "en-IN",
    "en-in": "en-IN",
    "hindi": "hi-IN",
    "hi": "hi-IN",
    "hi-in": "hi-IN",
}


def extract_video_id(url: str):
    if not isinstance(url, str) or not url.strip():
        return None
    url = url.strip()
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    host = (parsed.netloc or "").lower()
    if "youtu.be" in host:
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid or None
    if "youtube.com" not in host and "youtube-nocookie.com" not in host:
        return None
    path = parsed.path or ""
    if path.startswith("/shorts/"):
        vid = path.split("/")[2] if len(path.split("/")) > 2 else ""
        return vid or None
    if path.startswith("/embed/"):
        vid = path.split("/")[2] if len(path.split("/")) > 2 else ""
        return vid or None
    qs = parse_qs(parsed.query)
    if "v" in qs and qs["v"]:
        return qs["v"][0]
    return None


def parse_mmss(value) -> int:
    if value is None:
        raise ValueError("missing time")
    s = str(value).strip()
    if not s:
        raise ValueError("empty time")
    if re.fullmatch(r"\d+", s):
        return int(s)
    parts = s.split(":")
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + int(sec)
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + int(sec)
    raise ValueError(f"unparseable time: {value!r}")


def normalize_emotion(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    key = s.lower()
    if key in EMOTION_OVERRIDES:
        return EMOTION_OVERRIDES[key]
    return s.title()


def normalize_language(value) -> str:
    if value is None:
        return ""
    key = str(value).strip().lower()
    return LANG_MAP.get(key, "")


def _find_header_row(path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if "Clip ID" in line and "Source URL" in line:
                return i
    return 0


def _pick(row, *names):
    for n in names:
        if n in row and pd.notna(row[n]):
            return row[n]
    return None


def split_into_chunks(base_row: dict) -> list:
    """Split a span into back-to-back ≤CHUNK_SEC sub-rows."""
    start = base_row["start_sec"]
    end = base_row["end_sec"]
    chunks = []
    n = math.ceil((end - start) / CHUNK_SEC)
    for i in range(n):
        cs = start + i * CHUNK_SEC
        ce = min(start + (i + 1) * CHUNK_SEC, end)
        if ce - cs < MIN_CHUNK_SEC:
            continue
        chunk = dict(base_row)
        chunk["start_sec"] = cs
        chunk["end_sec"] = ce
        chunk["duration_sec"] = ce - cs
        chunk["chunk_index"] = i
        chunk["clip_id"] = f"{base_row['video_id']}_{cs}_{ce}"
        chunks.append(chunk)
    return chunks


def select_budget(df: pd.DataFrame) -> pd.DataFrame:
    """Mark rows with selected=Y to hit ~PER_LANG_TARGET_SEC per language,
    distributed roughly evenly across emotions in that language."""
    df = df.copy()
    df["selected"] = ""

    for lang in sorted(df["language"].unique()):
        lang_mask = df["language"] == lang
        emotions = sorted(df.loc[lang_mask, "emotion"].unique())
        if not emotions:
            continue
        per_em_target = PER_LANG_TARGET_SEC / len(emotions)

        lang_selected_sec = 0

        # Pass 1: fair per-emotion share.
        for em in emotions:
            cell_idx = df.index[lang_mask & (df["emotion"] == em)].tolist()
            # Prefer earlier chunks of earlier-starting clips for stability.
            cell_idx.sort(key=lambda i: (df.at[i, "video_id"], df.at[i, "start_sec"]))
            picked = 0.0
            for i in cell_idx:
                if picked >= per_em_target:
                    break
                df.at[i, "selected"] = "Y"
                picked += df.at[i, "duration_sec"]
                lang_selected_sec += df.at[i, "duration_sec"]

        # Pass 2: if the language is under budget, top up from any remaining
        # chunks in that language (round-robin across emotions for balance).
        remaining = PER_LANG_TARGET_SEC - lang_selected_sec
        if remaining > 0:
            spare_by_em = {
                em: [i for i in df.index[lang_mask & (df["emotion"] == em)]
                     if df.at[i, "selected"] != "Y"]
                for em in emotions
            }
            for v in spare_by_em.values():
                v.sort(key=lambda i: (df.at[i, "video_id"], df.at[i, "start_sec"]))

            progress = True
            while remaining > 0 and progress:
                progress = False
                for em in emotions:
                    if remaining <= 0:
                        break
                    if not spare_by_em[em]:
                        continue
                    i = spare_by_em[em].pop(0)
                    df.at[i, "selected"] = "Y"
                    remaining -= df.at[i, "duration_sec"]
                    progress = True

    return df


def build_manifest() -> pd.DataFrame:
    log.info("reading sources from %s", SOURCES_CSV)
    if not SOURCES_CSV.exists():
        log.error("sources file not found: %s", SOURCES_CSV)
        sys.exit(1)

    header_row = _find_header_row(SOURCES_CSV)
    log.info("detected header at row %d (0-indexed)", header_row)
    df = pd.read_csv(SOURCES_CSV, skiprows=header_row)
    df.columns = [str(c).strip() for c in df.columns]
    log.info("loaded %d source rows; columns: %s", len(df), list(df.columns))

    base_rows = []
    issues = []
    seen_spans = set()

    for idx, row in df.iterrows():
        url = _pick(row, "Source URL")
        video_id = extract_video_id(url)
        if not video_id:
            log.warning("row %s: bad/missing URL: %r", idx, url)
            issues.append((idx, "bad/missing URL", url))
            continue

        try:
            start_sec = parse_mmss(_pick(row, "Start", "Start (mm:ss)"))
            end_sec = parse_mmss(_pick(row, "End", "End (mm:ss)"))
        except Exception as e:
            log.warning("row %s: bad time (%s) for %s", idx, e, url)
            issues.append((idx, f"bad time ({e})", url))
            continue

        if end_sec <= start_sec:
            log.warning("row %s: end<=start (%ss..%ss) for %s", idx, start_sec, end_sec, url)
            issues.append((idx, "end<=start", url))
            continue

        lang_raw = _pick(row, "Language")
        lang = normalize_language(lang_raw)
        if not lang:
            log.warning("row %s: unknown language %r for %s", idx, lang_raw, url)
            issues.append((idx, f"unknown language: {lang_raw!r}", url))
            continue

        span_key = (video_id, start_sec, end_sec)
        if span_key in seen_spans:
            log.debug("row %s: duplicate span %s, skipping", idx, span_key)
            continue
        seen_spans.add(span_key)
        log.debug("row %s: %s [%s] %s..%s (%ss) lang=%s",
                  idx, video_id, _pick(row, "Emotion"), start_sec, end_sec,
                  end_sec - start_sec, lang)

        parent_clip_id = f"{video_id}_{start_sec}"
        base_rows.append({
            "parent_clip_id": parent_clip_id,
            "clip_id": parent_clip_id,
            "video_id": video_id,
            "speaker_id": video_id,
            "language": lang,
            "emotion": normalize_emotion(_pick(row, "Emotion")),
            "source_url": str(url).split("&list=")[0],
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": end_sec - start_sec,
            "chunk_index": 0,
            "transcription": "",
            "transcribed": "",
        })

    log.info("kept %d valid source spans (skipped %d)", len(base_rows), len(issues))

    # Chunk into ≤30s segments
    chunked = []
    for r in base_rows:
        sub = split_into_chunks(r)
        if len(sub) > 1:
            log.debug("chunked %s (%ss) -> %d pieces",
                      r["clip_id"], r["duration_sec"], len(sub))
        chunked.extend(sub)
    log.info("expanded into %d ≤%ss chunks", len(chunked), CHUNK_SEC)

    manifest = pd.DataFrame(chunked)
    if manifest.empty:
        log.warning("no usable rows after chunking")
        return manifest

    # Apply budget selection
    log.info("running budget selection: target=%ss total, %ss per language",
             TOTAL_TARGET_SEC, PER_LANG_TARGET_SEC)
    manifest = select_budget(manifest)

    # Reporting
    sel = manifest[manifest["selected"] == "Y"]
    log.info("=== Selection summary ===")
    for lang in sorted(manifest["language"].unique()):
        lang_sel = sel[sel["language"] == lang]
        total = lang_sel["duration_sec"].sum()
        log.info("  %s: %ss (%.1f min) across %d chunks",
                 lang, total, total / 60, len(lang_sel))
        for em in sorted(lang_sel["emotion"].unique()):
            cell = lang_sel[lang_sel["emotion"] == em]
            log.info("     %-14s %4ss  (%d chunks)",
                     em, cell["duration_sec"].sum(), len(cell))
    grand = sel["duration_sec"].sum()
    log.info("  TOTAL selected: %ss (%.1f min) of %ss target (%d chunks)",
             grand, grand / 60, TOTAL_TARGET_SEC, len(sel))

    if issues:
        log.warning("skipped %d source rows:", len(issues))
        for idx, reason, url in issues:
            log.warning("  - row %s: %s :: %s", idx, reason, url)

    return manifest


def merge_prior_transcriptions(manifest: pd.DataFrame) -> pd.DataFrame:
    """If a previous manifest exists, copy its transcription/transcribed values
    onto rows in the new manifest that share the same clip_id."""
    if not MANIFEST_CSV.exists():
        return manifest
    try:
        old = pd.read_csv(MANIFEST_CSV)
    except Exception as e:
        log.warning("could not read prior manifest (%s); not merging", e)
        return manifest
    if "clip_id" not in old.columns or "transcription" not in old.columns:
        return manifest
    old = old[["clip_id", "transcription", "transcribed"]].copy()
    old["transcription"] = old["transcription"].astype("object").fillna("")
    old["transcribed"] = old["transcribed"].astype("object").fillna("")
    lookup = old.set_index("clip_id")
    carried = 0
    for idx, row in manifest.iterrows():
        cid = row["clip_id"]
        if cid in lookup.index:
            t = lookup.at[cid, "transcription"]
            d = lookup.at[cid, "transcribed"]
            if isinstance(t, str) and t:
                manifest.at[idx, "transcription"] = t
            if isinstance(d, str) and d.strip().upper() == "Y":
                manifest.at[idx, "transcribed"] = "Y"
                carried += 1
    log.info("carried %d prior transcriptions forward from existing manifest", carried)
    return manifest


def main():
    manifest = build_manifest()
    manifest = merge_prior_transcriptions(manifest)
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_CSV, index=False)
    selected = (manifest["selected"] == "Y").sum() if "selected" in manifest.columns else 0
    log.info("wrote %d chunk rows to %s (%d selected for download)",
             len(manifest), MANIFEST_CSV, selected)


if __name__ == "__main__":
    main()
