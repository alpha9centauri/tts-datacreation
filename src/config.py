import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CLIPS_DIR = DATA_DIR / "clips"
RAW_DIR = DATA_DIR / "raw"
SOURCES_CSV = DATA_DIR / "sources.csv"
MANIFEST_CSV = DATA_DIR / "manifest.csv"

SARVAM_MAX_SYNC_SECONDS = 30

load_dotenv(ROOT / ".env")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")


def require_sarvam_key() -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY missing. Copy .env.example to .env and set it.")
    return SARVAM_API_KEY


def setup_logging(name: str, level: str = None) -> logging.Logger:
    """Configure root logger with timestamps + levels and return a named child."""
    lvl_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    lvl = getattr(logging, lvl_name, logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            fmt="%(asctime)s [%(levelname)-5s] %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(h)
    root.setLevel(lvl)
    return logging.getLogger(name)
