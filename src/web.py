"""Tiny review UI for the dataset.

Run:  python src/web.py
Then open http://127.0.0.1:5050
"""
import argparse
from threading import Lock

import pandas as pd
from flask import Flask, abort, jsonify, render_template_string, request, send_file

from config import CLIPS_DIR, MANIFEST_CSV, setup_logging

log = setup_logging("web")
app = Flask(__name__)
_csv_lock = Lock()


def _read_manifest() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST_CSV)
    if "transcription" not in df.columns:
        df["transcription"] = ""
    if "reviewed" not in df.columns:
        df["reviewed"] = ""
    df["transcription"] = df["transcription"].astype("object").fillna("")
    df["reviewed"] = df["reviewed"].astype("object").fillna("")
    return df


@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


@app.get("/api/clips")
def list_clips():
    df = _read_manifest()
    df = df[df["selected"].astype(str).str.upper() == "Y"].copy()
    cols = [
        "clip_id", "language", "emotion", "duration_sec",
        "transcription", "reviewed", "source_url", "start_sec", "end_sec",
    ]
    return jsonify({"clips": df[cols].to_dict(orient="records")})


@app.get("/audio/<clip_id>")
def audio(clip_id: str):
    if "/" in clip_id or ".." in clip_id:
        abort(400)
    path = CLIPS_DIR / f"{clip_id}.wav"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="audio/wav", conditional=True)


@app.post("/api/clips/<clip_id>")
def update_clip(clip_id: str):
    data = request.get_json(silent=True) or {}
    with _csv_lock:
        df = _read_manifest()
        mask = df["clip_id"] == clip_id
        if not mask.any():
            abort(404)
        if "transcription" in data:
            df.loc[mask, "transcription"] = str(data["transcription"])
        if "reviewed" in data:
            df.loc[mask, "reviewed"] = "Y" if data["reviewed"] else ""
        df.to_csv(MANIFEST_CSV, index=False)
    log.info("updated %s (reviewed=%s, %d chars)",
             clip_id, data.get("reviewed"), len(str(data.get("transcription", ""))))
    return jsonify({"ok": True})


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Clip Reviewer</title>
<style>
  :root {
    --bg:#0f1115; --fg:#e8e8ea; --muted:#9aa0a6;
    --card:#171a21; --card2:#1d2129;
    --accent:#5b9dff; --good:#3acb78; --border:#2a2f3a;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; background: var(--bg); color: var(--fg); }
  body { font: 14px/1.4 -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif; }
  header { padding: 14px 24px; border-bottom: 1px solid var(--border);
           display: flex; align-items: center; gap: 14px; }
  h1 { margin: 0; font-size: 16px; font-weight: 600; letter-spacing: 0.2px; }
  .stats { color: var(--muted); font-size: 13px; margin-left: auto; }
  .filters { padding: 10px 24px; border-bottom: 1px solid var(--border);
             display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .filters select, .filters input {
    background: var(--card2); color: var(--fg); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 10px; font-size: 13px; font-family: inherit;
  }
  .filters input[type="text"] { min-width: 240px; }
  main { padding: 16px 24px 80px; display: grid; gap: 10px; max-width: 1100px; margin: 0 auto; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
  .card.reviewed { border-left: 3px solid var(--good); }
  .row1 { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .clipid { font-family: ui-monospace, "SF Mono", monospace; font-size: 12px; color: var(--muted); }
  .tag { font-size: 11px; padding: 2px 7px; border-radius: 999px; background: #232a36; color: var(--fg); }
  .tag.lang { background: #1f3055; }
  .tag.emo { background: #3a2a55; }
  .sourcelink { color: var(--accent); font-size: 12px; text-decoration: none; margin-left: auto; }
  .sourcelink:hover { text-decoration: underline; }
  audio { width: 100%; margin-top: 8px; height: 36px; }
  textarea {
    width: 100%; min-height: 64px; margin-top: 8px; padding: 8px 10px;
    background: var(--card2); color: var(--fg); border: 1px solid var(--border);
    border-radius: 6px; font-size: 14px; font-family: inherit; resize: vertical;
    line-height: 1.45;
  }
  textarea:focus { outline: none; border-color: var(--accent); }
  .actions { display: flex; gap: 10px; margin-top: 8px; align-items: center; }
  button { background: var(--accent); color: white; border: 0; padding: 6px 14px;
           border-radius: 6px; font-size: 13px; cursor: pointer; font-family: inherit; }
  button:disabled { opacity: 0.5; cursor: default; }
  .reviewed-chk { display: flex; gap: 6px; align-items: center; color: var(--muted);
                  font-size: 13px; cursor: pointer; user-select: none; }
  .reviewed-chk input { accent-color: var(--good); }
  .saved-flash { color: var(--good); font-size: 12px; opacity: 0; transition: opacity 0.3s; }
  .saved-flash.show { opacity: 1; }
  .empty { padding: 60px 0; text-align: center; color: var(--muted); }
  kbd { font-family: ui-monospace, monospace; font-size: 11px; padding: 1px 5px;
        background: var(--card2); border: 1px solid var(--border); border-radius: 4px; color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Clip Reviewer</h1>
  <div class="stats" id="stats">loading…</div>
</header>
<div class="filters">
  <select id="lang"><option value="">All languages</option></select>
  <select id="emo"><option value="">All emotions</option></select>
  <select id="rev">
    <option value="">All statuses</option>
    <option value="N">Unreviewed</option>
    <option value="Y">Reviewed</option>
  </select>
  <input id="q" type="text" placeholder="Search transcript or clip id…" />
  <span style="color:var(--muted);font-size:12px;margin-left:auto;">
    Save: <kbd>⌘S</kbd> / <kbd>Ctrl+S</kbd>
  </span>
</div>
<main id="list"></main>

<script>
let clips = [];

async function load() {
  const r = await fetch('/api/clips');
  const j = await r.json();
  clips = j.clips;
  populateFilters();
  render();
}

function uniq(arr) { return [...new Set(arr)].sort(); }

function populateFilters() {
  for (const l of uniq(clips.map(c => c.language))) {
    const o = document.createElement('option'); o.value = l; o.textContent = l;
    document.getElementById('lang').appendChild(o);
  }
  for (const e of uniq(clips.map(c => c.emotion))) {
    const o = document.createElement('option'); o.value = e; o.textContent = e;
    document.getElementById('emo').appendChild(o);
  }
}

function filtered() {
  const l = document.getElementById('lang').value;
  const e = document.getElementById('emo').value;
  const r = document.getElementById('rev').value;
  const q = document.getElementById('q').value.toLowerCase().trim();
  return clips.filter(c => {
    if (l && c.language !== l) return false;
    if (e && c.emotion !== e) return false;
    if (r === 'Y' && c.reviewed !== 'Y') return false;
    if (r === 'N' && c.reviewed === 'Y') return false;
    if (q) {
      const hay = (c.clip_id + ' ' + (c.transcription || '')).toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

function updateStats() {
  const reviewed = clips.filter(c => c.reviewed === 'Y').length;
  document.getElementById('stats').textContent =
    `${filtered().length} / ${clips.length} shown · ${reviewed} reviewed`;
}

function render() {
  const list = document.getElementById('list');
  const items = filtered();
  list.innerHTML = '';
  if (items.length === 0) {
    list.innerHTML = '<div class="empty">No clips match filters.</div>';
    updateStats();
    return;
  }
  for (const c of items) list.appendChild(card(c));
  updateStats();
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g,
    ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

function card(c) {
  const div = document.createElement('div');
  div.className = 'card' + (c.reviewed === 'Y' ? ' reviewed' : '');
  const srcWithT = `${c.source_url}${c.source_url.includes('?') ? '&' : '?'}t=${c.start_sec}s`;
  div.innerHTML = `
    <div class="row1">
      <span class="clipid">${esc(c.clip_id)}</span>
      <span class="tag lang">${esc(c.language)}</span>
      <span class="tag emo">${esc(c.emotion)}</span>
      <span class="tag">${esc(c.duration_sec)}s</span>
      <a class="sourcelink" target="_blank" rel="noopener" href="${esc(srcWithT)}">source ↗</a>
    </div>
    <audio controls preload="none" src="/audio/${encodeURIComponent(c.clip_id)}"></audio>
    <textarea spellcheck="false">${esc(c.transcription || '')}</textarea>
    <div class="actions">
      <button class="save">Save</button>
      <label class="reviewed-chk">
        <input type="checkbox" class="rev" ${c.reviewed === 'Y' ? 'checked' : ''}/> reviewed
      </label>
      <span class="saved-flash">✓ saved</span>
    </div>
  `;
  const ta = div.querySelector('textarea');
  const saveBtn = div.querySelector('.save');
  const revBox = div.querySelector('.rev');
  const flash = div.querySelector('.saved-flash');

  async function save() {
    saveBtn.disabled = true;
    try {
      const r = await fetch('/api/clips/' + encodeURIComponent(c.clip_id), {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({transcription: ta.value, reviewed: revBox.checked}),
      });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      c.transcription = ta.value;
      c.reviewed = revBox.checked ? 'Y' : '';
      div.classList.toggle('reviewed', revBox.checked);
      flash.classList.add('show');
      setTimeout(() => flash.classList.remove('show'), 1200);
      updateStats();
    } catch (err) {
      alert('Save failed: ' + err.message);
    } finally {
      saveBtn.disabled = false;
    }
  }

  saveBtn.addEventListener('click', save);
  revBox.addEventListener('change', save);
  ta.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 's') { e.preventDefault(); save(); }
  });
  return div;
}

['lang', 'emo', 'rev', 'q'].forEach(id =>
  document.getElementById(id).addEventListener('input', render));

load();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()
    log.info("serving manifest=%s clips=%s on http://%s:%d",
             MANIFEST_CSV, CLIPS_DIR, args.host, args.port)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
