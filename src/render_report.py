"""Render REPORT.md to a print-ready HTML file.

Usage:
    python src/render_report.py
    open REPORT.html        # then Cmd+P -> Save as PDF
"""
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "REPORT.md"
DST = ROOT / "REPORT.html"

CSS = r"""
@page { size: A4; margin: 22mm 20mm 24mm; }
:root {
  --fg: #1a1a1a;
  --muted: #555;
  --accent: #1f4ea3;
  --rule: #d8dce3;
  --code-bg: #f4f5f7;
  --table-head: #f0f2f5;
}
html, body { background: #fff; color: var(--fg); }
body {
  font-family: "Charter", "Georgia", "Iowan Old Style", "Source Serif Pro", serif;
  font-size: 11pt;
  line-height: 1.55;
  max-width: 780px;
  margin: 32px auto;
  padding: 0 28px 56px;
}
h1 { font-size: 22pt; margin: 0 0 4pt; letter-spacing: -0.2px; }
h2 { font-size: 14pt; margin: 22pt 0 8pt; padding-top: 6pt; border-top: 1px solid var(--rule); color: var(--accent); }
h3 { font-size: 12pt; margin: 16pt 0 4pt; color: #233; }
h4 { font-size: 11pt; margin: 12pt 0 2pt; color: #333; font-weight: 600; }
p { margin: 6pt 0; }
ul, ol { padding-left: 22pt; margin: 6pt 0; }
li { margin: 2pt 0; }
strong { color: #111; }
em { color: #244; }
code {
  font-family: "SF Mono", "JetBrains Mono", "Menlo", "Consolas", monospace;
  font-size: 9.5pt;
  background: var(--code-bg);
  padding: 1.5px 4px;
  border-radius: 3px;
}
pre {
  background: var(--code-bg);
  padding: 9pt 12pt;
  border-radius: 5px;
  overflow-x: auto;
  font-size: 9pt;
  line-height: 1.45;
  border: 1px solid var(--rule);
  page-break-inside: avoid;
}
pre code { background: transparent; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; font-size: 10pt; page-break-inside: avoid; }
th, td { border: 1px solid var(--rule); padding: 5pt 8pt; text-align: left; vertical-align: top; }
th { background: var(--table-head); font-weight: 600; }
hr { border: none; border-top: 1px solid var(--rule); margin: 18pt 0; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
blockquote {
  margin: 8pt 0; padding: 4pt 12pt; color: var(--muted);
  border-left: 3px solid var(--rule); background: #fafbfc;
}
.meta { color: var(--muted); font-size: 10pt; margin: 0 0 12pt; }
h1 + p, h1 + ul { font-size: 10.5pt; color: var(--muted); }
h2, h3 { page-break-after: avoid; }
table, pre, blockquote { page-break-inside: avoid; }
"""

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def main():
    if not SRC.exists():
        raise SystemExit(f"missing source: {SRC}")
    text = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        output_format="html5",
    )
    html = HTML_TEMPLATE.format(
        title="Sarvam TTS Dataset — Construction & Curation Report",
        css=CSS,
        body=body,
    )
    DST.write_text(html, encoding="utf-8")
    print(f"wrote {DST}")
    print("open in your browser and Cmd+P -> Save as PDF")


if __name__ == "__main__":
    main()
