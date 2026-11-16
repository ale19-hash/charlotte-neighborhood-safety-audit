"""Render D3_modeling_note.md to D3_modeling_note.pdf (pandoc + headless Chrome).

Run from the repository root after evaluate_models.py:
    python d3_modeling/build_note.py
Set CHROME=/path/to/chrome if Chrome is not in a standard location.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD, PDF = ROOT / "D3_modeling_note.md", ROOT / "D3_modeling_note.pdf"
HTML = ROOT / "d3_modeling" / "D3_modeling_note_print.html"

CSS = """
@page { size: Letter; margin: 0.6in 0.65in; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 9.3pt; line-height: 1.33;
       color: #1a1a19; background: #fff; max-width: none; margin: 0; }
h1 { font-family: 'Helvetica Neue', Arial, sans-serif; color: #17324d; font-size: 19pt;
     line-height: 1.15; margin: 0 0 4pt; }
h1 + p strong { color: #17324d; }
h2 { font-family: 'Helvetica Neue', Arial, sans-serif; color: #17324d; font-size: 12pt;
     border-bottom: 1px solid #b9c6d3; padding-bottom: 2pt; margin: 9pt 0 4pt; }
p { margin: 0 0 6pt; }
ul { margin: 0 0 6pt; padding-left: 16pt; } li { margin-bottom: 3pt; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.4pt; background: #eef1f4;
       padding: 0 2px; border-radius: 2px; }
table { border-collapse: collapse; width: 100%; font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 7.9pt; margin: 2pt 0 8pt; }
th, td { border-bottom: 1px solid #d6dde4; padding: 2.5pt 4pt; text-align: left; }
th { background: #eef1f4; color: #17324d; }
img { display: block; max-width: 70%; margin: 4pt auto 2pt; }
figure { margin: 0; } figcaption { display: none; }
p:has(> img) + p, p:has(> strong:first-child) { }
.pb { break-after: page; }
"""


def find_chrome() -> str:
    for c in [os.environ.get("CHROME"),
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "/Applications/Google Chrome 2.app/Contents/MacOS/Google Chrome",
              "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
              shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("msedge")]:
        if c and Path(c).exists():
            return c
    raise SystemExit("Chrome/Edge not found; set CHROME=/path/to/browser")


def main() -> None:
    md = MD.read_text().replace("<!-- PAGE BREAK -->", '<div class="pb"></div>')
    body = subprocess.run(["pandoc", "-f", "gfm", "-t", "html"], input=md, text=True,
                          capture_output=True, check=True).stdout
    HTML.write_text(f'<!doctype html><html><head><meta charset="utf-8">'
                    f'<title>D3 modeling note</title><base href="{ROOT.as_uri()}/">'
                    f"<style>{CSS}</style></head><body>{body}</body></html>")
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([find_chrome(), "--headless", "--disable-gpu", "--no-first-run",
                        "--no-pdf-header-footer", f"--user-data-dir={prof}",
                        f"--print-to-pdf={PDF}", HTML.as_uri()],
                       check=True, capture_output=True)
    print("Created", PDF)


if __name__ == "__main__":
    main()
