# data/pdf_fulltext/

This directory stores full-text dumps of arXiv papers fetched by
`autosci/arxiv_pdf.py`.

## Why no PDF binaries here?

The original arXiv PDFs (e.g. `arxiv_2402_16901.pdf`) are 10-15 MB each.
The GitHub App used to push to `github.com/shawncsx/AutoSCI` returns
HTTP 502 when the request payload exceeds ~10 MB, so the binary PDFs
were excluded from the public release.

## What's here

- `<arxiv-id>.fulltext.md` — markdown dump with all pages' text
  extracted by `pypdf`. Lossless for our purposes (the paper's prose
  is the input to IdeaMiner / MethodDesigner anyway).
- `<arxiv-id>.txt` — optional plain-text companion.

## How to regenerate the binaries locally

If you actually need the raw PDF (e.g. for figures), run:

```bash
python -c "from autosci.arxiv_pdf import fetch_one_pdf; \
  from pathlib import Path; \
  r = fetch_one_pdf('2402.16901', Path('data/pdf_fulltext')); \
  print('PDF saved to', r.pdf_path)"
```

## A typical fulltext file

A 20-page arXiv paper becomes a ~75 KB markdown file with structure:

```
# arxiv_fulltext:2402.16901

## [开头/摘要]

**arXiv id:** 2402.16901
**PDF URL:** https://arxiv.org/pdf/2402.16901
**Pages:** 20
**Characters:** 72,291

**Full text:**

## page 1
<page 1 text>

## page 2
<page 2 text>
...
```

The Director uses this content as ground-truth context for
`MethodDesigner` and `ExpDesigner` instead of relying solely on the
abstract.
