"""AutoSCI — HTML report generator (v0.6).

Converts a ``runs/<idea>/paper.md`` (or vault copy) into a self-contained
HTML page with:
    - a sticky table-of-contents
    - a cover page (title, abstract, verdict badge)
    - collapsible <details> per section (mobile-friendly)
    - code blocks pre-rendered with simple markdown-to-HTML (no external libs)

Usage:
    python -m autosci.html_report --paper runs/<id>/paper.md --out report.html
    python -m autosci.html_report --vault-dir vault/kept --out report.html  # index page
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path


def _slugify(s: str) -> str:
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", s.lower()).strip("-")
    return s or "section"


def _md_to_html(text: str) -> str:
    """Very small markdown subset: headings (handled by caller), bold, italic,
    inline code, fenced code blocks, links, and paragraphs."""
    lines: list[str] = []
    in_code = False
    code_buf: list[str] = []
    para_buf: list[str] = []

    def flush_para():
        if para_buf:
            content = " ".join(para_buf).strip()
            if content:
                lines.append(f"<p>{_inline(content)}</p>")
            para_buf.clear()

    for raw in text.splitlines():
        if raw.strip().startswith("```"):
            if in_code:
                # close code block
                esc = html.escape("\n".join(code_buf))
                lines.append(f"<pre><code>{esc}</code></pre>")
                code_buf.clear()
                in_code = False
            else:
                flush_para()
                in_code = True
            continue
        if in_code:
            code_buf.append(raw)
            continue
        if not raw.strip():
            flush_para()
            continue
        # detect heading (##)
        m = re.match(r"^(#{1,6})\s+(.*)$", raw)
        if m:
            flush_para()
            level = len(m.group(1))
            content = _inline(m.group(2))
            lines.append(f"<h{level}>{content}</h{level}>")
            continue
        para_buf.append(raw)

    flush_para()
    return "\n".join(lines)


def _inline(s: str) -> str:
    # Escape first
    s = html.escape(s)
    # bold **x**
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    # italic *x*
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    # inline code `x`
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    # links [text](url)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


_VERDICT_COLORS = {
    "kept": "#16a34a",  # green
    "revised": "#ca8a04",  # amber
    "refuted": "#dc2626",  # red
    "pending": "#6b7280",  # gray
}


def render_paper_html(paper_md: str, *, title_override: str | None = None,
                      verdict: str = "pending") -> str:
    """Render a single paper.md to a self-contained HTML document."""
    title = title_override
    abstract = ""
    keywords: list[str] = []
    toc: list[tuple[str, str]] = []  # (slug, heading)
    body_chunks: list[str] = []
    references: list[str] = []

    current_section: str = ""
    current_body: list[str] = []

    def flush_section():
        if not current_section:
            return
        body = "\n".join(current_body).strip()
        slug = _slugify(current_section)
        toc.append((slug, current_section))
        body_html = _md_to_html(body)
        # Wrap in a collapsible <details>; default open
        body_chunks.append(
            f'<details open class="section" id="{slug}">'
            f'<summary>{html.escape(current_section)}</summary>'
            f'{body_html}'
            f'</details>'
        )
        current_body.clear()

    in_refs = False
    for raw in paper_md.splitlines():
        line = raw.rstrip()
        if line.startswith("# "):
            # main title
            if title is None:
                title = line[2:].strip()
            continue
        if line.startswith("## ") and not line.startswith("## Internal"):
            # a new section
            flush_section()
            current_section = line[3:].strip()
            in_refs = (current_section.lower() == "references")
            continue
        if not in_refs:
            current_body.append(line)
        else:
            if line.strip():
                references.append(line.strip())

    flush_section()

    color = _VERDICT_COLORS.get(verdict, "#6b7280")
    title_str = title or "Untitled"
    body_html = "\n".join(body_chunks)
    toc_html = "\n".join(
        f'<li><a href="#{slug}">{html.escape(h)}</a></li>' for slug, h in toc
    )
    refs_html = (
        "<ol>" + "".join(f"<li>{html.escape(r)}</li>" for r in references) + "</ol>"
        if references else "<p><em>No references.</em></p>"
    )

    return f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{html.escape(title_str)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; line-height: 1.55; max-width: 920px; margin: 2rem auto; padding: 0 1rem; color: #1f2937; }}
  h1 {{ font-size: 1.7rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem; }}
  h2 {{ font-size: 1.3rem; margin-top: 1.6rem; }}
  .cover {{ padding: 1rem 1.2rem; border-radius: 8px; background: #f9fafb; border-left: 4px solid {color}; margin-bottom: 1.4rem; }}
  .verdict {{ display: inline-block; padding: 0.15rem 0.6rem; border-radius: 4px; color: white; background: {color}; font-size: 0.85rem; font-weight: 600; }}
  .toc {{ position: sticky; top: 1rem; padding: 0.8rem 1rem; background: #f3f4f6; border-radius: 6px; margin-bottom: 1.4rem; }}
  .toc ol {{ margin: 0.3rem 0 0 1.1rem; padding: 0; }}
  details.section {{ margin: 0.8rem 0; padding: 0.6rem 0.9rem; border: 1px solid #e5e7eb; border-radius: 6px; background: #fff; }}
  details.section > summary {{ cursor: pointer; font-weight: 600; }}
  pre {{ background: #f6f8fa; padding: 0.7rem; border-radius: 4px; overflow-x: auto; font-size: 0.85rem; }}
  code {{ background: #f6f8fa; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.9em; }}
  .meta {{ color: #6b7280; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>{html.escape(title_str)}</h1>
<div class="cover">
  <span class="verdict">{html.escape(verdict)}</span>
  <p class="meta">Generated by AutoSCI v0.6</p>
</div>

<details open class="toc">
<summary>📑 目录 / Table of Contents</summary>
<ol>{toc_html}</ol>
</details>

{body_html}

<h2 id="references">References</h2>
{refs_html}
</body>
</html>
"""


def render_paper_to_file(paper_md_path: Path, out_path: Path,
                         verdict: str | None = None) -> Path:
    """Render a single paper.md file to HTML."""
    text = paper_md_path.read_text(encoding="utf-8", errors="replace")
    title = paper_md_path.parent.name
    # Try to read state.json for verdict if not provided
    if not verdict:
        sj = paper_md_path.parent / "state.json"
        if sj.exists():
            try:
                state = json.loads(sj.read_text(encoding="utf-8", errors="replace"))
                verdict = (state.get("verdict") or "pending")
            except Exception:
                verdict = "pending"
    html_text = render_paper_html(text, title_override=title, verdict=verdict or "pending")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render AutoSCI paper.md → HTML")
    parser.add_argument("--paper", help="Path to a paper.md")
    parser.add_argument("--vault-dir", help="Render all papers in this vault subdir (kept/ or refuted/)")
    parser.add_argument("--out", required=True, help="Output path (HTML file or directory)")
    args = parser.parse_args(argv)

    if not args.paper and not args.vault_dir:
        parser.error("Either --paper or --vault-dir is required")

    if args.paper:
        out = Path(args.out)
        render_paper_to_file(Path(args.paper), out)
        print(f"wrote {out}")
    else:
        v = Path(args.vault_dir)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(v.glob("*/paper.md")):
            verdict = p.parent.name and ("kept" if "kept" in str(v) else "refuted")
            target = out_dir / f"{p.parent.name}.html"
            render_paper_to_file(p, target, verdict=verdict or "kept")
            print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
