"""AutoSCI — Multi-idea paper compilation (v0.7).

Given a list of paper.md paths (typically from ``vault/kept/<id>/paper.md``),
produce a single compiled ``compilation.md`` and ``compilation.html``
that groups them by modality / verdict and writes an executive summary
table on top.

Usage:
    python -m autosci.compilation --vault-dir vault/kept --out compilation.html
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from autosci.html_report import _md_to_html, _slugify


# ---------------------------------------------------------------------------
# Markdown → section parsing (re-using the format produced by PaperWriter)
# ---------------------------------------------------------------------------


H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.M)


def parse_paper_sections(paper_md: str) -> dict[str, str]:
    """Return {section_name: body} for every `## name` block.

    The title (first `# ...`) is exposed under key ``"title"`` (no body).
    """
    out: dict[str, str] = {}
    title = None
    # find the H1 first
    m = re.search(r"^#\s+(.+?)\s*$", paper_md, re.M)
    if m:
        title = m.group(1).strip()
        out["title"] = title
    # iterate H2 sections
    matches = list(H2_RE.finditer(paper_md))
    for i, mm in enumerate(matches):
        name = mm.group(1).strip()
        # Skip the auto-generated "Internal Audit" section — it's machine noise
        if name.startswith("Internal"):
            continue
        body_start = mm.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(paper_md)
        out[name] = paper_md[body_start:body_end].strip()
    return out


def paper_id_from_path(p: Path) -> str:
    return p.parent.name


def verdict_from_state(state_path: Path) -> str:
    if not state_path.exists():
        return "unknown"
    try:
        import json
        s = json.loads(state_path.read_text(encoding="utf-8", errors="replace"))
        return s.get("verdict", "unknown")
    except Exception:
        return "unknown"


def modalities_from_state(state_path: Path) -> list[str]:
    if not state_path.exists():
        return []
    try:
        import json
        s = json.loads(state_path.read_text(encoding="utf-8", errors="replace"))
        seed = (s.get("idea_seed") or {})
        return [m for m in (seed.get("modalities") or [])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Compilation builders
# ---------------------------------------------------------------------------


def build_compilation_markdown(vault_dir: Path, kept_only: bool = True) -> str:
    """Walk vault/(kept|refuted)/<id>/paper.md and produce a single markdown doc."""
    if not vault_dir.exists():
        return f"# AutoSCI Compilation\n\n(no vault dir at {vault_dir})"
    rows: list[dict] = []
    for paper in sorted(vault_dir.glob("*/paper.md")):
        verdict = verdict_from_state(paper.parent / "state.json")
        if kept_only and verdict not in ("kept", "revised"):
            continue
        sections = parse_paper_sections(paper.read_text(encoding="utf-8", errors="replace"))
        rows.append({
            "id": paper_id_from_path(paper),
            "verdict": verdict,
            "title": sections.get("title", paper.parent.name),
            "abstract": sections.get("Abstract", "").strip()[:600],
            "modalities": modalities_from_state(paper.parent / "state.json"),
        })
    rows.sort(key=lambda r: ({"kept": 0, "revised": 1, "refuted": 2}.get(r["verdict"], 9), r["id"]))

    out: list[str] = ["# AutoSCI Compilation", ""]
    out.append(f"_{len(rows)} papers in this bundle. Vault: `{vault_dir}`._")
    out.append("")

    # Executive summary table
    out.append("## Executive Summary")
    out.append("")
    out.append("| ID | Verdict | Title | Modalities |")
    out.append("|----|---------|-------|------------|")
    for r in rows:
        mods = ", ".join(r["modalities"]) or "—"
        out.append(f"| `{r['id']}` | {r['verdict']} | {r['title'][:60]} | {mods} |")
    out.append("")

    # Group by verdict
    by_verdict: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_verdict[r["verdict"]].append(r)

    for verdict in ("kept", "revised", "refuted", "unknown"):
        group = by_verdict.get(verdict, [])
        if not group:
            continue
        out.append(f"## {verdict.title()} ({len(group)})")
        out.append("")
        for r in group:
            out.append(f"### {r['title']}")
            out.append("")
            if r["abstract"]:
                out.append(f"_{r['abstract']}_")
                out.append("")
            out.append(f"**Verdict:** `{r['verdict']}` &nbsp; **Modalities:** {', '.join(r['modalities']) or '—'}")
            out.append("")

    return "\n".join(out)


def build_compilation_html(vault_dir: Path, kept_only: bool = True) -> str:
    md = build_compilation_markdown(vault_dir, kept_only=kept_only)
    body = _md_to_html(md)
    return f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>AutoSCI Compilation</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif;
          line-height: 1.55; max-width: 920px; margin: 2rem auto; padding: 0 1rem; color: #1f2937; }}
  h1 {{ border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.8rem 0; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 0.4rem 0.6rem; text-align: left; font-size: 0.92rem; }}
  th {{ background: #f9fafb; }}
  h2 {{ margin-top: 1.4rem; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.3rem; }}
  h3 {{ margin-top: 1.0rem; }}
  pre {{ background: #f6f8fa; padding: 0.7rem; border-radius: 4px; overflow-x: auto; font-size: 0.85rem; }}
  code {{ background: #f6f8fa; padding: 0.1rem 0.3rem; border-radius: 3px; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile multiple AutoSCI papers into one report.")
    parser.add_argument("--vault-dir", required=True, help="Path to vault/kept/ or vault/refuted/")
    parser.add_argument("--out", required=True, help="Output .html or .md file")
    parser.add_argument("--include-refuted", action="store_true",
                        help="Also include refuted papers (default: kept + revised only)")
    args = parser.parse_args(argv)

    v = Path(args.vault_dir)
    if args.out.endswith(".html"):
        out_text = build_compilation_html(v, kept_only=not args.include_refuted)
    else:
        out_text = build_compilation_markdown(v, kept_only=not args.include_refuted)
    Path(args.out).write_text(out_text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
