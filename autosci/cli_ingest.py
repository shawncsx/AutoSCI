"""AutoSCI — CLI to ingest the digest folder and dump a JSONL.

Usage:
    python -m autosci.cli_ingest --digest-dir D:/shawn/paper/_digest --out data/kb_corpus.jsonl
"""
from __future__ import annotations

import argparse
import logging
import sys

from autosci.kb import KnowledgeBase


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--digest-dir", required=True)
    parser.add_argument("--out", default="data/kb_corpus.jsonl")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    log = logging.getLogger("ingest")

    kb = KnowledgeBase.from_digest_dir(args.digest_dir)
    log.info("doc count: %d", len(kb.docs))
    log.info("first 3:")
    for d in kb.docs[:3]:
        log.info("  %s | %s (%d) sections=%s", d.doc_id, d.title[:60], d.year, list(d.sections))

    kb.to_jsonl(args.out)
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
