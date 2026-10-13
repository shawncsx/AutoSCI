"""AutoSCI — Director's cross-idea memory.

Tracks every idea that has been attempted, plus its verdict + a short
summary, so future IdeaMiner calls and Director cycles can avoid
re-running the same topic.

Persists to ``runs/director_memory.jsonl`` (append-only log) and
re-loads on next Director run.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from autosci.schemas import IdeaSeed, Verdict


log = logging.getLogger("autosci.memory")


@dataclass
class MemoryEntry:
    id: str
    title: str
    modalities: list[str]
    verdict: str
    fatal_flaws: list[str] = field(default_factory=list)
    refuted_keywords: list[str] = field(default_factory=list)  # tokens to avoid re-mining
    at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def _keywords(text: str, k: int = 8) -> list[str]:
    """Pick a few salient tokens from free text — naive but useful."""
    text = text.lower()
    text = re.sub(r"[\s\W_]+", " ", text, flags=re.UNICODE)
    toks = [w for w in text.split() if 3 <= len(w) <= 20 and not w.isdigit()]
    # most common k
    return [t for t, _ in Counter(toks).most_common(k)]


def _entry_from_state(state) -> MemoryEntry:
    fatal_kw: list[str] = []
    if state.refute and state.refute.fatal_flaws:
        for f in state.refute.fatal_flaws[:5]:
            text = f.get("flaw", "") + " " + f.get("evidence", "")
            fatal_kw.extend(_keywords(text, k=4))
    return MemoryEntry(
        id=state.idea_seed.id,
        title=state.idea_seed.title,
        modalities=[m.value for m in state.idea_seed.modalities],
        verdict=state.verdict.value if isinstance(state.verdict, Verdict) else str(state.verdict),
        fatal_flaws=[f.get("flaw", "") for f in (state.refute.fatal_flaws or [])] if state.refute else [],
        refuted_keywords=fatal_kw,
    )


class DirectorMemory:
    """Append-only memory of every idea + verdict we've touched."""

    def __init__(self, path: Path | str = "runs/director_memory.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[MemoryEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self.entries.append(MemoryEntry(**json.loads(line)))
            except Exception as e:
                log.debug("skip bad memory line: %s", e)

    def record(self, state) -> None:
        e = _entry_from_state(state)
        # Replace any prior entry for the same idea_id
        self.entries = [x for x in self.entries if x.id != e.id]
        self.entries.append(e)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # queries used by Director + IdeaMiner
    # ------------------------------------------------------------------

    def seen_titles(self) -> set[str]:
        return {e.title.strip().lower() for e in self.entries}

    def refuted_titles(self) -> set[str]:
        return {e.title.strip().lower() for e in self.entries if e.verdict == "refuted"}

    def refuted_keywords(self) -> Counter:
        c: Counter = Counter()
        for e in self.entries:
            if e.verdict == "refuted":
                for kw in e.refuted_keywords:
                    c[kw] += 1
        return c

    def refuted_modality_counts(self) -> Counter:
        c: Counter = Counter()
        for e in self.entries:
            if e.verdict == "refuted":
                for m in e.modalities:
                    c[m] += 1
        return c

    def summary(self) -> dict:
        return {
            "n_total": len(self.entries),
            "by_verdict": dict(Counter(e.verdict for e in self.entries)),
            "refuted_modalities": dict(self.refuted_modality_counts()),
            "refuted_top_keywords": dict(self.refuted_keywords().most_common(15)),
        }

    def is_duplicate(self, idea: IdeaSeed) -> bool:
        """Cheap duplicate check by title similarity."""
        key = idea.title.strip().lower()
        if not key:
            return False
        for t in self.seen_titles():
            # trivial fuzzy: substring or token overlap > 70 %
            if key == t or key in t or t in key:
                return True
        return False
