"""AutoSCI — GitHubIngestor.

Pulls READMEs (and optionally selected file trees) from a list of GitHub
repos that AutoSCI cares about. The result is a folder of synthetic
``*.digest.md`` files that look identical to the human-authored digests
in ``_digest``, so the existing KB indexer can ingest them.

We deliberately avoid downloading full source trees: README + LICENSE +
the top-level directory listing give us enough to ground new ideas.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx


log = logging.getLogger("autosci.github")


@dataclass
class RepoSpec:
    """Which GitHub repo to fetch. Either `owner/repo` or a full URL."""
    repo: str           # e.g. "Genoscope/MGnify" or "https://github.com/..."
    section_label: str  # e.g. "开头/摘要"
    branch: str = "HEAD"

    @property
    def owner_repo(self) -> str:
        r = self.repo.strip().rstrip("/")
        if r.startswith("http"):
            m = re.search(r"github\.com/([^/]+/[^/]+)", r)
            if m:
                return m.group(1).rstrip(".git")
        return r


# Default repos list is intentionally empty: the agent doesn't guess.
# Users populate it via autopilot config or pass their own list.
DEFAULT_REPOS: list[RepoSpec] = []


@dataclass
class FetchedRepo:
    owner_repo: str
    readme: str
    license_text: Optional[str] = None
    description: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return self.owner_repo.replace("/", "_")


def _raw_url(owner_repo: str, branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{path}"


def fetch_repo(spec: RepoSpec, client: Optional[httpx.Client] = None, timeout: float = 10) -> Optional[FetchedRepo]:
    """Best-effort fetch of a single repo's README + metadata.

    Returns None on any failure (404 / timeout / parse error).
    """
    client = client or httpx.Client(timeout=timeout, trust_env=False)
    owner_repo = spec.owner_repo

    # README candidates (we try main → master → HEAD)
    branches_to_try = [b for b in (spec.branch, "main", "master", "HEAD") if b]
    readme_text: Optional[str] = None
    for branch in branches_to_try:
        for candidate in ("README.md", "README.rst", "Readme.md", "readme.md"):
            try:
                r = client.get(_raw_url(owner_repo, branch, candidate))
            except httpx.HTTPError as e:
                log.debug("network fail for %s@%s/%s: %s", owner_repo, branch, candidate, e)
                continue
            if r.status_code == 200 and r.text.strip():
                readme_text = r.text
                break
        if readme_text:
            break

    if not readme_text:
        log.debug("Could not fetch README for %s", owner_repo)
        return None

    # Optional: LICENSE
    license_text: Optional[str] = None
    for branch in branches_to_try:
        for lic in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
            try:
                r = client.get(_raw_url(owner_repo, branch, lic))
            except httpx.HTTPError:
                continue
            if r.status_code == 200 and r.text.strip():
                license_text = r.text[:2000]
                break
        if license_text:
            break

    # Optional: GitHub repo metadata
    description: Optional[str] = None
    try:
        meta = client.get(f"https://api.github.com/repos/{owner_repo}")
        if meta.status_code == 200:
            j = meta.json()
            description = j.get("description")
            spec.section_label = "开头/摘要"
    except httpx.HTTPError:
        pass

    return FetchedRepo(
        owner_repo=owner_repo,
        readme=readme_text[:16000],
        license_text=license_text,
        description=description,
    )


def ingest_repos(
    out_dir: Path | str,
    repos: list[RepoSpec] | None = None,
    client: Optional[httpx.Client] = None,
    max_total_seconds: float = 60.0,
) -> list[Path]:
    """Fetch and write synthetic digests. Returns the list of files produced.

    Bounded by ``max_total_seconds`` overall so a slow / blocked GitHub
    doesn't pin the autopilot for hours.
    """
    import time
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repos = repos or DEFAULT_REPOS
    written: list[Path] = []
    t0 = time.time()
    for spec in repos:
        if time.time() - t0 > max_total_seconds:
            log.info("KB ingest budget exhausted (%ds); stopping.", int(max_total_seconds))
            break
        log.info("fetching %s ...", spec.owner_repo)
        fetched = fetch_repo(spec, client=client)
        if not fetched:
            continue
        target = out_dir / f"github_{fetched.slug}.digest.md"
        body = format_repo_as_digest(fetched, spec)
        target.write_text(body, encoding="utf-8")
        written.append(target)
        log.info("  -> %s (%d chars)", target, len(body))
    return written


def format_repo_as_digest(repo: FetchedRepo, spec: RepoSpec) -> str:
    """Format a fetched repo as an AutoSCI-flavoured digest."""
    parts = [
        f"# github:{repo.owner_repo}",
        "",
        f"## [{spec.section_label}]",
        "",
    ]
    if repo.description:
        parts.append(f"_(GitHub metadata)_\n\n{repo.description}\n")
    parts.append("### README")
    parts.append("")
    parts.append(repo.readme.strip())
    if repo.license_text:
        parts += ["", "### LICENSE (excerpt)", "", repo.license_text.strip()[:1500]]
    return "\n".join(parts) + "\n"
