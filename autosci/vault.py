"""AutoSCI — Vault: where successful (or refuted) papers are archived.

Layout::

    vault/
        kept/<idea_id>.paper.md     # mirror of the canonical paper
        kept/<idea_id>.code/        # mirror of the code skeleton
        kept/<idea_id>.state.json   # full final state
        refuted/<idea_id>.paper.md
        refuted/<idea_id>.state.json
        refuted/<idea_id>.reason.md # why the refuter said no
        manifest.json               # index across runs
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from autosci.schemas import IdeaState, Verdict


log = logging.getLogger("autosci.vault")


class Vault:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.kept = self.root / "kept"
        self.refuted = self.root / "refuted"
        self.manifest_path = self.root / "manifest.json"

    def bootstrap(self) -> None:
        self.kept.mkdir(parents=True, exist_ok=True)
        self.refuted.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def record(self, state: IdeaState, paper_md_path: Path | None = None) -> Path:
        """Copy state + paper + code into the appropriate vault subdir.

        Returns the destination directory.
        """
        verdict = state.verdict
        bucket = self.kept if verdict in (Verdict.KEPT, Verdict.REVISED) else self.refuted
        dest = bucket / state.idea_seed.id
        dest.mkdir(parents=True, exist_ok=True)

        # state
        (dest / "state.json").write_text(
            state.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # paper.md
        if paper_md_path and Path(paper_md_path).exists():
            shutil.copy2(paper_md_path, dest / "paper.md")

        # code
        if state.code_paths:
            code_dir = dest / "code"
            code_dir.mkdir(exist_ok=True)
            for p in state.code_paths:
                src = Path(p)
                if src.exists():
                    target = code_dir / src.relative_to(src.parents[len(src.parts) - 3])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, target)
            if (Path(state.code_paths[0]).parent / "README.md").exists() and state.code_paths:
                # copy top-level README
                rd = Path(state.code_paths[0]).parent / "README.md"
                if rd.exists():
                    shutil.copy2(rd, code_dir / "README.md")

        if state.refute and verdict == Verdict.REFUTED:
            (dest / "reason.md").write_text(
                self._render_refute_reason(state), encoding="utf-8"
            )

        # Recipe — minimal reproducibility record (idea + model + token costs)
        try:
            (dest / "recipe.json").write_text(
                self._render_recipe(state), encoding="utf-8"
            )
        except Exception:
            pass

        self._update_manifest(state)
        return dest

    @staticmethod
    def _render_recipe(state: IdeaState) -> str:
        # last entry in history_log may carry token info (Director records it)
        token_info: dict[str, int] = {}
        for h in reversed(state.history_log):
            if h.startswith("__token__"):
                # parse: "__token__ in=123 out=456 cost_cny=0.012 calls=8"
                for part in h.replace("__token__", "").strip().split():
                    if "=" in part:
                        k, v = part.split("=", 1)
                        try:
                            token_info[k] = float(v) if "." in v else int(v)
                        except ValueError:
                            pass
                break
        recipe = {
            "idea_id": state.idea_seed.id,
            "idea_title": state.idea_seed.title,
            "one_liner": state.idea_seed.one_liner,
            "modalities": [m.value for m in state.idea_seed.modalities],
            "verdict": state.verdict.value if isinstance(state.verdict, Verdict) else str(state.verdict),
            "model": __import__("os").environ.get("AGNES_MODEL", "agnes-3.0-flash"),
            "base_url": __import__("os").environ.get("AGNES_BASE_URL", "https://api.agnes-ai.cn/v1"),
            "kb_corpus_dir": str(Path(__import__("os").environ.get("AUTOSCI_PAPER_ROOT", "")) / "_digest")
                if __import__("os").environ.get("AUTOSCI_PAPER_ROOT") else None,
            "tokens": token_info,
            "sections": list((state.paper_draft.sections or {}).keys()) if state.paper_draft else [],
            "n_references": len(state.paper_draft.references or []) if state.paper_draft else 0,
        }
        return json.dumps(recipe, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------

    def _update_manifest(self, state: IdeaState) -> None:
        if self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = {"items": []}
        else:
            manifest = {"items": []}

        items = [it for it in manifest["items"] if it.get("id") != state.idea_seed.id]
        items.append(
            {
                "id": state.idea_seed.id,
                "title": state.idea_seed.title,
                "verdict": state.verdict.value if isinstance(state.verdict, Verdict) else str(state.verdict),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "modality": [m.value for m in state.idea_seed.modalities],
            }
        )
        manifest["items"] = items
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _render_refute_reason(state: IdeaState) -> str:
        r = state.refute
        if not r:
            return "# No reason recorded.\n"
        out = [f"# Refute reason for {state.idea_seed.id}", ""]
        out.append(f"**Verdict:** {r.verdict}\n")
        if r.fatal_flaws:
            out.append("## Fatal flaws")
            for f in r.fatal_flaws:
                out.append(f"- **{f.get('flaw','?')}**\n  - Evidence: {f.get('evidence','?')}\n  - Fix: {f.get('fix','?')}")
            out.append("")
        if r.revise_suggestions:
            out.append("## Revise suggestions")
            for s in r.revise_suggestions:
                out.append(f"- {s}")
            out.append("")
        if r.notes:
            out.append("## Notes")
            out.append(r.notes)
            out.append("")
        return "\n".join(out)
