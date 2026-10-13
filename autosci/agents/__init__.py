"""AutoSCI — research-team sub-agents."""
from autosci.agents.idea_closer import IdeaCloserAgent
from autosci.agents.lit_surveyer import LitSurveyerAgent
from autosci.agents.method_designer import MethodDesignerAgent
from autosci.agents.exp_designer import ExpDesignerAgent
from autosci.agents.paper_writer import PaperWriterAgent
from autosci.agents.code_sketcher import CodeSketcherAgent
from autosci.agents.citation_grounder import CitationGrounderAgent
from autosci.agents.section_validator import SectionValidatorAgent
from autosci.agents.reviewer import ReviewerAgent
from autosci.agents.refuter import RefuterAgent

__all__ = [
    "IdeaCloserAgent",
    "LitSurveyerAgent",
    "MethodDesignerAgent",
    "ExpDesignerAgent",
    "PaperWriterAgent",
    "CodeSketcherAgent",
    "CitationGrounderAgent",
    "SectionValidatorAgent",
    "ReviewerAgent",
    "RefuterAgent",
]
