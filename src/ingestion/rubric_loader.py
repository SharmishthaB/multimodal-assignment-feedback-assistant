"""
Rubric Loading Module (Backward-Compatibility Layer)
Redirects to the consolidated KnowledgeRepository in src.repository.
"""

from src.repository import parse_rubric_markdown, build_vector_database as load_rubric_to_chroma

__all__ = ["parse_rubric_markdown", "load_rubric_to_chroma"]
