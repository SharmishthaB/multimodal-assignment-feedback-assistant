"""
Retriever Module (Backward-Compatibility Layer)
Redirects to the consolidated KnowledgeRepository in src.repository.
"""

from src.repository import KnowledgeRepository, KnowledgeRepository as ChromaRetriever, KnowledgeRepository as RAGSystem

__all__ = ["KnowledgeRepository", "ChromaRetriever", "RAGSystem"]
