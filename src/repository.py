"""
Knowledge Repository Module
===========================
Unified module for rubric parsing, vector store ingestion, and hybrid retrieval.
Provides seamless fallback between persistent ChromaDB and in-memory parsing.
"""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

from .config import CHROMA_PATH, RUBRIC_PATH

logger = logging.getLogger(__name__)


def parse_rubric_markdown(path: str = RUBRIC_PATH) -> List[Dict[str, Any]]:
    """
    Canonical parser for ml_rubric.md.
    Splits YAML blocks separated by '---' and returns clean rule dictionaries.
    """
    if not os.path.exists(path):
        logger.warning(f"Rubric file not found at {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    rules = []
    seen = set()
    for block in content.split("---"):
        block = block.strip()
        if not block or re.match(r"^-+$", block):
            continue
        if block.startswith("-"):
            block = re.sub(r"^-+\s*", "", block)
        if not block:
            continue
        try:
            rule = yaml.safe_load(block)
            if (
                rule
                and isinstance(rule, dict)
                and "rule_id" in rule
                and rule["rule_id"] not in seen
            ):
                rules.append(rule)
                seen.add(rule["rule_id"])
        except Exception:
            continue

    return rules


class KnowledgeRepository:
    """
    Deep Module for all rubric and lecture retrieval operations.
    Queries ChromaDB when available; falls back to in-memory markdown rules.
    """

    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        rubric_path: str = RUBRIC_PATH,
    ):
        self.chroma_path = chroma_path
        self.rubric_path = rubric_path
        self._client = None
        self._rubric_col = None
        self._lecture_col = None
        self._in_memory_rules: Optional[List[Dict[str, Any]]] = None

    def _ensure_chroma(self):
        """Lazy connection to ChromaDB collections."""
        if self._client is None and os.path.exists(self.chroma_path):
            try:
                import chromadb
                self._client = chromadb.PersistentClient(path=self.chroma_path)
                try:
                    self._rubric_col = self._client.get_collection("rubric_rules")
                except Exception:
                    self._rubric_col = None
                try:
                    self._lecture_col = self._client.get_collection("lecture_knowledge")
                except Exception:
                    self._lecture_col = None
            except Exception as e:
                logger.warning(f"ChromaDB connection unavailable at {self.chroma_path}: {e}")

    def retrieve_rubric(
        self,
        topic: str,
        error_text: str = "",
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: Searches ChromaDB with topic and semantic error text,
        re-ranking exact topic matches first. Falls back to in-memory rubric.
        """
        self._ensure_chroma()

        if self._rubric_col is not None:
            try:
                query = f"Topic: {topic}. Error: {error_text}"
                results = self._rubric_col.query(
                    query_texts=[query],
                    n_results=min(15, k * 3),
                )
                if results and results.get("ids") and results["ids"][0]:
                    all_results = []
                    for idx, doc_id in enumerate(results["ids"][0]):
                        meta = results["metadatas"][0][idx]
                        all_results.append({
                            "rule_id": meta.get("rule_id", doc_id),
                            "topic": meta.get("topic", topic),
                            "subtopic": meta.get("subtopic", ""),
                            "criteria": results["documents"][0][idx],
                            "socratic_hint": meta.get("socratic_hint", ""),
                            "common_error": meta.get("common_error", ""),
                            "points": meta.get("points", "1"),
                            "distance": results["distances"][0][idx] if "distances" in results else 0.0,
                        })

                    # Priority 1: Exact topic match
                    exact = [r for r in all_results if r["topic"].lower() == topic.lower()]
                    others = [r for r in all_results if r not in exact]
                    return (exact + others)[:k]
            except Exception as e:
                logger.warning(f"Chroma query failed, falling back to memory: {e}")

        # In-memory fallback (shortcut: pure topic match without vector distances)
        if self._in_memory_rules is None:
            self._in_memory_rules = parse_rubric_markdown(self.rubric_path)

        exact = [r for r in self._in_memory_rules if r.get("topic", "").lower() == topic.lower()]
        return exact[:k] if exact else self._in_memory_rules[:k]

    def retrieve_context(self, topic: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves lecture slide context chunks for a topic."""
        self._ensure_chroma()

        if self._lecture_col is not None:
            try:
                results = self._lecture_col.query(query_texts=[topic], n_results=min(10, k * 3))
                if results and results.get("ids") and results["ids"][0]:
                    chunks = []
                    for idx in range(len(results["ids"][0])):
                        chunks.append({
                            "text": results["documents"][0][idx],
                            "metadata": results["metadatas"][0][idx],
                        })
                    exact = [c for c in chunks if c["metadata"].get("topic", "").lower() == topic.lower()]
                    others = [c for c in chunks if c not in exact]
                    return (exact + others)[:k]
            except Exception as e:
                logger.warning(f"Chroma lecture context query failed: {e}")

        return []

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on active collections and rules."""
        self._ensure_chroma()
        return {
            "rubric_rules_count": self._rubric_col.count() if self._rubric_col else len(self.retrieve_rubric("")),
            "lecture_chunks_count": self._lecture_col.count() if self._lecture_col else 0,
            "backend": "ChromaDB" if self._rubric_col else "In-Memory Markdown",
        }


def build_vector_database(
    chroma_path: str = CHROMA_PATH,
    rubric_path: str = RUBRIC_PATH,
    pdf_dir: Optional[str] = None,
):
    """
    Builds the persistent ChromaDB collections from rubric markdown and lecture PDFs.
    Run once during environment setup.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    os.makedirs(chroma_path, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path)
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # 1. Ingest Rubric Rules
    rules = parse_rubric_markdown(rubric_path)
    try:
        client.delete_collection("rubric_rules")
    except Exception:
        pass
    rubric_col = client.create_collection("rubric_rules", metadata={"hnsw:space": "cosine"})

    docs, embs, metas, ids = [], [], [], []
    for i, r in enumerate(rules):
        txt = f"{r.get('topic','')} {r.get('subtopic','')} {r.get('keywords','')} {r.get('criteria','')}"
        docs.append(r.get("criteria", ""))
        embs.append(embedder.encode(txt, convert_to_tensor=False).tolist())
        metas.append({
            k: str(r.get(k, ""))
            for k in ["rule_id", "topic", "subtopic", "points", "common_error", "socratic_hint", "keywords"]
        })
        ids.append(str(r.get("rule_id", f"rule_{i}")))

    if docs:
        rubric_col.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)
    logger.info(f"✓ Ingested {rubric_col.count()} rubric rules into ChromaDB")

    # 2. Ingest Lecture PDFs (if directory provided)
    if pdf_dir and os.path.exists(pdf_dir):
        import fitz
        try:
            client.delete_collection("lecture_knowledge")
        except Exception:
            pass
        lecture_col = client.create_collection("lecture_knowledge", metadata={"hnsw:space": "cosine"})

        all_chunks = []
        for pdf_file in sorted(Path(pdf_dir).glob("*.pdf")):
            try:
                doc = fitz.open(str(pdf_file))
                topic = pdf_file.stem
                for pn, page in enumerate(doc):
                    text = page.get_text("text")
                    annots = [a.info.get("content", "") for a in (page.annots() or []) if a.info.get("content")]
                    full = (text + "\n" + "\n".join(annots)).strip()
                    if len(full) > 50:
                        all_chunks.append({
                            "text": full[:2000],
                            "meta": {"topic": topic, "page": str(pn + 1), "source": pdf_file.name},
                        })
                doc.close()
            except Exception as e:
                logger.warning(f"Skipping {pdf_file.name}: {e}")

        BATCH = 64
        for i in range(0, len(all_chunks), BATCH):
            batch = all_chunks[i : i + BATCH]
            texts = [c["text"] for c in batch]
            batch_embs = embedder.encode(texts, batch_size=BATCH, convert_to_tensor=False).tolist()
            lecture_col.add(
                documents=texts,
                embeddings=batch_embs,
                metadatas=[c["meta"] for c in batch],
                ids=[f"chunk_{i + j}" for j in range(len(batch))],
            )
        logger.info(f"✓ Ingested {lecture_col.count()} lecture chunks into ChromaDB")
