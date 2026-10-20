"""Local TF-IDF based RAG store for injecting relevant past solutions into agent prompts."""
import json
import hashlib
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class RAGDocument:
    id: str
    content: str
    tags: List[str]
    source: str  # "solution", "code_pattern", "decision", "architecture"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class RAGStore:
    """
    Lightweight local RAG store using TF-IDF cosine similarity.
    No GPU required. ~20-30MB RAM overhead for 500 documents.
    Falls back gracefully if scikit-learn not installed.
    """

    def __init__(self, persist_dir: str = "./output/rag", top_k: int = 3):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.documents: List[RAGDocument] = []
        self._vectorizer = None
        self._matrix = None
        self._sklearn_available = self._check_sklearn()
        self._dirty = True  # needs refit before next retrieve()
        self._load()

    def _check_sklearn(self) -> bool:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa
            return True
        except ImportError:
            return False

    def _load(self):
        index_path = self.persist_dir / "rag_index.json"
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text())
                self.documents = [RAGDocument(**d) for d in data]
                self._refit()
                self._dirty = False
            except Exception:
                self.documents = []

    def _save(self):
        index_path = self.persist_dir / "rag_index.json"
        index_path.write_text(json.dumps([asdict(d) for d in self.documents], indent=2))

    def _refit(self):
        if not self._sklearn_available or len(self.documents) < 2:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
        corpus = [d.content for d in self.documents]
        try:
            self._matrix = self._vectorizer.fit_transform(corpus)
        except ValueError:
            # Empty vocabulary — e.g. every document is stop words only.
            # Leave matrix unset so retrieve() falls back to keyword overlap
            # instead of raising on every call.
            self._vectorizer = None
            self._matrix = None

    def add(self, content: str, tags: List[str], source: str) -> str:
        # Hash what is actually stored: ids must match the truncated content,
        # otherwise the id cannot be reproduced from the stored document.
        stored_content = content[:3000]
        doc_id = hashlib.md5(stored_content.encode()).hexdigest()[:12]
        # Avoid duplicates
        if any(d.id == doc_id for d in self.documents):
            return doc_id
        doc = RAGDocument(id=doc_id, content=stored_content, tags=tags, source=source)
        self.documents.append(doc)
        self._dirty = True  # refit lazily before next retrieve()
        self._save()
        return doc_id

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[RAGDocument, float]]:
        k = top_k or self.top_k
        if not self.documents:
            return []
        if self._dirty:
            self._refit()
            self._dirty = False
        if not self._sklearn_available or self._matrix is None or len(self.documents) < 2:
            # Fallback: simple keyword overlap
            results = []
            query_words = set(query.lower().split())
            for doc in self.documents:
                doc_words = set(doc.content.lower().split())
                score = len(query_words & doc_words) / max(len(query_words), 1)
                results.append((doc, score))
            results.sort(key=lambda x: x[1], reverse=True)
            return [(d, s) for d, s in results[:k] if s > 0]

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self.documents[i], float(scores[i])) for i in top_indices if scores[i] > 0.01]

    def format_for_prompt(self, query: str) -> str:
        results = self.retrieve(query)
        if not results:
            return ""
        lines = ["[RELEVANT PAST SOLUTIONS & PATTERNS]"]
        for doc, score in results:
            if score < 0.05:
                continue
            lines.append(f"[{doc.source.upper()} | relevance: {score:.2f}]")
            lines.append(doc.content[:500])
            lines.append("---")
        return "\n".join(lines) if len(lines) > 1 else ""
