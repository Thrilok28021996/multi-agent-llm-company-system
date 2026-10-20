"""Tests for the RAG store — lazy rebuild, deduplication, retrieval."""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from memory.rag_store import RAGStore


@pytest.fixture
def rag_store(tmp_path):
    """Create a temporary RAG store for testing."""
    return RAGStore(persist_dir=str(tmp_path / "rag"), top_k=3)


class TestRAGStoreAdd:
    def test_add_returns_id(self, rag_store):
        doc_id = rag_store.add("test content", ["tag1"], "test")
        assert isinstance(doc_id, str)
        assert len(doc_id) == 12

    def test_add_sets_dirty_flag(self, rag_store):
        """After add(), _dirty should be True until retrieve() is called."""
        assert rag_store._dirty is True  # initially dirty from empty load
        rag_store.add("content A", ["a"], "test")
        assert rag_store._dirty is True

    def test_add_deduplicates(self, rag_store):
        """Adding the same content twice should not create duplicates."""
        id1 = rag_store.add("duplicate content", ["tag"], "test")
        id2 = rag_store.add("duplicate content", ["tag"], "test")
        assert id1 == id2
        assert len(rag_store.documents) == 1

    def test_add_truncates_long_content(self, rag_store):
        """Content longer than 3000 chars should be truncated."""
        long_content = "x" * 5000
        rag_store.add(long_content, ["tag"], "test")
        assert len(rag_store.documents[0].content) == 3000

    def test_id_matches_stored_content(self, rag_store):
        """IDs must be derived from what is actually stored (truncated)."""
        long_content = "y" * 5000
        doc_id = rag_store.add(long_content, ["tag"], "test")
        stored = rag_store.documents[0].content
        assert len(stored) == 3000
        assert doc_id == hashlib.md5(stored.encode()).hexdigest()[:12]

    def test_short_content_id_matches_stored_content(self, rag_store):
        doc_id = rag_store.add("short content", ["tag"], "test")
        assert doc_id == hashlib.md5(b"short content").hexdigest()[:12]


class TestRAGStoreRetrieve:
    def test_retrieve_empty_store(self, rag_store):
        results = rag_store.retrieve("anything")
        assert results == []

    def test_retrieve_after_add(self, rag_store):
        rag_store.add("python programming language", ["python"], "solution")
        rag_store.add("javascript web development", ["js"], "solution")
        results = rag_store.retrieve("python")
        assert len(results) > 0
        assert results[0][0].content == "python programming language"

    def test_retrieve_clears_dirty_flag(self, rag_store):
        """retrieve() should refit if dirty and clear the flag."""
        rag_store.add("content A", ["a"], "test")
        rag_store.add("content B", ["b"], "test")
        assert rag_store._dirty is True
        rag_store.retrieve("query")
        assert rag_store._dirty is False

    def test_retrieve_keyword_fallback(self, rag_store):
        """Without sklearn, retrieve should use keyword overlap fallback."""
        # Force sklearn unavailable
        rag_store._sklearn_available = False
        rag_store._matrix = None
        rag_store.add("python is great for data science", ["python"], "test")
        results = rag_store.retrieve("python data")
        assert len(results) > 0

    def test_retrieve_top_k(self, rag_store):
        for i in range(10):
            rag_store.add(f"document {i} about topic {i % 3}", ["t"], "test")
        results = rag_store.retrieve("topic", top_k=2)
        assert len(results) <= 2

    def test_retrieve_all_stop_word_corpus_does_not_raise(self, rag_store):
        """TF-IDF raises 'empty vocabulary' on stop-word-only corpora — the
        store must handle that instead of propagating the error."""
        rag_store.add("the and of is it to for with on at by", ["a"], "test")
        rag_store.add("in as at by from into over under", ["b"], "test")

        results = rag_store.retrieve("the and of is it to")

        assert isinstance(results, list)

    def test_all_stop_word_refit_leaves_store_retrievable(self, rag_store):
        rag_store.add("the and of is it to", ["a"], "test")
        rag_store.add("for with on at by", ["b"], "test")

        rag_store._refit()  # must not raise

        assert rag_store._matrix is None  # keyword-overlap fallback is used
        assert isinstance(rag_store.retrieve("the and"), list)


class TestRAGStorePersistence:
    def test_save_and_reload(self, tmp_path):
        store_dir = tmp_path / "rag"
        store = RAGStore(persist_dir=str(store_dir))
        store.add("persistent content", ["persist"], "test")
        store._save()

        # Reload from disk
        store2 = RAGStore(persist_dir=str(store_dir))
        assert len(store2.documents) == 1
        assert store2.documents[0].content == "persistent content"

    def test_reload_of_stop_word_corpus_keeps_documents(self, tmp_path):
        """A failed TF-IDF fit during load must not wipe the index."""
        store_dir = tmp_path / "rag"
        store = RAGStore(persist_dir=str(store_dir))
        store.add("the and of is it to", ["a"], "test")
        store.add("for with on at by", ["b"], "test")
        store._save()

        reloaded = RAGStore(persist_dir=str(store_dir))

        assert len(reloaded.documents) == 2
        assert isinstance(reloaded.retrieve("the and of"), list)


class TestRAGStoreFormatForPrompt:
    def test_format_empty(self, rag_store):
        result = rag_store.format_for_prompt("anything")
        assert result == ""

    def test_format_with_results(self, rag_store):
        rag_store.add("fastapi is a modern python web framework", ["web"], "solution")
        result = rag_store.format_for_prompt("web framework python")
        assert "RELEVANT PAST SOLUTIONS" in result
        assert "fastapi" in result.lower()
