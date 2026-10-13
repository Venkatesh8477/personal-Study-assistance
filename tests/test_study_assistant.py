import unittest
import hashlib
import tempfile
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings

from study_assistant import (
    build_faithfulness_prompt,
    extract_json_object,
    rank_documents,
    relevance_score,
    retrieve_and_rerank,
    validate_pdf_bytes,
)


class Document:
    def __init__(self, text):
        self.page_content = text


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return self.scores[: len(pairs)]


class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs
        self.last_filter = None

    def similarity_search(self, question, k, filter=None):
        self.last_filter = filter
        return self.docs[:k]


class TinyEmbeddings(Embeddings):
    """Deterministic local embeddings for FAISS persistence tests."""
    def _embed(self, text):
        vector = [0.0] * 64
        for token in text.casefold().split():
            index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:2], "big") % len(vector)
            vector[index] += 1.0
        return vector

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


class StudyAssistantHelperTests(unittest.TestCase):
    def test_pdf_validation(self):
        validate_pdf_bytes(b"%PDF-1.7\nbody", "notes.pdf", max_bytes=20)
        for data, name, limit in [
            (b"", "notes.pdf", 10),
            (b"not a pdf", "notes.pdf", 20),
            (b"%PDF-1.7\nlarge", "notes.pdf", 5),
            (b"%PDF-1.7", "notes.txt", 20),
        ]:
            with self.subTest(data=data, name=name), self.assertRaises(ValueError):
                validate_pdf_bytes(data, name, limit)

    def test_reranking_orders_passages_and_runs_once(self):
        docs = [Document("weak"), Document("strong"), Document("middle")]
        reranker = FakeReranker([-0.2, 1.3, 0.4])
        ranked, score = rank_documents("question", docs, reranker, top_n=2)
        self.assertEqual([doc.page_content for doc in ranked], ["strong", "middle"])
        self.assertEqual(score, 1.3)
        self.assertEqual(reranker.calls, 1)
        self.assertEqual(relevance_score(score), 1.3)
        self.assertIsNone(relevance_score(None))

    def test_retrieval_passes_document_filter_and_reranks_once(self):
        docs = [Document("one"), Document("two")]
        store = FakeVectorStore(docs)
        reranker = FakeReranker([0.2, 0.8])
        ranked, score, _, _ = retrieve_and_rerank(store, "q", ["doc-a"], reranker)
        self.assertEqual(store.last_filter, {"source_id": {"$in": ["doc-a"]}})
        self.assertEqual(ranked[0].page_content, "two")
        self.assertEqual(score, 0.8)
        self.assertEqual(reranker.calls, 1)

    def test_faithfulness_json_parsing_and_prompt_scope(self):
        parsed = extract_json_object('```json\n{"verdict":"SUPPORTED","explanation":"Matches."}\n```')
        self.assertEqual(parsed["verdict"], "SUPPORTED")
        prompt = build_faithfulness_prompt("answer", "actual retrieved passage")
        self.assertIn("answer", prompt)
        self.assertIn("actual retrieved passage", prompt)
        self.assertIn("untrusted quoted evidence", prompt)
        with self.assertRaises(ValueError):
            extract_json_object("not json")

    def test_faiss_metadata_filter_save_reload_and_rebuild(self):
        embeddings = TinyEmbeddings()
        first = LCDocument(
            page_content="A primary key uniquely identifies a row in a database table.",
            metadata={"source": "dbms-a.pdf", "page": 0, "source_id": "a"},
        )
        overlapping = LCDocument(
            page_content="A primary key is unique and a foreign key references another table.",
            metadata={"source": "dbms-b.pdf", "page": 3, "source_id": "b"},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "index"
            store = FAISS.from_documents([first, overlapping], embeddings)
            store.save_local(str(index_path))
            reloaded = FAISS.load_local(
                str(index_path), embeddings, allow_dangerous_deserialization=True
            )
            matches = reloaded.similarity_search(
                "primary key", k=5, filter={"source_id": {"$in": ["b"]}}
            )
            self.assertTrue(matches)
            self.assertTrue(all(doc.metadata["source_id"] == "b" for doc in matches))
            self.assertEqual(matches[0].metadata["source"], "dbms-b.pdf")
            self.assertEqual(matches[0].metadata["page"], 3)

            # Rebuilding from the remaining PDF removes the removed source.
            rebuilt = FAISS.from_documents([first], embeddings)
            rebuilt_matches = rebuilt.similarity_search("primary key", k=5)
            self.assertTrue(all(doc.metadata["source_id"] == "a" for doc in rebuilt_matches))


if __name__ == "__main__":
    unittest.main()
