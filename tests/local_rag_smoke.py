"""Optional Ollama-backed smoke check; run directly, not in offline unittest discovery."""

from __future__ import annotations

import hashlib
import argparse
from pathlib import Path
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["primary_key", "operating_system", "unrelated", "all"], default="all")
    args = parser.parse_args()
    root = ROOT
    sources = [root / "DBMS_Notes (2).pdf", root / "OS_Full_Notes.pdf"]
    missing = [str(path.name) for path in sources if not path.is_file()]
    if missing:
        raise SystemExit(f"Required sample PDFs are missing: {', '.join(missing)}")

    embeddings, llm, reranker = app.load_models()
    pages = []
    ids = []
    for path in sources:
        document_id = hashlib.sha256(path.read_bytes()).hexdigest()
        ids.append(document_id)
        pages_for_file = PyPDFLoader(str(path)).load()
        keyword = "primary key" if "DBMS" in path.name else "operating system"
        relevant_pages = [page for page in pages_for_file if keyword in page.page_content.casefold()][:3]
        if not relevant_pages:
            relevant_pages = [page for page in pages_for_file if page.page_content.strip()][:2]
        for page in relevant_pages:
            page.metadata["source"] = path.name
            page.metadata["source_id"] = document_id
            pages.append(page)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=app.CHUNK_SIZE, chunk_overlap=app.CHUNK_OVERLAP
    ).split_documents(pages)
    if not chunks:
        raise SystemExit("Sample PDFs did not produce searchable text.")
    store = FAISS.from_documents(chunks, embeddings)

    cases = {
        "primary_key": "What is a primary key in a database?",
        "operating_system": "What does an operating system do?",
        "unrelated": "What is the capital of France?",
    }
    selected_cases = cases.items() if args.case == "all" else [(args.case, cases[args.case])]
    for case, question in selected_cases:
        result = app.run_study_turn(question, store, ids, reranker, llm)
        print(f"\nQUESTION: {question}")
        print(f"SHOWN ANSWER: {result['answer']}")
        print(f"GENERATED DRAFT: {result.get('generated_answer', result['answer'])}")
        print(f"CITATIONS: {result['citations']}")
        print(f"FAITHFULNESS: {result['faithfulness']}")
        print(f"TIMING: retrieval={result['retrieval_seconds']:.2f}s rerank={result['rerank_seconds']:.2f}s total={result['total_seconds']:.2f}s")
        if case == "unrelated" and "paris" in result["answer"].casefold():
            raise AssertionError("An unsupported geographic answer passed the faithfulness gate.")


if __name__ == "__main__":
    main()
