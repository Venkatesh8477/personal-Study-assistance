# Personal Study Assistant

A local Streamlit app for asking questions about study PDFs. It uses Ollama for chat and embeddings, FAISS for vector search, and a CrossEncoder to rerank candidate passages. No paid cloud model is used.

## Requirements and setup (Windows)

Install Python 3.12 or newer and [Ollama](https://ollama.com/download/windows). Open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Start Ollama, then run:

```powershell
streamlit run app.py
```

The CrossEncoder is downloaded by Sentence Transformers the first time it is needed. That first run needs access to the model source. Ollama must remain running with both named models available.

The legacy root-level `faiss_index/` is left untouched and is not auto-imported because it has no source-document manifest. Upload the source PDFs into the app-managed library to create a verified index there.

## Using the app

Add one or more text-based PDFs in the sidebar, then process them. Each file is limited to 50 MB. Exact duplicate files are skipped. The sidebar lists saved PDFs, lets you preview or download them, remove a PDF and rebuild the index, and select which PDFs should inform chat. Removing or adding a document clears chat state so old answers do not carry into a different notes set. App-managed uploads and indexes are stored in `study_assistant_data/`; the existing project `data` file is left untouched.

Ask a study question, request a quiz (optionally specify a number and difficulty), or request a revision summary. Answers show the retrieved passages, filename/page citations, a raw CrossEncoder heuristic score, an LLM faithfulness estimate, and timing. Clear Chat resets the current conversation.

## Evaluation

Edit `evaluation_cases.json` to add cases. Each case has a `subject`, `question`, and optional `expected_facts` list. Run the RAG Evaluation Dashboard in the app to see retrieved passages, generated answers, citations, heuristic relevance, faithfulness verdicts, case results, and CSV export.

Retrieval pass means every configured expected fact occurs literally in the returned passages; this is a simple keyword check, not semantic grading. Faithfulness is judged by the same local chat model and can be wrong. Relevance is a raw CrossEncoder score, not a probability or answer-accuracy measurement. The app reports only results from cases actually run.

Observed live smoke run on 13 October 2026 using `DBMS_Notes (2).pdf`, `OS_Full_Notes.pdf`, `qwen2.5:3b`, and `nomic-embed-text`:

| Case | Faithfulness | Shown behavior | Total time |
| --- | --- | --- | ---: |
| Primary-key definition | SUPPORTED | Answer shown with PDF/page citations | 34.16 s |
| What an OS does | PARTIALLY SUPPORTED | Draft withheld; asks for a narrower question | 48.50 s |
| Capital of France | UNSUPPORTED | Draft withheld; says evidence does not support an answer | 19.42 s |

These are three observations, not a benchmark or a dashboard aggregate. A separate initial UI render completed with zero Streamlit exceptions. App-level upload/removal interactions and chat-history controls were not exercised interactively in this run.

Run the model-independent checks with:

```powershell
python -m unittest discover -s tests -v
```

The PDF/FAISS/Ollama end-to-end checks require the models and a running local Ollama server. Do not treat the helper tests as proof that generation is accurate.

With the included sample PDFs and both Ollama models available, run the optional live smoke check with:

```powershell
python tests\local_rag_smoke.py
```

It asks one DBMS question, one operating systems question, and one unrelated question using two PDFs. Its printed answers and judgments are observations from that run, not a quality guarantee.

## Architecture

1. Uploaded bytes are validated, content-hashed, and stored under `study_assistant_data/documents/` with a generated filename. The original display filename is kept as metadata.
2. `PyPDFLoader` reads pages; the splitter creates 800-character chunks with 100-character overlap while retaining source and page metadata.
3. `nomic-embed-text` embeds chunks and a local FAISS index is stored in `study_assistant_data/faiss_index/`.
4. Each question retrieves up to 12 candidates from the selected documents and the CrossEncoder reranks them once. The same passages are returned by the `search_notes` agent tool and used for citations and answer-faithfulness review.
5. `qwen2.5:3b` writes the answer or study material from those passages. The agent has `search_notes`, `generate_quiz`, and `revision_summary` tools.

FAISS LangChain persistence uses a pickle file. This app loads only the index under its app-managed local storage directory after checking the document set and saved index file hashes. Never copy an index from an untrusted source into that directory; local pickle deserialization can execute code. If the manifest or index is missing or does not match, the app rebuilds it from the managed PDFs.

## Deployment and limitations

This setup is intended for one person running the app locally. The document directory and index are shared by the process, so do not expose the app as a multi-user service. A cloud deployment needs a reachable Ollama server, persistent private storage, and an explicit access-control design; a typical ephemeral Streamlit host does not provide those automatically. No cloud deployment is claimed or tested here.

Scanned/image-only PDFs need OCR before their contents can be searched. Reranker scores are not calibrated confidence, and LLM faithfulness checks are fallible. Evaluation quality depends on the cases you write. Large PDF collections can use significant memory and indexing time.

## Troubleshooting

- **Ollama connection/model error:** Start Ollama and run `ollama list`; pull the two model tags above if missing.
- **No text found:** The PDF may be scanned or image-only. OCR it before upload.
- **CrossEncoder download failure:** Check network access on the first run and retry after the model is cached.
- **Index rebuild warning:** The local index may be incomplete or stale; the app rebuilds it from managed PDFs. If source PDFs are missing, add them again.
- **PowerShell activation blocked:** Run the project’s `.venv\Scripts\python.exe -m streamlit run app.py` directly, or use an approved local PowerShell execution policy.

## Interview notes

The project demonstrates a local retrieval-augmented generation flow: chunking preserves page metadata, embeddings support semantic candidate search, FAISS stores vectors, and a CrossEncoder reranks candidates. An agent exposes note search, quizzes, and summaries as tools. Sources are displayed from retrieved metadata. Evaluation combines literal expected-fact retrieval checks with an LLM faithfulness estimate; neither is a complete quality benchmark. Limitations include scanned PDFs without OCR, single-user local storage, model-dependent output, and uncalibrated relevance/faithfulness judgments.
