from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings,  ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

pdf_path=input("enter  your pdf path: ").strip().strip('"')
loader=PyPDFLoader(pdf_path)
documents=loader.load()
print("number of pages loaded:",len(documents))
splitter=RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks=splitter.split_documents(documents)
if not chunks:
    raise ValueError(
        "no text based text related found. Try text based on pdf"
    )
print("Number of chunks created:", len(chunks))
print("\nFirst chunk:\n", chunks[0].page_content[:300])

embeddings=OllamaEmbeddings(
    model='nomic-embed-text'
)
vector_store=FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
print("\nFAISS vector database created!")
retriever=vector_store.as_retriever(
    search_kwargs={"k":4}
)
llm=ChatOllama(
    model='qwen2.5:3b',
    temperature=0
)
prompt=ChatPromptTemplate.from_template("""
    you helpful to study ai assistance.
    Answer the question using only this content.
    If answer is not present in the content, say:
    "I could  not find as referal material, not an instructor.
    Context:
    {context}
    Question:
    {question}
    answer: 
"""

)
rag_chain=prompt|llm
while True:
    question=input(
        "\nAsk a question(type exit)"
    ).strip()
    if question.lower()=="exit":
        print("good bye")
        break
    if not question:
        continue
    relevant_docs=retriever.invoke(question)
    context_parts=[]
    for doc in relevant_docs:
        source=doc.metadata.get("source","unknown")
        page=doc.metadata.get("page",0)+1
        context_parts.append(
            f"Source:{source}\n"
            f"Page:{page}\n"
            f"Text:{doc.page_content}\n"

        )
    context="\n\n".join(context_parts)
    response=rag_chain.invoke({
        "context":context,
        "question":question
    })
    print("\nAnswer:\n")
    print(response.content)

    # Display retrieved sources
    print("\nRetrieved sources:")

    for doc in relevant_docs:
        print(
            f"- {doc.metadata.get('source', 'Unknown')}, "
            f"page {doc.metadata.get('page', 0) + 1}"
        )


