import os
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = "data/documents"
PERSIST_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are a controlled document assistant for Wirra Mine Site.
Your role is to answer questions using only the approved controlled documents provided.

Rules:
- Answer using only information contained in the provided documents.
- Always cite the document name and section number when giving an answer.
- If the answer is not covered in the provided documents, respond with:
  "This information is not covered in the current controlled documents. Please consult your supervisor or the document controller."
- Never guess, infer, or draw on knowledge outside the provided documents.
- Prioritise accuracy and safety over helpfulness."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Documents:\n\n{context}\n\nQuestion: {question}"),
])


def build_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if Path(PERSIST_DIR).exists():
        print("Loading existing document index...")
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    print("Indexing documents for the first time...")
    loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents.")
    return vectorstore


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Source: {Path(doc.metadata.get('source', 'Unknown')).name}]\n{doc.page_content}"
        for doc in docs
    )


def build_chain(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


def main():
    vectorstore = build_vectorstore()
    chain = build_chain(vectorstore)

    print("\nControlled Document Interpreter — Wirra Mine Site")
    print("Ask a question about site procedures, permits, or environmental conditions.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer = chain.invoke(question)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
