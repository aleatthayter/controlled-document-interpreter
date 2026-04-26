from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from mcp.server.fastmcp import FastMCP

PERSIST_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

mcp = FastMCP("controlled-document-interpreter")

_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    return _vectorstore


@mcp.tool()
def search_controlled_documents(question: str) -> str:
    """Search Wirra Mine Site controlled documents including safety procedures,
    permit conditions, and environmental licence requirements. Returns relevant
    document sections with source citations. Only returns information that exists
    in the approved documents — does not infer or generalise."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    if not docs:
        return "No relevant sections found in the controlled documents."

    return "\n\n".join(
        f"[Source: {Path(doc.metadata.get('source', 'Unknown')).name}]\n{doc.page_content}"
        for doc in docs
    )


if __name__ == "__main__":
    mcp.run()
