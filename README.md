# Controlled Document Interpreter

An AI agent that answers plain-language questions from approved controlled documents — maintenance procedures, safety protocols, and environmental permit conditions — grounded strictly in the source documents, with evaluation and tracing built in.

Built with Python, LangChain, Claude (Anthropic), ChromaDB, LangSmith, Pydantic, and MCP.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set environment variables:

```bash
export ANTHROPIC_API_KEY=your-key

# Optional: LangSmith tracing
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your-langsmith-key
export LANGCHAIN_PROJECT=controlled-document-interpreter
```

## Usage

### Interactive query

```bash
python main.py
```

On first run, documents in `data/documents/` are indexed into a local ChromaDB vector store. Subsequent runs load the existing index.

### Run evaluation suite

```bash
python evaluate.py
```

Runs 12 test cases covering answer accuracy, faithfulness, retrieval relevance, and correct refusal for out-of-scope questions. Results are written to `output/eval_results.xlsx`.

### MCP server (Claude Desktop)

The document search is also exposed as an MCP server, allowing any MCP-compatible client — including Claude Desktop — to query the controlled documents directly.

Run the index first (`python main.py`, then exit) so `chroma_db/` exists, then add this to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "controlled-document-interpreter": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/mcp_server.py"],
      "cwd": "/absolute/path/to/controlled-document-interpreter"
    }
  }
}
```

Restart Claude Desktop. You can then ask Claude questions about Wirra Mine Site procedures directly in the chat interface and it will call the MCP tool to retrieve the relevant document sections.

## Adding your own documents

Place `.txt` files in `data/documents/`. Delete the `chroma_db/` directory to force re-indexing.

## Project structure

```
main.py              # interactive query CLI
mcp_server.py        # MCP server exposing document search as a tool
evaluate.py          # eval suite (faithfulness, relevance, refusal accuracy)
data/documents/      # controlled documents (txt)
evals/               # eval dataset
output/              # eval results (git-ignored)
chroma_db/           # local vector store (git-ignored)
```
