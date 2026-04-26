# Controlled Document Interpreter

An AI agent that answers plain-language questions from approved controlled documents — maintenance procedures, safety protocols, and permit conditions — grounded strictly in the source text, with no inference beyond what the documents say.

## The Problem

Mining and energy sites accumulate thousands of pages of controlled documents. A field worker preparing for a confined space entry needs the exact permit conditions, quickly. Navigating a document management system under time pressure is slow and error-prone. This agent makes those documents queryable in natural language while preserving the strict source-grounding that safety-critical information requires.

## How It Works

1. **Indexes** documents from `data/documents/` into a local ChromaDB vector store on first run
2. **Retrieves** the most relevant sections using semantic search when a question is asked
3. **Answers** using only the retrieved text — the model cites the source document and section
4. **Refuses** if the answer is not in the documents, directing the worker to a supervisor

The agent does not improvise. If the information is not in the indexed documents, it says so.

## Tech Stack

Python · Claude (Anthropic) · LangChain · ChromaDB · LangSmith · Pydantic · MCP

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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

Documents in `data/documents/` are indexed on first run. Subsequent runs load the existing index.

### Evaluation suite

```bash
python evaluate.py
```

Runs 12 test cases covering answer accuracy, faithfulness, retrieval relevance, and correct refusal for out-of-scope questions. Results are written to `output/eval_results.xlsx`.

### MCP server (Claude Desktop)

The document search is also exposed as an MCP server, allowing Claude Desktop to query the controlled documents directly from the chat interface.

Run the index first (`python main.py`, then exit), then add this to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

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

## Adding Your Own Documents

Place `.txt` files in `data/documents/` and delete `chroma_db/` to force re-indexing.

## Project Structure

```
main.py              # interactive query CLI
mcp_server.py        # MCP server exposing document search as a tool
evaluate.py          # eval suite (faithfulness, relevance, refusal accuracy)
data/documents/      # controlled documents (txt)
evals/               # eval dataset
output/              # eval results (git-ignored)
chroma_db/           # local vector store (git-ignored)
```
