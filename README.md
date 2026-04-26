# Controlled Document Interpreter

An AI agent that answers plain-language questions from approved controlled documents — maintenance procedures, safety protocols, and environmental permit conditions — grounded strictly in the source documents, with evaluation and tracing built in.

Built with Python, LangChain, Claude (Anthropic), ChromaDB, LangSmith, and Pydantic.

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

## Adding your own documents

Place `.txt` files in `data/documents/`. Delete the `chroma_db/` directory to force re-indexing.

## Project structure

```
main.py              # interactive query CLI
evaluate.py          # eval suite (faithfulness, relevance, refusal accuracy)
data/documents/      # controlled documents (txt)
evals/               # eval dataset
output/              # eval results (git-ignored)
chroma_db/           # local vector store (git-ignored)
```
