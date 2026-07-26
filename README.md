# PDF + Web Search RAG Agent (FastAPI + MCP + Groq)

An agent that answers questions either from an uploaded PDF or, if the PDF
has nothing relevant, from a live DuckDuckGo web search (both exposed as
tools on a stdio **MCP server**), falling back to the LLM's own knowledge
only as a last resort, and truthfully saying **"No relevant information
found."** rather than fabricating an answer.

## Pipeline

```
User query
   │
   ▼
Lakera Guard + NeMo Guardrails (prompt-injection screen)
   │
   ▼
Is a PDF active for this session? ──No──► DuckDuckGo web search (MCP tool)
   │ Yes                                        │
   ▼                                            │
Chroma similarity search (HF embeddings)        │
   │                                            │
Relevant chunks found? ──No─────────────────────┤
   │ Yes                                        │
   ▼                                            ▼
Answer from PDF context              Any web results? ──No──► LLM's own knowledge
                                              │ Yes                  │
                                              ▼                      ▼
                                     Answer from web context   Confident? → answer
                                                                Not confident? →
                                                                "No relevant information found."
```

The LLM may also emit `ASK_HUMAN: <question>` at any point (via the
`ask_human` MCP tool convention) when the request is ambiguous. Because
FastAPI is stateless, this doesn't block: the question is returned
immediately in the HTTP response (`requires_human_input: true`), and the
**next** request in the same `session_id` is treated as the user's answer
and the pipeline resumes.

PDF chunks are screened by Lakera + NeMo **after chunking, before
embedding/indexing** -- so a prompt injection buried inside an uploaded PDF
page is dropped rather than silently entering the vector store and later
hijacking the LLM at answer time.

## Project layout

```
mcp_server/server.py        stdio MCP server: ask_human, duckduckgo_search tools
app/main.py                 FastAPI app (/upload, /query, /health)
app/orchestrator.py         core decision pipeline (PDF -> web -> LLM -> "no info")
app/rag/                    pdf_processor, embeddings (HF API), vector_store (Chroma), retriever
app/security/               lakera_guard.py, nemo_guardrails.py
app/mcp_client/client.py    stdio MCP client (spawns mcp_server/server.py)
app/llm/groq_client.py      Groq chat completion wrapper
app/evaluation/metrics.py   precision@k, recall@k, ndcg@k, all @traceable (LangSmith)
app/session_manager.py      in-memory session store: last 5 turns + pending ask_human state
app/schemas.py               Pydantic request/response models (structured output)
app/exceptions.py           custom exception hierarchy + FastAPI handlers
app/logging_config.py       structured JSON logging
guardrails_config/          NeMo Guardrails config (config.yml, rails/rails.co)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY, HF_API_TOKEN, LANGCHAIN_API_KEY, LAKERA_API_KEY
```

Run the API:

```bash
uvicorn app.main:app --reload
```

The MCP server (`mcp_server/server.py`) is spawned automatically as a
subprocess by `app/mcp_client/client.py` on each tool call -- you don't run
it separately, but you can smoke-test it standalone with:

```bash
python mcp_server/server.py
```

## Usage

**Upload a PDF** (starts/continues a session):

```bash
curl -X POST "http://localhost:8000/upload?session_id=demo-1" \
  -F "file=@/path/to/document.pdf"
```

**Ask a question** (uses the PDF from that session if present, else web
search / LLM knowledge):

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "query": "What does the document say about refund policy?"}'
```

If the response has `"requires_human_input": true`, send the user's answer
as the next `/query` call with the **same `session_id`** -- the pipeline
picks the pending question back up automatically.

## Notes / things you'll want to tune before production

- `SessionManager` is an in-process dict -- swap for Redis if you run
  multiple workers/replicas.
- `RELEVANCE_SCORE_THRESHOLD` in `.env` controls how strict the "does the
  PDF actually answer this" cutoff is; tune against your embedding model.
- Lakera/NeMo currently **fail open** on transport errors (log + continue)
  so an outage in either service doesn't take the whole app down; flip that
  to fail-closed if your threat model requires it.
- `evaluate_retrieval` uses the relevance-threshold pass/fail as a *proxy*
  for ground truth in production monitoring. For real precision/recall/nDCG
  numbers, build a labeled eval dataset in LangSmith and run
  `evaluate_retrieval` against it offline.
