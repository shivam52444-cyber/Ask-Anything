"""
Standalone MCP server, communicating over stdio, exposing two tools:

  1. ask_human      -> signals that the agent needs clarification from the
                        end user. Since the FastAPI layer is stateless/HTTP,
                        this tool does NOT block waiting for a reply -- it
                        simply returns a structured payload; the FastAPI
                        client is responsible for surfacing the question to
                        the user and resuming the conversation on their next
                        HTTP call.

  2. duckduckgo_search -> free web search via the `duckduckgo-search` package,
                        used as the fallback when the PDF/knowledge base has
                        no relevant answer.

Run standalone for local testing:
    python mcp_server/server.py
The FastAPI app spawns this same script as a subprocess over stdio.
"""
import logging
import sys

from duckduckgo_search import DDGS
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mcp_server")

mcp = FastMCP("pdf-rag-tools")


@mcp.tool()
def ask_human(question: str) -> dict:
    """
    Signal that the agent needs clarification from the human user before it
    can proceed. Does not block -- returns immediately with a payload the
    calling FastAPI app must surface to the user.

    Args:
        question: The clarification question to show the user.
    """
    logger.info("ask_human_invoked question=%s", question)
    return {
        "type": "ask_human",
        "question": question,
        "instructions": "Return this question to the end user via HTTP response "
        "and resume this session once they reply.",
    }


@mcp.tool()
def duckduckgo_search(query: str, max_results: int = 5) -> dict:
    """
    Search the web via DuckDuckGo and return top results.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (default 5).
    """
    logger.info("duckduckgo_search_invoked query=%s", query)
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                    }
                )
    except Exception as exc:
        logger.error("duckduckgo_search_failed error=%s", exc)
        return {"type": "web_search", "query": query, "results": [], "error": str(exc)}

    return {"type": "web_search", "query": query, "results": results}


if __name__ == "__main__":
    mcp.run(transport="stdio")
