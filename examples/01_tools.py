"""Tools: a typed function becomes a model-callable tool.

Run: python examples/01_tools.py
"""

from __future__ import annotations

import json

from genai_studio.agents import tool


@tool
def search_papers(query: str, limit: int = 10) -> str:
    """Search the literature for papers matching a query.

    Args:
        query: Search terms.
        limit: Maximum number of results.
    """
    return f"(pretend) {limit} results for {query!r}"


if __name__ == "__main__":
    # The auto-generated JSON Schema the model will see:
    print("=== search_papers.spec.parameters ===")
    print(json.dumps(search_papers.spec.parameters, indent=2))
    print("\ndescription:", search_papers.spec.description)

    # A @tool is still an ordinary function — call it directly, no model involved:
    print("\ndirect call:", search_papers(query="protein folding", limit=3))

    # And this is exactly what gets sent to an OpenAI-compatible gateway:
    print("\n=== OpenAI tool definition ===")
    print(json.dumps(search_papers.spec.to_openai(), indent=2))
