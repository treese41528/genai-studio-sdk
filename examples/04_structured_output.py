"""Structured output: a validated, typed final answer.

`output_schema` forces the model to return data matching a pydantic model
(via a hidden return_result tool), validated with one self-correct retry.

Run: python examples/04_structured_output.py   (needs pydantic: pip install '.[structured]')
"""

from __future__ import annotations

from pydantic import BaseModel

from genai_studio.agents import Agent, ConsoleTracer
from _common import make_client


class Sentiment(BaseModel):
    label: str          # "positive" | "negative" | "neutral"
    confidence: float   # 0..1
    rationale: str


if __name__ == "__main__":
    agent = Agent(client=make_client(), output_schema=Sentiment, tracer=ConsoleTracer())
    result = agent.run(
        "Classify the sentiment of this review: "
        "'The presentation was clear and genuinely engaging, but the room was freezing.'"
    )
    out = result.output
    print("\n=== typed result ===")
    print("type:", type(out).__name__)
    print("label:", out.label)
    print("confidence:", out.confidence)
    print("rationale:", out.rationale)
