"""The capped always-on memory index + its injection into the system prompt.

The index is a small, byte-bounded list of the most recent live facts, headed with an honesty
notice (recalled facts are point-in-time and may be stale). The long tail is reachable only via
``recall_memory``. Injection routes through the shared ``assemble_system`` so the block obeys the
same single-injection budget as project memory + the skills catalog.
"""

from __future__ import annotations

_HEADER = ("# Recalled memory (durable facts the agent saved; point-in-time — may be stale, "
           "verify before relying)")


def memory_index_text(store, *, budget_chars: int = 1500, max_facts: int = 40) -> str:
    """Render the most-recent live facts as a capped bullet list (newest first). Returns ""
    when there are no facts."""
    live = store.live()
    if not live:
        return ""
    facts = sorted(live, key=lambda f: f.ts, reverse=True)[:max_facts]
    lines = [_HEADER]
    used = len(_HEADER)
    shown = 0
    for f in facts:
        line = f"- {f.text}" + (f"  (tags: {', '.join(map(str, f.tags))})" if f.tags else "")
        if budget_chars and used + len(line) + 1 > budget_chars:
            break
        lines.append(line)
        used += len(line) + 1
        shown += 1
    if shown < len(live):
        lines.append(f"- … ({len(live) - shown} more; use recall_memory to search)")
    return "\n".join(lines)


def inject_memory(system: str, memory_index: str) -> str:
    """Append the recalled-memory block to a system prompt via the shared combiner."""
    from ..compose import assemble_system
    return assemble_system(system, memory_index)
