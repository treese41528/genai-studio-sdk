"""JSONL session persistence: record / list / resume / compact.

Each session is an append-only JSONL log under ``~/.genai_studio/sessions/``. The
canonical resumable records are ``message`` lines (full fidelity, incl. ``tool_calls`` +
``tool_call_id`` — required or the gateway 400s on replay). ``/resume`` replays them;
a ``compact`` marker resets the accumulator (last compaction wins).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..client import Message, ToolCall


def serialize_message(m: Message) -> dict:
    return {"role": m.role, "content": m.content,
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments,
                            "raw_arguments": tc.raw_arguments} for tc in (m.tool_calls or [])],
            "tool_call_id": m.tool_call_id, "name": m.name}


def deserialize_message(d: dict) -> Message:
    tcs = [ToolCall(id=t.get("id"), name=t.get("name"), arguments=t.get("arguments") or {},
                    raw_arguments=t.get("raw_arguments")) for t in (d.get("tool_calls") or [])]
    return Message(role=d.get("role"), content=d.get("content"), tool_calls=tcs,
                   tool_call_id=d.get("tool_call_id"), name=d.get("name"))


class SessionRecorder:
    def __init__(self, sessions_dir, *, model: str, cwd):
        self.dir = Path(sessions_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.id = uuid.uuid4().hex[:12]
        self.path = self.dir / f"{time.strftime('%Y%m%dT%H%M%S')}-{self.id}.jsonl"
        self._f = open(self.path, "a", encoding="utf-8")
        self._write({"type": "session_meta", "id": self.id, "ts": time.time(),
                     "model": model, "cwd": str(cwd)})

    def _write(self, obj: dict) -> None:
        self._f.write(json.dumps(obj) + "\n")
        self._f.flush()

    def write_input(self, raw: str) -> None:
        self._write({"type": "input", "ts": time.time(), "raw": raw})

    def write_messages(self, msgs, result=None) -> None:
        for m in msgs:
            rec = serialize_message(m)
            rec["type"] = "message"
            rec["ts"] = time.time()
            self._write(rec)
        if result is not None:
            self._write({"type": "turn_end", "ts": time.time(), "stopped": getattr(result, "stopped", None)})

    def write_marker(self, type_: str, **fields) -> None:
        self._write({"type": type_, "ts": time.time(), **fields})

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


@dataclass
class SessionInfo:
    id: str
    path: Path
    ts: float
    model: str
    preview: str
    turns: int


def list_sessions(sessions_dir) -> list:
    d = Path(sessions_dir)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
        meta, preview, turns = {}, "", 0
        try:
            for line in open(p, encoding="utf-8"):
                try:
                    o = json.loads(line)
                except ValueError:
                    continue
                if o.get("type") == "session_meta":
                    meta = o
                elif o.get("type") == "message" and o.get("role") == "user" and not preview:
                    preview = (o.get("content") or "")[:60]
                elif o.get("type") == "turn_end":
                    turns += 1
        except OSError:
            continue
        out.append(SessionInfo(id=meta.get("id", p.stem), path=p,
                               ts=meta.get("ts", p.stat().st_mtime), model=meta.get("model", "?"),
                               preview=preview, turns=turns))
    return out


def load_history(path) -> list:
    history: list = []
    for line in open(path, encoding="utf-8"):
        try:
            o = json.loads(line)
        except ValueError:
            continue
        t = o.get("type")
        if t == "message":
            if o.get("role") != "system":
                history.append(deserialize_message(o))
        elif t == "compact":
            history = [Message.user(o.get("summary", ""))]      # last compaction wins
        elif t == "clear":
            history = []
    return history


def compact_history(history, client, model, *, keep: int = 4):
    """Summarize the conversation (one model call) and return ``(new_history, summary)``."""
    rendered = []
    for m in history:
        if m.role == "tool":
            rendered.append(f"[tool {m.name or ''} result]: {(m.content or '')[:500]}")
        elif m.tool_calls:
            rendered.append("[assistant calls]: " + ", ".join(f"{tc.name}({tc.arguments})" for tc in m.tool_calls))
        else:
            rendered.append(f"{m.role}: {(m.content or '')[:1000]}")
    prompt = [Message.system("Summarize the conversation so far. Preserve decisions, file paths, "
                             "command results, and open TODOs. Be concise."),
              Message.user("\n".join(rendered))]
    summary = client.complete(prompt, model=model).text or ""
    new = [Message.user("[summary of earlier conversation]\n" + summary)] + history[-keep:]
    return new, summary
