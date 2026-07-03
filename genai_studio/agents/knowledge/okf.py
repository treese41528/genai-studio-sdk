"""
OKF — Open Knowledge Format v0.1 loader (Google Cloud, Apache-2.0).

OKF is *"a starting point, not a finished standard"*: a directory tree of UTF-8
markdown, **one concept per file**, where the file PATH is the concept's identity.
Reserved names ``index.md`` (bundle index) and ``log.md`` (changelog) are skipped
by default. Frontmatter (a leading ``---`` block) has exactly **one required
field, ``type``**; ``title`` / ``description`` / ``resource`` / ``tags`` /
``timestamp`` are recommended. Per the spec a consumer **MUST NOT reject unknown
keys or missing optional fields** — so this loader validates only the one MUST
(``type`` present) and preserves everything else verbatim.

This gives course authors a git-friendly knowledge convention that maps 1:1 onto
:class:`~genai_studio.agents.tool.Source` provenance (``resource`` → ``url``,
``title`` → ``title``, ``type`` / ``tags`` → ``metadata``). :func:`ingest_bundle`
pushes a bundle into a gateway knowledge base for **server-side** RAG — the only
sane path, since only ~15 of 35 gateway models embed, so embedding must happen on
the operator's configured model, not client-side.

Frontmatter is parsed with PyYAML when available, else a tiny stdlib fallback that
handles the recommended scalar fields plus inline ``[a, b]`` / block ``- item``
tag lists. PyYAML is an optional ``[knowledge]`` extra.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field

from ..tool import Source

RESERVED = {"index.md", "log.md"}   # OKF reserves these; not concept docs
_READY_STATES = {"processed", "completed", "complete", "success", "done", "ready"}


@dataclass
class OKFDoc:
    """One OKF concept: its path-identity, required ``type``, body, and the full
    (unknown-key-preserving) frontmatter."""

    identity: str          # path relative to the bundle root, sans .md (the concept id)
    type: str              # the one required frontmatter field
    body: str = ""         # markdown after the frontmatter
    frontmatter: dict = field(default_factory=dict)
    path: str | None = None   # absolute file path (for upload)

    @property
    def title(self) -> str:
        return self.frontmatter.get("title") or self.identity

    @property
    def resource(self) -> str | None:
        return self.frontmatter.get("resource")

    @property
    def tags(self) -> list:
        t = self.frontmatter.get("tags") or []
        return t if isinstance(t, list) else [t]

    def to_source(self) -> Source:
        """Map onto a :class:`Source` so retrieved/cited knowledge keeps provenance."""
        meta = {"type": self.type}
        if self.tags:
            meta["tags"] = self.tags
        for k in ("description", "timestamp"):
            if self.frontmatter.get(k):
                meta[k] = self.frontmatter[k]
        snippet = (self.body or "").strip()[:200] or None
        return Source(id=self.identity, title=self.title, url=self.resource,
                      snippet=snippet, metadata=meta)


# ════════════════════════════════════════════════════════════════════════════
# Loading
# ════════════════════════════════════════════════════════════════════════════

def load_bundle(path: str, *, include_reserved: bool = False,
                strict: bool = False) -> list[OKFDoc]:
    """Load an OKF bundle directory into a list of :class:`OKFDoc` (sorted by identity).

    Args:
        path: the bundle root (a directory of ``*.md`` concept files).
        include_reserved: also load ``index.md`` / ``log.md`` (skipped by default).
        strict: raise on a doc missing the required ``type`` field; otherwise such
            a doc is skipped with a warning (the spec's "MUST NOT reject" applies to
            unknown/optional keys, not the one required field).
    """
    root = os.path.abspath(path)
    if not os.path.isdir(root):
        raise NotADirectoryError(f"OKF bundle not found or not a directory: {path!r}")
    docs: list[OKFDoc] = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".md"):
                continue
            fpath = os.path.join(dirpath, fn)
            rel = os.path.relpath(fpath, root)
            if rel in RESERVED and not include_reserved:
                continue                         # index.md/log.md are reserved at the ROOT only
            identity = rel[:-3].replace(os.sep, "/")
            try:
                with open(fpath, encoding="utf-8-sig") as fh:   # utf-8-sig transparently strips a BOM
                    text = fh.read()
            except (UnicodeDecodeError, OSError) as exc:         # one bad file must not abort the bundle
                warnings.warn(f"OKF doc {identity!r} unreadable, skipped: {exc}", stacklevel=2)
                continue
            fm, body = _split_frontmatter(text)
            typ = fm.get("type")
            if not typ:
                msg = f"OKF doc {identity!r} is missing the required frontmatter field 'type'"
                if strict:
                    raise ValueError(msg)
                warnings.warn(msg, stacklevel=2)
                continue
            docs.append(OKFDoc(identity=identity, type=str(typ), body=body,
                               frontmatter=fm, path=fpath))
    docs.sort(key=lambda d: d.identity)
    return docs


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a leading ``---`` frontmatter block from the markdown body."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    close = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if close is None:
        return {}, text                      # no closing fence — treat all as body
    fm = _parse_frontmatter("\n".join(lines[1:close]))
    body = "\n".join(lines[close + 1:]).lstrip("\n")
    return fm, body


def _parse_frontmatter(fm_text: str) -> dict:
    if not fm_text.strip():
        return {}
    try:
        import yaml                          # optional [knowledge] extra
        data = yaml.safe_load(fm_text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return _mini_yaml(fm_text)           # missing PyYAML or malformed YAML


def _mini_yaml(text: str) -> dict:
    """Stdlib fallback: ``key: value``, inline ``[a, b]``, and block ``- item`` lists.
    Deliberately small — for richer frontmatter, install the ``[knowledge]`` extra."""
    out: dict = {}
    cur_key: str | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        item = re.match(r"\s*-\s+(.*)$", raw)
        if item and cur_key is not None:
            if not isinstance(out.get(cur_key), list):
                out[cur_key] = []
            out[cur_key].append(_scalar(item.group(1)))
            continue
        kv = re.match(r"([A-Za-z0-9_\-]+)\s*:\s*(.*)$", raw)
        if not kv:
            continue
        key, val = kv.group(1), kv.group(2).strip()
        if val == "":
            out[key], cur_key = "", key      # may be promoted to a list by block items
        elif val.startswith("[") and val.endswith("]"):
            out[key] = [_scalar(x) for x in val[1:-1].split(",") if x.strip()]
            cur_key = None
        else:
            out[key], cur_key = _scalar(val), None
    return out


def _scalar(s: str):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


# ════════════════════════════════════════════════════════════════════════════
# Ingestion into a gateway knowledge base (server-side embedding)
# ════════════════════════════════════════════════════════════════════════════

def ingest_bundle(studio, bundle_path: str, kb_name: str, *, description: str = "",
                  settle: float = 2.0, poll_interval: float = 2.0,
                  timeout: float = 120.0, include_reserved: bool = False) -> str:
    """Walk an OKF bundle, create a gateway KB, upload + link every concept doc, and
    wait until the KB reports its files processed. Returns the new KB id.

    Composes the gateway CRUD (``upload_file`` → ``create_knowledge_base`` →
    ``add_file_to_knowledge_base`` → poll ``get_knowledge_base``), so it needs a
    live ``GenAIStudio``. The processed-poll is best-effort: it returns once the KB
    reports all files present/ready, or after ``timeout`` (the gateway may keep
    embedding server-side past that). Embedding is server-side by necessity — only
    ~15/35 gateway models embed.

    Args:
        studio: a connected ``GenAIStudio`` client.
        bundle_path: the OKF bundle root.
        kb_name: name for the knowledge base to create.
        description: optional KB description.
        settle: seconds to wait after uploads before polling (uploads need to land).
        poll_interval: seconds between processed-polls.
        timeout: max seconds to wait for processing before returning anyway.
        include_reserved: also ingest ``index.md`` / ``log.md``.
    """
    import time

    docs = load_bundle(bundle_path, include_reserved=include_reserved)
    if not docs:
        raise ValueError(f"no OKF concept docs found under {bundle_path!r}")
    kb = studio.create_knowledge_base(kb_name, description)
    for d in docs:
        info = studio.upload_file(d.path)
        studio.add_file_to_knowledge_base(kb.id, info.id)

    time.sleep(settle)                       # let uploads settle before linking is queryable
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if _kb_ready(studio.get_knowledge_base(kb.id), len(docs)):
                break
        except Exception:                    # transient gateway error — keep polling
            pass
        time.sleep(poll_interval)
    return kb.id


def _kb_ready(kb, n_files: int) -> bool:
    """Best-effort readiness check from a KnowledgeBase's raw payload: all expected
    files present and (if a status is exposed) all in a ready state."""
    raw = getattr(kb, "raw", None)
    files = raw.get("files") if isinstance(raw, dict) else None
    if not isinstance(files, list) or len(files) < n_files:
        return False
    # A file that EXPOSES a status (even null) is judged by it (null -> pending);
    # a file with no status key at all counts as present == ready (best-effort).
    statuses = [str(f.get("status") or f.get("processing_status") or "pending").lower()
                for f in files
                if isinstance(f, dict) and ("status" in f or "processing_status" in f)]
    return all(s in _READY_STATES for s in statuses) if statuses else True
