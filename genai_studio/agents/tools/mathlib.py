"""Optional Lean + **mathlib** proof track.

Lean core (`decide`/`omega`/`rfl`) proves basic goals; real competition math (`ring`/`norm_num`/
`linarith`/`positivity` + the ~25k-lemma library) needs mathlib — a heavy, separate install. This
module wires mathlib in WHEN PRESENT and stays inert otherwise:

  - ``mathlib_project()`` locates a Lean project that has mathlib (`$GENAI_STUDIO_LEAN_PROJECT`, else
    a default under `~/lean-work/proofs`).
  - the mathlib-backed ``lean_check``/``grade_proof`` run the kernel INSIDE that project
    (`make_lean_check(project_dir=…)`) so ``import Mathlib`` resolves.
  - ``search_lemmas`` retrieves relevant mathlib declarations (scan the source once → hybrid
    keyword + optional embedding rank) so the model can FIND the lemma it needs before proving.
  - ``setup_mathlib(dir)`` scaffolds a project + ``lake exe cache get`` (prebuilt oleans) for users
    without one.

Nothing here imports heavy deps at module load; ``mathlib_tools`` returns ``[]`` when no project.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from genai_studio.agents import ToolResult, tool

_DEFAULT_PROJECT = "~/lean-work/proofs"


def mathlib_project(path: str | None = None) -> str | None:
    """Path to a Lean project that has mathlib built, or None. Checks (in order): the ``path`` arg,
    ``$GENAI_STUDIO_LEAN_PROJECT``, then the default. A project qualifies iff its mathlib package
    source is present under ``.lake/packages/mathlib``."""
    for cand in (path, os.environ.get("GENAI_STUDIO_LEAN_PROJECT"), _DEFAULT_PROJECT):
        if not cand:
            continue
        root = Path(os.path.expanduser(cand))
        if (root / ".lake" / "packages" / "mathlib" / "Mathlib").is_dir():
            return str(root)
    return None


def _mathlib_src(project: str) -> Path:
    return Path(project) / ".lake" / "packages" / "mathlib" / "Mathlib"


# ── declaration scanner (build the search corpus from mathlib source) ─────────────────────────────
@dataclass(frozen=True)
class LemmaDecl:
    name: str
    signature: str            # the statement text (binders + `: type`), whitespace-collapsed
    module: str               # dotted module path, e.g. Mathlib.Algebra.Order.Ring.Lemmas
    doc: str = ""             # leading `/-- … -/` doc comment, if any

    def search_text(self) -> str:
        return f"{self.name} {self.signature} {self.doc}".strip()


# a declaration: optional `@[attr]`, optional modifiers, `theorem|lemma NAME <sig> :=|where|<eol>`
_DECL_RE = re.compile(
    r"^(?:@\[[^\]]*\]\s*)?"
    r"(?:protected\s+|private\s+|nonrec\s+|scoped\s+)*"
    r"(?:theorem|lemma)\s+(?P<name>[A-Za-z_][A-Za-z0-9_'.]*)"
    r"(?P<sig>.*?)(?::=|\bwhere\b|\Z)",
    re.DOTALL | re.MULTILINE)
_DOC_RE = re.compile(r"/--(?P<doc>(?:[^-]|-(?!/))*?)-/\s*\Z", re.DOTALL)
_WS = re.compile(r"\s+")


def scan_file(text: str, module: str) -> list[LemmaDecl]:
    """Extract theorem/lemma declarations from one Lean source string."""
    out = []
    for m in _DECL_RE.finditer(text):
        sig = _WS.sub(" ", m.group("sig") or "").strip(" :")
        if not sig:
            continue
        dm = _DOC_RE.search(text[:m.start()])              # doc comment immediately preceding
        doc = _WS.sub(" ", dm.group("doc")).strip() if dm else ""
        out.append(LemmaDecl(m.group("name"), sig[:400], module, doc[:300]))
    return out


def scan_declarations(project: str, *, limit: int | None = None) -> list[LemmaDecl]:
    """Scan a mathlib project's source into a list of declarations (name + signature + module + doc)."""
    src = _mathlib_src(project)
    decls: list[LemmaDecl] = []
    for path in sorted(src.rglob("*.lean")):
        module = "Mathlib." + str(path.relative_to(src)).removesuffix(".lean").replace(os.sep, ".")
        try:
            decls.extend(scan_file(path.read_text(encoding="utf-8", errors="ignore"), module))
        except Exception:
            continue
        if limit and len(decls) >= limit:
            break
    return decls


def build_lemma_index(project: str, *, cache: str | None = None, rebuild: bool = False) -> list[LemmaDecl]:
    """Scan (or load a cached scan of) the mathlib declaration index. Cached as JSON since scanning
    ~8k files takes a few seconds."""
    cache_path = Path(os.path.expanduser(cache)) if cache else Path(project) / ".lake" / "genai_lemmas.json"
    if cache_path.exists() and not rebuild:
        try:
            return [LemmaDecl(**d) for d in json.loads(cache_path.read_text())]
        except Exception:
            pass
    decls = scan_declarations(project)
    try:
        cache_path.write_text(json.dumps([d.__dict__ for d in decls]))
    except Exception:
        pass
    return decls


# ── search_lemmas (hybrid keyword + optional embedding retrieval) ─────────────────────────────────
def _tokens(s: str) -> set:
    return {w for w in re.split(r"[^A-Za-z0-9]+", s.lower()) if len(w) > 1}


def make_search_lemmas(index, *, embedder=None, k: int = 8, prefilter: int = 40):
    """Build the ``search_lemmas`` tool over a declaration index. Keyword-ranks the whole corpus, then
    (if an embedder is given) reranks only the top ``prefilter`` by cosine — so at most ``prefilter``
    embeddings are computed per query, never the whole ~180k corpus. Fails open to keyword-only.

    ``index`` may be a ``list[LemmaDecl]`` or a zero-arg callable returning one (loaded LAZILY on the
    first search, so building the index never slows agent startup)."""
    state: dict = {}

    def _corpus():
        if "decls" not in state:
            decls = index() if callable(index) else index
            state["decls"] = decls
            state["texts"] = [d.search_text() for d in decls]
        return state["decls"], state["texts"]

    @tool
    def search_lemmas(query: str, k: int = k) -> ToolResult:
        """Search the mathlib library for lemmas/theorems relevant to a goal, BEFORE proving. Returns
        matching declaration names + signatures you can then use (e.g. `exact?`, `apply <name>`, or
        cite in a proof). Search by concept or by the shape of the statement.

        Args:
            query: what you need, e.g. "sum of squares nonneg", "cauchy schwarz inner product",
                "gcd dvd", "triangle inequality norm".
        """
        decls, texts = _corpus()
        qtok = _tokens(query)
        scored = [(len(qtok & _tokens(t)), i) for i, t in enumerate(texts)]
        scored = [(s, i) for s, i in scored if s > 0]
        scored.sort(key=lambda si: si[0], reverse=True)
        top = [i for _, i in scored[:prefilter]]
        if not top:
            return ToolResult(content=f"No mathlib lemmas matched {query!r}. Try different terms.")
        if embedder is not None:
            qv = embedder(query)
            vecs = embedder([texts[i] for i in top]) if qv else None
            if qv and isinstance(vecs, list) and len(vecs) == len(top) and all(vecs):
                from ..embed import cosine
                top = [i for _, i in sorted(zip(vecs, top), key=lambda vi: cosine(qv, vi[0]), reverse=True)]
        hits = top[:k]
        listing = "\n".join(f"- {decls[i].name}  :  {decls[i].signature}" for i in hits)
        return ToolResult(content=f"{len(hits)} mathlib lemma(s) for {query!r}:\n{listing}",
                          data={"names": [decls[i].name for i in hits]})

    return search_lemmas


# ── assembly + setup ──────────────────────────────────────────────────────────────────────────────
def mathlib_tools(client=None, *, project: str | None = None, model: str | None = None,
                  search: bool = True):
    """The mathlib-backed proof tools when a project is present, else ``[]``. Returns
    ``[lean_check, grade_proof, (search_lemmas)]`` — lean_check/grade_proof run inside the project so
    ``import Mathlib`` works; search_lemmas indexes the source (embedding rerank if ``client`` embeds)."""
    proj = mathlib_project(project)
    if proj is None:
        return []
    from .lean import make_grade_proof, make_lean_check
    tools = [make_lean_check(project_dir=proj), make_grade_proof(project_dir=proj)]
    if search:
        embedder = None
        if client is not None:
            try:
                from ..embed import DEFAULT_EMBED_MODEL, make_embedder
                embedder = make_embedder(client, model=model or DEFAULT_EMBED_MODEL)
            except Exception:
                embedder = None
        tools.append(make_search_lemmas(lambda: build_lemma_index(proj), embedder=embedder))
    return tools


def setup_mathlib(directory: str, *, name: str = "proofs", timeout: float = 1800) -> ToolResult:
    """Scaffold a Lean project with mathlib and fetch the PREBUILT cache (`lake exe cache get`) — a
    large download, not a from-scratch compile. One-time, opt-in. Requires the Lean toolchain (elan)."""
    from .lean import lake_available
    lake = lake_available()
    if lake is None:
        return ToolResult(content="", error="lake/elan not installed — see "
                          "https://leanprover-community.github.io/get_started.html")
    root = Path(os.path.expanduser(directory))
    root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PATH": os.path.dirname(lake) + os.pathsep + os.environ.get("PATH", "")}
    try:
        r = subprocess.run([lake, "new", name, "math"], cwd=root, env=env,
                           capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        return ToolResult(content="", error=f"lake new failed: {e}")
    proj = root / name
    ok = (proj / ".lake" / "packages" / "mathlib" / "Mathlib").is_dir()
    tail = (r.stdout + r.stderr)[-400:]
    if ok:
        return ToolResult(content=f"mathlib project ready at {proj}. Set GENAI_STUDIO_LEAN_PROJECT={proj} "
                          "to use it.", data={"project": str(proj)})
    return ToolResult(content="", error=f"mathlib setup incomplete (run `lake exe cache get` in {proj}).\n{tail}")
