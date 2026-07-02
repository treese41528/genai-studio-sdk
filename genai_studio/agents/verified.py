"""``verified_best_of`` — the *check≪solve* primitive.

When VERIFYING an answer is much cheaper than PRODUCING one, sampling k candidates and keeping only
those a SOUND checker accepts turns *pass@k* into *accuracy*: the model only has to be right ONCE,
and the checker discards the wrong tries. This was established empirically for root-solving
(``benchmarks/root_solve_eval.py``: the completeness filter matched pass@k, +13pp over greedy) and is
generalized here to any class that has a sound, cheap checker — inequalities (z3), factorizations
(expand-and-compare), roots (substitute).

This module is model-agnostic: candidates come from anywhere (k sampled completions, a fan-out, a
tool). It only does the *selection* — filter by the checker, then vote among the survivors, with a
graceful fallback so there is always an answer. The checkers themselves live next to their tools
(``verify_factorization`` in ``tools/symbolic.py``; ``z3_decide`` in ``tools/smt.py``).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Sequence


@dataclass(frozen=True)
class VerifiedPick:
    """The outcome of a verified selection over candidate answers."""
    answer: object                 # the chosen candidate (None only when there were no candidates)
    verified: bool                 # True iff the choice passed the sound checker (trustworthy)
    n_candidates: int              # how many non-null candidates were considered
    n_verified: int                # how many passed the checker
    method: str                    # "verified-vote" | "validity-vote" | "fallback-vote" | "empty"


def _safe(fn: Callable, c) -> bool:
    try:
        return bool(fn(c))
    except Exception:
        return False


def _vote(items: Sequence, key: Callable):
    """Majority vote by ``key``, returning the first item bearing the winning key (stable)."""
    counts = Counter(key(c) for c in items)
    winner = counts.most_common(1)[0][0]
    return next(c for c in items if key(c) == winner)


def verified_best_of(candidates, check: Callable, *, complete: Optional[Callable] = None,
                     key: Callable = str) -> VerifiedPick:
    """Pick the best candidate under a SOUND checker (the *check≪solve* selection).

    Args:
        candidates: iterable of candidate answers (any type; ``None`` entries are ignored).
        check: ``check(candidate) -> bool``, the sound VALIDITY check (cheap). Only accepted
            candidates are trusted; a checker that raises counts as reject.
        complete: optional STRONGER check (e.g. "all roots found", not just "each root valid").
            Preferred over ``check`` when any candidate satisfies it; falls back to ``check``.
        key: ``key(candidate) -> hashable`` for voting/dedup (default ``str``).

    Returns a :class:`VerifiedPick`. If any candidate passes the strongest available checker, votes
    among those survivors (``verified=True``). Otherwise votes over all candidates (``verified=False``)
    so callers always get an answer — but can see it was unverified.
    """
    cands = [c for c in candidates if c is not None]
    if not cands:
        return VerifiedPick(None, False, 0, 0, "empty")
    if complete is not None:
        strong = [c for c in cands if _safe(complete, c)]
        if strong:
            return VerifiedPick(_vote(strong, key), True, len(cands), len(strong), "verified-vote")
    valid = [c for c in cands if _safe(check, c)]
    if valid:
        method = "validity-vote" if complete is not None else "verified-vote"
        return VerifiedPick(_vote(valid, key), True, len(cands), len(valid), method)
    return VerifiedPick(_vote(cands, key), False, len(cands), 0, "fallback-vote")


# ── ready-made sound checkers (thin wrappers over the existing tools) ─────────────────────────────
def inequality_check(claim: str) -> bool:
    """Sound: is ``claim`` (a polynomial (in)equality) PROVEN for all reals by z3?"""
    try:
        from .tools.smt import z3_decide
    except Exception:
        return False
    return z3_decide(claim, "real")[0] == "proven"


def factorization_check(expression: str, factored: str) -> bool:
    """Sound: does ``factored`` expand to ``expression`` AND is it genuinely factored (a product)?"""
    try:
        from .tools.symbolic import is_factorization
    except Exception:
        return False
    return is_factorization(expression, factored)
