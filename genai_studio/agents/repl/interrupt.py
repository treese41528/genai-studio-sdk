"""Two-tier Ctrl-C for one agent turn.

The agent's cooperative ``cancel`` token is only polled at step boundaries and after
each tool, so a single in-flight blocking call won't honour it immediately. Hence two
tiers: the FIRST Ctrl-C sets the token (the agent finalizes cleanly with
``stopped="cancelled"`` at the next checkpoint and the turn is kept); a SECOND Ctrl-C
raises ``KeyboardInterrupt`` to force-unwind the generator (the turn is discarded). At
the input prompt the default handler is restored, so Ctrl-C there behaves normally.
"""

from __future__ import annotations

import signal
import sys
from contextlib import contextmanager


@contextmanager
def turn_interrupt(tok):
    state = {"hits": 0}

    def handler(signum, frame):
        state["hits"] += 1
        if state["hits"] == 1:
            tok.cancel()
            sys.stderr.write("\n(interrupting… press Ctrl-C again to force)\n")
            sys.stderr.flush()
        else:
            raise KeyboardInterrupt

    try:
        prev = signal.signal(signal.SIGINT, handler)
    except ValueError:                  # not on the main thread — no signal handling
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, prev)
