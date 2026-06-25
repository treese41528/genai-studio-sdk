"""Probe whether the chat gateway returns per-token logprobs — THOROUGH.

Calls the chat endpoint DIRECTLY (client.chat.completions.create, bypassing the
SDK's empty-completion retry/RateLimiter) so each response is a single raw call.
For every (model, variant) it records:
  - content length + finish_reason  -> distinguishes an EMPTY completion (no
    tokens to attach logprobs to => inconclusive) from a real answer with no
    logprobs (=> genuinely unsupported);
  - whether choices[0].logprobs is present and populated, and #tokens;
and it dumps the literal HTTP JSON for one model so logprobs hidden under a
non-standard key would still be visible (grep for 'logprob').

Two prompt variants guard against the empty-greedy-completion trap: a one-word
reply and a multi-token count. Calls are paced with time.sleep (the gateway
silently drops bursts). Writes incrementally to _results/logprobs_probe.log.
"""
import os
import sys
import time
from genai_studio import GenAIStudio

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "_results", "logprobs_probe.log")
RAW = os.path.join(_HERE, "_results", "logprobs_probe_raw.json")
RAW_HTTP = os.path.join(_HERE, "_results", "logprobs_probe_http.json")

MODELS = ["qwen2.5:72b", "llama3.3:70b", "gpt-oss:120b", "gemma3:27b",
          "mistral:latest", "phi4:latest", "llama3.2:latest"]

VARIANTS = [
    ("1word_greedy", {"messages": [{"role": "user", "content": "Reply with exactly one word: pong"}],
                      "max_tokens": 8, "temperature": 0}),
    ("multitok_greedy", {"messages": [{"role": "user", "content": "Count from 1 to 8, space-separated, digits only."}],
                         "max_tokens": 40, "temperature": 0}),
]
LOGPROB_KW = {"logprobs": True, "top_logprobs": 5}
PACE_S = 3.0


def emit(line):
    print(line, flush=True)
    with open(OUT, "a") as fh:
        fh.write(line + "\n")


def probe(client, model, params):
    resp = client.chat.completions.create(model=model, **params, **LOGPROB_KW)
    ch = resp.choices[0]
    content = ch.message.content or ""
    lp = getattr(ch, "logprobs", None)
    has = bool(lp and getattr(lp, "content", None))
    n = len(lp.content) if has else 0
    sample = ""
    if has:
        t0 = lp.content[0]
        sample = (f" sample(tok={getattr(t0,'token',None)!r} "
                  f"lp={getattr(t0,'logprob',None)} top={len(getattr(t0,'top_logprobs',[]) or [])})")
    return content, ch.finish_reason, (lp is not None), has, n, sample, resp


def main():
    ai = GenAIStudio(validate_model=False)
    client = ai.client
    open(OUT, "w").close()
    emit("THOROUGH logprobs probe — direct chat.completions.create, logprobs=True top_logprobs=5")
    emit("=" * 104)
    raw_saved = False
    for m in MODELS:
        emit(m)
        for vname, params in VARIANTS:
            try:
                content, fr, obj, has, n, sample, resp = probe(client, m, params)
                emit(f"  [{vname:16}] content_len={len(content):<4} finish={str(fr):11} "
                     f"logprobs_obj={'Y' if obj else 'N'} populated={'Y' if has else 'N'} "
                     f"ntok={n:<3} content={content[:42]!r}{sample}")
                if (not raw_saved) and vname == "multitok_greedy":
                    try:
                        with open(RAW, "w") as fh:
                            fh.write(resp.model_dump_json(indent=2))
                        emit(f"      (parsed response dumped -> {os.path.basename(RAW)})")
                        raw_saved = True
                    except Exception as e:
                        emit(f"      (raw dump failed: {e})")
            except Exception as e:
                emit(f"  [{vname:16}] ERROR {type(e).__name__}: {str(e)[:110]}")
            time.sleep(PACE_S)

    # Literal HTTP JSON for the flagship — catches logprobs under a non-standard key.
    try:
        rr = client.chat.completions.with_raw_response.create(
            model="qwen2.5:72b",
            messages=[{"role": "user", "content": "Count from 1 to 8, space-separated, digits only."}],
            max_tokens=40, temperature=0, **LOGPROB_KW)
        txt = rr.text
        with open(RAW_HTTP, "w") as fh:
            fh.write(txt)
        emit("-" * 104)
        emit(f"LITERAL HTTP JSON (qwen2.5:72b): 'logprob' substring present = {'logprob' in txt}  "
             f"(saved -> {os.path.basename(RAW_HTTP)}, {len(txt)} bytes)")
    except Exception as e:
        emit(f"with_raw_response failed: {type(e).__name__}: {str(e)[:120]}")
    emit("DONE")


if __name__ == "__main__":
    sys.exit(main())
