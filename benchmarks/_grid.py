import json
d = json.load(open('_results/hotpotqa_paper_n500.json'))
models = ["qwen2.5:72b", "llama3.3:70b", "gpt-oss:120b", "llama4:latest", "gemma3:27b",
          "gemma3:1b", "qwen3:4b", "llama3.2:latest", "deepseek-r1:32b", "qwq:latest"]
conds = ["standard", "cot", "act", "react"]
print(f"=== {len(d)}/40 cells | total infra {sum(v['infra'] for v in d.values())} ===")
print("(paper PaLM-540B: Std 28.7 / CoT 29.4 / Act 25.7 / ReAct 27.4)\n")
print(f"  {'model':16}" + "".join(f"{c:>9}" for c in conds))
allinfra, hi = [], []
for m in models:
    cells = []
    for c in conds:
        k = m + '|' + c
        if k not in d:
            cells.append(f"{'MISS':>8} ")
        elif d[k]['valid'] == 0:
            cells.append(f"{'INFRA':>8} "); allinfra.append(k)
        else:
            cells.append(f"{d[k]['fair_em']*100:7.1f}% ")
            if d[k]['infra'] > 50:
                hi.append((k, d[k]['valid'], d[k]['infra']))
    print(f"  {m:16}" + "".join(cells))
print("\nall-infra cells:", allinfra or "none")
print("high-infra cells (infra>50 => smaller effective-n):",
      [f"{k}: valid={vv} infra={ii}" for k, vv, ii in hi] or "none")
# best per condition
print("\nbest per condition:")
for c in conds:
    vals = [(m, d[m + '|' + c]['fair_em']) for m in models if (m + '|' + c) in d and d[m + '|' + c]['valid'] > 0]
    bm, bv = max(vals, key=lambda x: x[1])
    print(f"  {c:9} -> {bm} {bv*100:.1f}%")
