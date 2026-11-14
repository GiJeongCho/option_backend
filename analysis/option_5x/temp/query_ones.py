# -*- coding: utf-8 -*-
import json, sys
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "output" / "verify_rise_heatmap.json"
d = json.load(open(p, encoding="utf-8"))

def get(scope, sub, K):
    return {(r["T"], r["M"]): r for r in d["scopes"][scope][sub][str(K)]}

out = []

# 1) full-period robust high-prob cells
for sub in ("combined", "call"):
    g = get("all", sub, 2)
    rows = [r for r in g.values() if r["valid"] >= 20 and r["pct"] is not None]
    rows.sort(key=lambda r: (-r["pct"], -r["valid"]))
    out.append(f"== all {sub} K2  valid>=20, top by pct ==")
    for r in rows[:15]:
        out.append(f"  T={r['T']:>3} M={r['M']}  {r['reached']}/{r['valid']}={r['pct']}%  만기일{r['expiry_days']}")

# 2) how 2026 prob=1 call cells generalize to full period
out.append("\n== 2026 call prob=1.0 cells -> same cell over ALL years ==")
g26 = get("2026", "call", 2); gall = get("all", "call", 2)
cells = [(t, m) for (t, m), r in g26.items() if r["pct"] == 100.0 and r["valid"] >= 5]
cells.sort()
for (t, m) in cells:
    a = gall.get((t, m))
    s = f"{a['reached']}/{a['valid']}={a['pct']}%" if a else "n/a"
    out.append(f"  T={t:>3} M={m}: 2026 {g26[(t,m)]['reached']}/{g26[(t,m)]['valid']}=100% | all {s}")

sys.stdout.buffer.write("\n".join(out).encode("utf-8"))
