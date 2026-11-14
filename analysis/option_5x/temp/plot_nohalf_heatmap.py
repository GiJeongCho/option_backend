# -*- coding: utf-8 -*-
"""2026 매수후2배 히트맵: 전체 vs 'P0 대비 절반 이하 하락 없음' 필터."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
d = json.load(open(ROOT / "output" / "verify_rise_heatmap.json", encoding="utf-8"))

T_VALUES = list(range(5, 121, 5))
M_VALUES = [round(1.0 + 0.5 * i, 1) for i in range(9)]
K = 2
SCOPE = "2026"

def matrix(sub):
    g = {(r["T"], r["M"]): r for r in d["scopes"][SCOPE][sub][str(K)]}
    pct = np.full((len(M_VALUES), len(T_VALUES)), np.nan)
    txt = [["" for _ in T_VALUES] for _ in M_VALUES]
    for i, M in enumerate(M_VALUES):
        for j, T in enumerate(T_VALUES):
            r = g.get((T, M))
            if r and r["valid"] > 0:
                pct[i, j] = r["pct"]
                txt[i][j] = f"{r['pct']:.0f}\n{r['reached']}/{r['valid']}"
    return pct, txt

panels = [
    ("combined", "Combined — ALL options"),
    ("combined_nohalf", "Combined — NO drop <=50% of P0"),
    ("call", "Call — ALL options"),
    ("call_nohalf", "Call — NO drop <=50% of P0"),
]

fig, axes = plt.subplots(2, 2, figsize=(26, 13))
cmap = plt.cm.Reds.copy()
cmap.set_bad("#e5e5e5")

for ax, (sub, title) in zip(axes.flat, panels):
    pct, txt = matrix(sub)
    im = ax.imshow(pct, cmap=cmap, vmin=0, vmax=100, aspect="auto", origin="lower")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(T_VALUES)))
    ax.set_xticklabels(T_VALUES, fontsize=8)
    ax.set_yticks(range(len(M_VALUES)))
    ax.set_yticklabels(M_VALUES, fontsize=9)
    ax.set_xlabel("T: minutes after P0 (signal time)", fontsize=10)
    ax.set_ylabel("M: signal multiple (close >= M x P0)", fontsize=10)
    for i in range(len(M_VALUES)):
        for j in range(len(T_VALUES)):
            if txt[i][j]:
                v = pct[i, j]
                ax.text(j, i, txt[i][j], ha="center", va="center",
                        fontsize=6, color="white" if v >= 55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

fig.suptitle(
    "KOSPI200 Weekly Options 2026 (Jan-Sep) — Post-buy 2x reach ratio (K=2)\n"
    "Right column: only options whose low never fell to <=50% of P0 between P0 and the signal",
    fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = ROOT / "output" / "heatmap_2026_nohalf.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("saved:", out)
