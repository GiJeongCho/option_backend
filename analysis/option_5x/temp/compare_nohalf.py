# -*- coding: utf-8 -*-
import json, sys
from pathlib import Path
d = json.load(open(Path(__file__).resolve().parent.parent/"output"/"verify_rise_heatmap.json", encoding="utf-8"))
K="2"; SC="2026"
def g(sub): return {(r["T"],r["M"]):r for r in d["scopes"][SC][sub][K]}
out=[]
for base_sub,nh_sub in (("combined","combined_nohalf"),("call","call_nohalf")):
    ga,gn=g(base_sub),g(nh_sub)
    ra=sum(r["reached"] for r in ga.values()); va=sum(r["valid"] for r in ga.values())
    rn=sum(r["reached"] for r in gn.values()); vn=sum(r["valid"] for r in gn.values())
    up=down=same=0
    for k in ga:
        a=ga[k]; n=gn.get(k)
        if a["valid"]<5 or not n or n["valid"]<5: continue
        if n["pct"]>a["pct"]+0.05: up+=1
        elif n["pct"]<a["pct"]-0.05: down+=1
        else: same+=1
    out.append(f"== {base_sub} (K=2, 2026) ==")
    out.append(f"  전체    : sum reached/valid = {ra}/{va} = {ra/va*100:.1f}%")
    out.append(f"  nohalf  : sum reached/valid = {rn}/{vn} = {rn/vn*100:.1f}%  (분모 {va}->{vn}, {va-vn}건 제외)")
    out.append(f"  셀비교(valid>=5): 개선 {up} / 악화 {down} / 동일 {same}")
sys.stdout.buffer.write("\n".join(out).encode("utf-8"))
