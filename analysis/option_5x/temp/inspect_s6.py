import json, sys
from pathlib import Path
p = Path(__file__).resolve().parents[1] / "output" / "strategy6_v01_pair.json"
d = json.loads(p.read_text(encoding="utf-8"))
daily = d["daily"]
print("SUMMARY"); 
for v,s in d["summary"].items(): print(" ",v,s)
# top/bottom by time variant
rows = [(x["date"], x["status"], round(x["pnl"]["time"]), round(x["pnl"]["x2"]), round(x["invested"]), len(x["positions"])) for x in daily]
rows.sort(key=lambda r: r[2])
print("\n-- BOTTOM 6 (time) --")
for r in rows[:6]: print(r)
print("\n-- TOP 6 (time) --")
for r in rows[-6:]: print(r)
# detail of biggest winner
big = max(daily, key=lambda x: x["pnl"]["time"])
print("\nBIGGEST WIN DAY", big["date"], big["status"])
for pos in big["positions"]:
    print("  ", pos)
# count positive days
pos_time = sum(1 for x in daily if x["pnl"]["time"]>0)
print("\npositive days time:", pos_time, "of 67; median time pnl:", sorted(r[2] for r in rows)[len(rows)//2])
