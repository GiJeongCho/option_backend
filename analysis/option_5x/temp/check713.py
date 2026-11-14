import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from count_opportunities import target_file_map
files = target_file_map()
path = files["20260713"]
codes = {"KR4CAFBMA496","KR4CAFBMA413","KR4CAFBMA454"}
data = {c: [] for c in codes}
with path.open(encoding="cp949") as f:
    for row in csv.DictReader(f):
        if row["시장ID"]!="DRV": continue
        if row["종목코드"] in codes:
            data[row["종목코드"]].append((row["기준시각"], row["종목명"], row["콜풋구분"], row["행사가격"], row["시가"],row["고가"],row["저가"],row["종가"],row["거래량"]))
for c in codes:
    rows = data[c]
    if not rows: 
        print(c, "NO DATA"); continue
    name = rows[0][1]; strike=rows[0][3]
    closes = [float(r[7]) for r in rows]
    highs = [float(r[5]) for r in rows]
    print(f"\n{c} {name} strike={strike} bars={len(rows)}")
    print("  first3:", rows[:3])
    print("  last3:", rows[-3:])
    print("  min_close=%.2f max_close=%.2f max_high=%.2f"%(min(closes),max(closes),max(highs)))
