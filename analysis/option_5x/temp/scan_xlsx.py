from pathlib import Path
import collections, re
import openpyxl

p = Path(r"c:\project\option\backend\rowdata\코스피200_위클리옵션_2026년10월_1분.xlsx")
wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
SERIES_RE = re.compile(r"\b(\d{4}W\d)\b")

name_prefix = collections.Counter()   # token before series
series_set = collections.Counter()
market = collections.Counter()
cp = collections.Counter()
has_M = 0
time_min = {}  # per market, min/max 기준시각 for DRV
name_samples = set()
code_by_series = collections.defaultdict(set)

for sheet in wb.sheetnames:
    if sheet == "자료설명":
        continue
    ws = wb[sheet]
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue
        date, mkt, code, name, t, exp, callput, strike, *_ = row
        market[mkt]+=1
        cp[callput]+=1
        m = SERIES_RE.search(name or "")
        ser = m.group(1) if m else "?"
        series_set[ser]+=1
        if m:
            prefix = name[:m.start()].strip()
            name_prefix[prefix]+=1
        if name and ("위클리M" in name or "코스피위클리M" in name):
            has_M+=1
        if "M" in (name or "")[:8]:
            name_samples.add(name)
        code_by_series[ser].add(code)
wb.close()

print("MARKET:", dict(market))
print("CP:", dict(cp))
print("name prefixes:", dict(name_prefix))
print("series:", dict(sorted(series_set.items())))
print("names containing M (위클리M):", has_M)
print("sample names with 'M' in first 8 chars:", list(name_samples)[:10])
print("\ncodes per series (count distinct):")
for s,codes in sorted(code_by_series.items()):
    print(" ", s, len(codes), list(codes)[:2])
