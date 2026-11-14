import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inventory import option_file_index
import minimal_baseline_backtest as base
from count_opportunities import product_kind, extract_series

files = option_file_index()
oct_dates = sorted(d for d in files if d.startswith("202610"))
print("total dates:", len(files), "| Oct dates:", len(oct_dates))
print("first/last Oct:", oct_dates[0], oct_dates[-1])

# parse one Oct day, filter DRV, pick a Thursday series group and load
path = files["20261030"]
import csv, collections
kinds = collections.Counter(); series = collections.Counter(); mkt = collections.Counter()
with path.open(encoding="cp949") as f:
    for row in csv.DictReader(f):
        mkt[row["시장ID"]] += 1
        if row["시장ID"] != "DRV": continue
        kinds[product_kind(row["종목명"])] += 1
        series[extract_series(row["종목명"])] += 1
print("20261030 market:", dict(mkt))
print("20261030 DRV kinds:", dict(kinds))
print("20261030 DRV series:", dict(series))

# try base.load_expiry_day for a THURSDAY series present on that day
ser = max(series, key=series.get)
grp = {"product_kind": product_kind("위클리C "+ser+" 1,000.0") if False else "THURSDAY", "series": ser}
# determine kind of that series by sampling
grids = base.load_expiry_day(path, {"product_kind": "THURSDAY", "series": ser})
print(f"load THURSDAY {ser}: {len(grids)} contracts")
grids_m = base.load_expiry_day(path, {"product_kind": "MONDAY", "series": ser})
print(f"load MONDAY {ser}: {len(grids_m)} contracts")
if grids:
    code = next(iter(grids))
    g = grids[code]
    traded = sum(1 for b in g.bars if b.traded)
    print("sample contract", code, g.call_put, "name", g.name, "traded bars", traded)
