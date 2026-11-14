import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import strategy6_v01_pair_half_backtest as s6
from count_opportunities import PROFILE_PATH, derive_expiry_groups, target_file_map
import minimal_baseline_backtest as base

profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
groups = [g for g in derive_expiry_groups(profile) if g["expiry_date"].startswith("2026")]
files = target_file_map()
dates = sorted(files)
print("2026 groups:", len(groups))

# test first 3 groups
def prior_file(g):
    exp = g["expiry_date"]
    for d in reversed([d for d in dates if d < exp]):
        if base.load_expiry_day(files[d], g):
            return files[d]
    return None

for g in groups[:8]:
    exp = g["expiry_date"]
    d = s6.run_day(g, files[exp], prior_file(g))
    print(exp, g["product_kind"], g["series"], "status=", d["status"],
          "callH=", d["call_halved"], "putH=", d["put_halved"],
          "pos=", len(d["positions"]), "invested=", round(d["invested"]),
          "pnl_settle=", round(d["pnl"]["settle"]), "pnl_x2=", round(d["pnl"]["x2"]))
    for p in d["positions"][:4]:
        print("   ", p["call_put"], "P0=", p["p0"], "fillmin=", p["fill_minute"],
              "fill=", p["fill_price"], "qty=", p["qty"])
