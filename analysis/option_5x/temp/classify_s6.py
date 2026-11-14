import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import strategy6_v01_pair_half_backtest as s6
import minimal_baseline_backtest as base
from count_opportunities import PROFILE_PATH, derive_expiry_groups, target_file_map

profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
groups = [g for g in derive_expiry_groups(profile) if g["expiry_date"].startswith("2026")]
files = target_file_map()

cnt = collections.Counter()
rows = []
for g in groups:
    exp = g["expiry_date"]
    contracts = base.load_expiry_day(files[exp], g)
    enriched = {}
    for code, c in contracts.items():
        res = s6.p0_of(c)
        if res:
            enriched[code] = (code, c, res[0], res[1])
    def side(cp):
        it = [v for v in enriched.values() if v[1].call_put==cp and s6.BUY_LOW<=v[2]<=s6.BUY_HIGH]
        it.sort(key=lambda v: v[2]); return it
    calls, puts = side("CALL"), side("PUT")
    def refhalf(it):
        if not it: return None,None
        code,c,p0,pidx = it[0]
        return p0, s6.half_index(c,p0,pidx)
    cp0,ch = refhalf(calls); pp0,ph = refhalf(puts)
    chd, phd = ch is not None, ph is not None
    if chd and phd: st="both_halved"
    elif chd: st="call_halved_buy_put"
    elif phd: st="put_halved_buy_call"
    else: st="neither"
    cnt[st]+=1
    # half minute
    cm = base.SESSION_MINUTES[ch] if ch is not None else None
    pm = base.SESSION_MINUTES[ph] if ph is not None else None
    rows.append((exp, st, len(calls), len(puts), cm, pm))

print("STATUS", dict(cnt))
for r in rows:
    print(r)
