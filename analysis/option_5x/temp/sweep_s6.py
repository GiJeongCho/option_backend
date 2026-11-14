import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import strategy6_v01_pair_half_backtest as s6
import minimal_baseline_backtest as base
from count_opportunities import PROFILE_PATH, derive_expiry_groups, target_file_map

profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
groups = [g for g in derive_expiry_groups(profile) if g["expiry_date"].startswith("2026")]
files = target_file_map()
OBS = s6.OBS_END_IDX

def half_t(contract, p0, field, cutoff_idx=None):
    stop = 0.5*p0
    end = cutoff_idx if cutoff_idx is not None else len(contract.bars)-1
    for idx in range(OBS+1, end+1):
        b = contract.bars[idx]
        if not b.traded: continue
        v = getattr(b, field)
        if v is not None and v <= stop+1e-9:
            return idx
    return None

# preload candidates per day
days = []
for g in groups:
    exp = g["expiry_date"]
    contracts = base.load_expiry_day(files[exp], g)
    enr = []
    for code,c in contracts.items():
        r = s6.p0_of(c)
        if r and s6.BUY_LOW<=r[0]<=1.20:
            enr.append((code,c,r[0],r[1]))
    calls=[v for v in enr if v[1].call_put=="CALL"]
    puts=[v for v in enr if v[1].call_put=="PUT"]
    days.append((exp,calls,puts))

total_cand = sum(len(c)+len(p) for _,c,p in days)
print("total candidates (0.4-1.2):", total_cand, "days:", len(days))

def side_half(items, field, rep, cutoff):
    if not items: return None
    if rep=="cheapest":
        items2=[min(items,key=lambda v:v[2])]
    elif rep=="atm":
        items2=[max(items,key=lambda v:v[2])]
    else:
        items2=items
    times=[half_t(c,p0,field,cutoff) for _,c,p0,_ in items2]
    times=[t for t in times if t is not None]
    return min(times) if times else None

for cut_min in (930, 1000, 1030, 1100):
    cutoff = base.MINUTE_INDEX[cut_min]
    for field in ("close","low"):
        for rep in ("cheapest","atm","any"):
            both=0; cbp=0; pbc=0; neither=0
            for exp,calls,puts in days:
                ch=side_half(calls,field,rep,cutoff); ph=side_half(puts,field,rep,cutoff)
                if ch is not None and ph is not None: both+=1
                elif ch is not None: cbp+=1
                elif ph is not None: pbc+=1
                else: neither+=1
            print(f"cut={cut_min} field={field:5s} rep={rep:8s} both={both:2d} callH->put={cbp:2d} putH->call={pbc:2d} neither={neither:2d}")
    print()
