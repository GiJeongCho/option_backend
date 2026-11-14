import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from count_opportunities import derive_expiry_groups, PROFILE_PATH

profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
groups = derive_expiry_groups(profile)
oct_groups = [g for g in groups if g["series"].startswith("2610") or g["series"] == "2611W1"]
print("total groups:", len(groups))
for g in oct_groups:
    print(g["series"], g["product_kind"], "expiry=", g.get("expiry_date"))

# data_service dates
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from backend.app.data_service import MarketDataService
svc = MarketDataService()
dates = svc.list_option_dates()
octd = [d for d in dates if d.startswith("202610")]
print("data_service total dates:", len(dates), "| Oct dates:", len(octd), octd[:3], octd[-3:])
# sample contract listing on a new date
cons = svc.list_contracts("20261015", "day")
print("20261015 day contracts:", len(cons), "sample:", cons[0]["name"], cons[0]["series"], cons[0]["productKind"])
