import sys
from pathlib import Path
# backend root on path so `app` package imports
repo_root = Path(__file__).resolve().parents[4]  # c:\project\option
sys.path.insert(0, str(repo_root))
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)
r = client.get("/api/strategy-pair/summary")
print("status", r.status_code)
d = r.json()
print("strategy", d["strategy"], "year", d["year"], "expiry_days", d["expiry_days"],
      "positions", d["total_positions"])
print("branch", d["branch_count"])
print("summary keys", list(d["summary"].keys()))
print("settle", d["summary"]["settle"])
print("by_year keys", list(d["by_year"].keys()))
print("daily len", len(d["daily"]), "first date", d["daily"][0]["date"])
