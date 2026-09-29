from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def truth(value: str) -> bool:
    return value.lower() == "true"


def summarize() -> dict[str, object]:
    opportunities = read_csv(OUTPUT_DIR / "opportunities.csv")
    band_details = read_csv(OUTPUT_DIR / "premium_band_details.csv")
    overall = json.loads(
        (OUTPUT_DIR / "opportunity_summary.json").read_text(encoding="utf-8")
    )
    years: list[dict[str, object]] = []

    for year in sorted({row["expiry_date"][:4] for row in opportunities}):
        rows = [
            row
            for row in opportunities
            if row["expiry_date"].startswith(year)
        ]
        expiry_dates = {row["expiry_date"] for row in rows}
        successes_5x = [row for row in rows if truth(row["success_5x"])]
        successes_10x = [row for row in rows if truth(row["success_10x"])]
        details_by_band: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in band_details:
            if (
                row["expiry_date"].startswith(year)
                and truth(row["eligible"])
            ):
                details_by_band[row["premium_band"]].append(row)

        bands: list[dict[str, object]] = []
        for band, details in sorted(details_by_band.items()):
            band_5x = [
                row for row in details if truth(row["success_5x"])
            ]
            band_10x = [
                row for row in details if truth(row["success_10x"])
            ]
            bands.append(
                {
                    "premium_band": band,
                    "eligible_contract_days": len(details),
                    "success_5x_contract_days": len(band_5x),
                    "success_5x_rate_pct": round(
                        len(band_5x) / len(details) * 100, 4
                    ),
                    "expiry_days_with_5x": len(
                        {row["expiry_date"] for row in band_5x}
                    ),
                    "success_10x_contract_days": len(band_10x),
                    "success_10x_rate_pct": round(
                        len(band_10x) / len(details) * 100, 4
                    ),
                    "expiry_days_with_10x": len(
                        {row["expiry_date"] for row in band_10x}
                    ),
                }
            )

        best_band = max(
            bands,
            key=lambda item: (
                int(item["success_5x_contract_days"]),
                float(item["success_5x_rate_pct"]),
            ),
        )
        years.append(
            {
                "year": year,
                "expiry_days": len(expiry_dates),
                "eligible_contract_days": len(rows),
                "success_5x_contract_days": len(successes_5x),
                "success_5x_contract_rate_pct": round(
                    len(successes_5x) / len(rows) * 100, 4
                ),
                "expiry_days_with_5x": len(
                    {row["expiry_date"] for row in successes_5x}
                ),
                "expiry_days_with_5x_rate_pct": round(
                    len({row["expiry_date"] for row in successes_5x})
                    / len(expiry_dates)
                    * 100,
                    4,
                ),
                "success_10x_contract_days": len(successes_10x),
                "success_10x_contract_rate_pct": round(
                    len(successes_10x) / len(rows) * 100, 4
                ),
                "expiry_days_with_10x": len(
                    {row["expiry_date"] for row in successes_10x}
                ),
                "expiry_days_with_10x_rate_pct": round(
                    len({row["expiry_date"] for row in successes_10x})
                    / len(expiry_dates)
                    * 100,
                    4,
                ),
                "best_premium_band": best_band,
                "premium_bands": bands,
            }
        )

    reconciliation = {
        "eligible_contract_days": sum(
            int(item["eligible_contract_days"]) for item in years
        )
        == int(overall["eligible_contract_days"]),
        "success_5x_contract_days": sum(
            int(item["success_5x_contract_days"]) for item in years
        )
        == int(overall["success_5x_contract_days"]),
        "success_10x_contract_days": sum(
            int(item["success_10x_contract_days"]) for item in years
        )
        == int(overall["success_10x_contract_days"]),
        "expiry_days": sum(int(item["expiry_days"]) for item in years)
        == int(overall["expiry_group_count"]),
    }
    if not all(reconciliation.values()):
        raise RuntimeError(f"연도별 합계 검산 실패: {reconciliation}")

    result = {
        "definition": {
            "premium_min": 1.0,
            "premium_max": 5.0,
            "same_bar_low_high_allowed": False,
            "unit": "expiry_date + option_code",
        },
        "years": years,
        "reconciliation": {
            "passed": True,
            **reconciliation,
        },
    }
    (OUTPUT_DIR / "yearly_opportunity_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(), ensure_ascii=False, indent=2))
