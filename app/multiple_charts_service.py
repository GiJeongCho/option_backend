from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Literal

Band = Literal["low", "premium"]
CallPut = Literal["ALL", "CALL", "PUT"]

# 표준 배수 래더(프론트 기본 버튼). 사용자 자유 배수도 허용한다.
STANDARD_RUNGS: tuple[float, ...] = (2.0, 3.0, 5.0, 10.0)

BAND_FILES: dict[str, str] = {
    "low": "opportunities_lowband.csv",
    "premium": "opportunities.csv",
}

BAND_LABELS: dict[str, str] = {
    "low": "0.40~1.20 전구간(S5 v0.7 기반)",
    "premium": "1.00~5.00 프리미엄 구간",
}


class MultipleChartsDataError(LookupError):
    pass


def _to_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None else None


class MultipleChartsService:
    """만기일 당일 옵션 중 특정 배수 이상(또는 정확 구간) 상승한 계약을 모아,
    진입 시점 공통 모양 탐색을 위한 차트 그리드 데이터를 제공한다.

    판정 기준은 기존 count_opportunities 산출물(max_multiple = 진입가 기준
    시간순 최대 배수)을 그대로 재사용하므로, 임의 배수(6·7배 등)도 지원한다.
    원자료는 정적 폴더로 복사하지 않고, 분봉은 기존 /api/options/bars가 담당한다.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        if output_dir is None:
            output_dir = (
                Path(__file__).resolve().parents[1]
                / "analysis"
                / "option_5x"
                / "output"
            )
        self.output_dir = output_dir

    def _path(self, band: str) -> Path:
        file_name = BAND_FILES.get(band)
        if file_name is None:
            raise MultipleChartsDataError(f"알 수 없는 가격대: {band}")
        path = self.output_dir / file_name
        if not path.exists():
            raise MultipleChartsDataError(
                f"기회 데이터 파일을 찾을 수 없습니다: {path.name}"
            )
        return path

    @lru_cache(maxsize=4)
    def _load(self, band: str) -> tuple[dict[str, object], ...]:
        path = self._path(band)
        rows: list[dict[str, object]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            for raw in csv.DictReader(source):
                max_multiple = _to_float(raw.get("max_multiple"))
                if max_multiple is None:
                    continue
                expiry = str(raw.get("expiry_date") or "")
                rows.append(
                    {
                        "expiryDate": expiry,
                        "year": expiry[:4],
                        "code": str(raw.get("option_code") or ""),
                        "name": str(raw.get("option_name") or ""),
                        "callPut": str(raw.get("call_put") or ""),
                        "strike": _to_float(raw.get("strike")),
                        "productKind": str(raw.get("product_kind") or ""),
                        "series": str(raw.get("series") or ""),
                        "maxMultiple": max_multiple,
                        "maxMultipleTime": _to_int(raw.get("max_multiple_time")),
                        "maxMultipleBase": _to_float(
                            raw.get("max_multiple_base")
                        ),
                        "maxMultipleBaseTime": _to_int(
                            raw.get("max_multiple_base_time")
                        ),
                        "firstEligibleLow": _to_float(
                            raw.get("first_eligible_low")
                        ),
                        "firstEligibleTime": _to_int(
                            raw.get("first_eligible_time")
                        ),
                        "base2xTime": _to_int(raw.get("base_time_2x")),
                        "target2xTime": _to_int(raw.get("target_time_2x")),
                        "totalVolume": _to_int(raw.get("total_volume")),
                    }
                )
        rows.sort(key=lambda item: (str(item["expiryDate"]), str(item["code"])))
        return tuple(rows)

    def _filtered(
        self,
        band: str,
        multiple_min: float,
        multiple_max: float | None,
        call_put: str,
        year_from: str | None,
        year_to: str | None,
    ) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for row in self._load(band):
            value = float(row["maxMultiple"])
            if value + 1e-9 < multiple_min:
                continue
            if multiple_max is not None and value >= multiple_max - 1e-9:
                continue
            if call_put != "ALL" and row["callPut"] != call_put:
                continue
            year = str(row["year"])
            if year_from and year < year_from:
                continue
            if year_to and year > year_to:
                continue
            result.append(row)
        return result

    def contracts(
        self,
        multiple_min: float,
        multiple_max: float | None = None,
        call_put: str = "ALL",
        band: str = "low",
        year_from: str | None = None,
        year_to: str | None = None,
        limit: int = 600,
    ) -> dict[str, object]:
        if multiple_min < 1.0:
            raise ValueError("배수는 1.0 이상이어야 합니다.")
        if multiple_max is not None and multiple_max <= multiple_min:
            raise ValueError("상한 배수는 하한보다 커야 합니다.")
        rows = self._filtered(
            band, multiple_min, multiple_max, call_put, year_from, year_to
        )
        # 최신 만기일을 앞쪽에 둬 탐색 편의를 높인다.
        rows.sort(
            key=lambda item: (str(item["expiryDate"]), float(item["maxMultiple"])),
            reverse=True,
        )
        expiry_days = sorted({str(row["expiryDate"]) for row in rows})
        mode = "exact" if multiple_max is not None else "union"
        return {
            "band": band,
            "bandLabel": BAND_LABELS.get(band, band),
            "mode": mode,
            "multipleMin": multiple_min,
            "multipleMax": multiple_max,
            "callPut": call_put,
            "yearFrom": year_from,
            "yearTo": year_to,
            "contractCount": len(rows),
            "expiryDayCount": len(expiry_days),
            "truncated": len(rows) > limit,
            "contracts": rows[:limit],
        }

    def summary(self, band: str = "low") -> dict[str, object]:
        rows = self._load(band)
        years = sorted({str(row["year"]) for row in rows})
        rung_stats: list[dict[str, object]] = []
        rungs = list(STANDARD_RUNGS)
        for index, rung in enumerate(rungs):
            upper = rungs[index + 1] if index + 1 < len(rungs) else None
            union_rows = [
                row for row in rows if float(row["maxMultiple"]) + 1e-9 >= rung
            ]
            exact_rows = [
                row
                for row in union_rows
                if upper is None
                or float(row["maxMultiple"]) < upper - 1e-9
            ]
            rung_stats.append(
                {
                    "multiple": rung,
                    "upperExclusive": upper,
                    "unionContractCount": len(union_rows),
                    "unionExpiryDayCount": len(
                        {str(r["expiryDate"]) for r in union_rows}
                    ),
                    "exactContractCount": len(exact_rows),
                    "exactExpiryDayCount": len(
                        {str(r["expiryDate"]) for r in exact_rows}
                    ),
                }
            )
        return {
            "band": band,
            "bandLabel": BAND_LABELS.get(band, band),
            "totalContracts": len(rows),
            "totalExpiryDays": len({str(r["expiryDate"]) for r in rows}),
            "years": years,
            "standardRungs": rung_stats,
            "bands": [
                {"key": key, "label": BAND_LABELS[key]}
                for key in BAND_FILES
                if (self.output_dir / BAND_FILES[key]).exists()
            ],
        }
