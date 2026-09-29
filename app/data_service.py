from __future__ import annotations

import csv
import re
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

from openpyxl import load_workbook


Session = Literal["day", "night"]

DATE_PATTERN = re.compile(r"^\d{8}$")
FILE_DATE_PATTERN = re.compile(r"_(\d{8})\.csv$")
SERIES_PATTERN = re.compile(r"\b(\d{4}W\d)\b")
KST = timezone(timedelta(hours=9))
SESSION_MARKET_ID: dict[Session, str] = {"day": "DRV", "night": "NDV"}


class DataNotFoundError(LookupError):
    pass


def _to_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str | None) -> int:
    number = _to_float(value)
    return int(number) if number is not None else 0


def _series(name: str) -> str:
    match = SERIES_PATTERN.search(name)
    return match.group(1) if match else ""


def _product_kind(name: str) -> str:
    return "MONDAY" if "위클리M" in name else "THURSDAY"


def _session_sort_key(hhmm: str, session: Session) -> int:
    value = int(hhmm.strip().zfill(4))
    if session == "night" and value >= 1800:
        return value - 2400
    return value


class MarketDataService:
    def __init__(self, backend_dir: Path | None = None) -> None:
        self.backend_dir = backend_dir or Path(__file__).resolve().parents[1]
        self.rowdata_dir = self.backend_dir / "rowdata"
        self.volatility_path = self.rowdata_dir / "코스피200 변동성지수.xlsx"

    @lru_cache(maxsize=1)
    def option_file_index(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        if not self.rowdata_dir.exists():
            return index

        for directory in self.rowdata_dir.iterdir():
            if not directory.is_dir() or "일중 매매정보(1분)" not in directory.name:
                continue
            for path in directory.glob("*.csv"):
                match = FILE_DATE_PATTERN.search(path.name)
                if match is None:
                    continue
                trading_date = match.group(1)
                if trading_date in index:
                    raise RuntimeError(f"중복 옵션 파일 날짜: {trading_date}")
                index[trading_date] = path
        return index

    def list_option_dates(self) -> list[str]:
        return sorted(self.option_file_index())

    def _option_path(self, trading_date: str) -> Path:
        if DATE_PATTERN.fullmatch(trading_date) is None:
            raise ValueError("날짜는 YYYYMMDD 형식이어야 합니다.")
        try:
            return self.option_file_index()[trading_date]
        except KeyError as error:
            raise DataNotFoundError(
                f"{trading_date} 옵션 파일을 찾을 수 없습니다."
            ) from error

    @lru_cache(maxsize=32)
    def list_contracts(
        self, trading_date: str, session: Session = "day"
    ) -> list[dict[str, object]]:
        path = self._option_path(trading_date)
        market_id = SESSION_MARKET_ID[session]
        contracts: dict[str, dict[str, object]] = {}

        with path.open("r", encoding="cp949", newline="") as source:
            for row in csv.DictReader(source):
                if row["시장ID"] != market_id:
                    continue

                open_price = _to_float(row["시가"])
                high = _to_float(row["고가"])
                low = _to_float(row["저가"])
                close = _to_float(row["종가"])
                if None in (open_price, high, low, close):
                    continue

                code = row["종목코드"]
                normalized_time = row["기준시각"].strip().zfill(4)
                sort_key = _session_sort_key(normalized_time, session)
                current = contracts.get(code)
                if current is None:
                    current = {
                        "code": code,
                        "name": row["종목명"],
                        "series": _series(row["종목명"]),
                        "productKind": _product_kind(row["종목명"]),
                        "callPut": row["콜풋구분"],
                        "strike": _to_float(row["행사가격"]),
                        "barCount": 0,
                        "totalVolume": 0,
                        "minLow": low,
                        "maxHigh": high,
                        "firstOpen": open_price,
                        "lastClose": close,
                        "firstTime": normalized_time,
                        "lastTime": normalized_time,
                        "touchedPremium1To5": False,
                        "_firstSortKey": sort_key,
                        "_lastSortKey": sort_key,
                    }
                    contracts[code] = current

                current["barCount"] = int(current["barCount"]) + 1
                current["totalVolume"] = int(current["totalVolume"]) + _to_int(
                    row["거래량"]
                )
                current["minLow"] = min(float(current["minLow"]), low)
                current["maxHigh"] = max(float(current["maxHigh"]), high)

                if sort_key < int(current["_firstSortKey"]):
                    current["_firstSortKey"] = sort_key
                    current["firstTime"] = normalized_time
                    current["firstOpen"] = open_price
                if sort_key >= int(current["_lastSortKey"]):
                    current["_lastSortKey"] = sort_key
                    current["lastTime"] = normalized_time
                    current["lastClose"] = close
                if 1.0 <= low <= 5.0:
                    current["touchedPremium1To5"] = True

        public_contracts = [
            {
                key: value
                for key, value in contract.items()
                if not key.startswith("_")
            }
            for contract in contracts.values()
        ]
        return sorted(
            public_contracts,
            key=lambda item: (
                str(item["productKind"]),
                str(item["callPut"]),
                float(item["strike"] or 0),
            ),
        )

    @staticmethod
    def _bar_timestamp(
        trading_date: str, hhmm: str, session: Session
    ) -> tuple[int, str, int]:
        trading_day = datetime.strptime(trading_date, "%Y%m%d").date()
        normalized = hhmm.strip().zfill(4)
        hour = int(normalized[:2])
        minute = int(normalized[2:])

        calendar_day = trading_day
        sort_key = hour * 100 + minute
        if session == "night" and hour >= 18:
            calendar_day -= timedelta(days=1)
            sort_key -= 2400

        moment = datetime.combine(calendar_day, time(hour, minute), tzinfo=KST)
        return int(moment.timestamp()), moment.strftime("%m-%d %H:%M"), sort_key

    @lru_cache(maxsize=128)
    def option_bars(
        self, trading_date: str, code: str, session: Session = "day"
    ) -> list[dict[str, object]]:
        path = self._option_path(trading_date)
        market_id = SESSION_MARKET_ID[session]
        bars: dict[int, dict[str, object]] = {}

        with path.open("r", encoding="cp949", newline="") as source:
            for row in csv.DictReader(source):
                if row["시장ID"] != market_id or row["종목코드"] != code:
                    continue

                open_price = _to_float(row["시가"])
                high = _to_float(row["고가"])
                low = _to_float(row["저가"])
                close = _to_float(row["종가"])
                if None in (open_price, high, low, close):
                    continue

                timestamp, label, sort_key = self._bar_timestamp(
                    trading_date, row["기준시각"], session
                )
                current = bars.get(timestamp)
                if current is None:
                    bars[timestamp] = {
                        "time": timestamp,
                        "label": label,
                        "sortKey": sort_key,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": _to_int(row["거래량"]),
                    }
                    continue

                current["high"] = max(float(current["high"]), high)
                current["low"] = min(float(current["low"]), low)
                current["close"] = close
                current["volume"] = int(current["volume"]) + _to_int(row["거래량"])

        if not bars:
            raise DataNotFoundError(
                f"{trading_date} {code} {session} 분봉을 찾을 수 없습니다."
            )
        return sorted(bars.values(), key=lambda item: int(item["time"]))

    @lru_cache(maxsize=128)
    def option_bar_grid(
        self, trading_date: str, code: str
    ) -> list[dict[str, object]]:
        actual = self.option_bars(trading_date, code, "day")
        by_minute = {int(row["sortKey"]): row for row in actual}
        result: list[dict[str, object]] = []
        previous_close: float | None = None
        for total in range(8 * 60 + 45, 15 * 60 + 20):
            hour, minute = divmod(total, 60)
            hhmm = hour * 100 + minute
            row = by_minute.get(hhmm)
            if row is not None:
                current = dict(row)
                current["synthetic"] = False
                result.append(current)
                previous_close = float(row["close"])
                continue
            if previous_close is None:
                continue
            timestamp, label, sort_key = self._bar_timestamp(
                trading_date, f"{hhmm:04d}", "day"
            )
            result.append(
                {
                    "time": timestamp,
                    "label": label,
                    "sortKey": sort_key,
                    "open": previous_close,
                    "high": previous_close,
                    "low": previous_close,
                    "close": previous_close,
                    "volume": 0,
                    "synthetic": True,
                }
            )
        return result

    @lru_cache(maxsize=1)
    def volatility_bars(self) -> list[dict[str, object]]:
        if not self.volatility_path.exists():
            raise DataNotFoundError(
                f"변동성지수 파일을 찾을 수 없습니다: {self.volatility_path}"
            )

        workbook = load_workbook(
            self.volatility_path, read_only=True, data_only=True
        )
        try:
            sheet = workbook["코스피200 변동성지수"]
            result: list[dict[str, object]] = []
            for row in sheet.iter_rows(min_row=2, max_col=5, values_only=True):
                raw_date, raw_open, raw_high, raw_low, raw_close = row
                if raw_date is None or None in (
                    raw_open,
                    raw_high,
                    raw_low,
                    raw_close,
                ):
                    continue

                if isinstance(raw_date, datetime):
                    trading_day = raw_date.date()
                elif isinstance(raw_date, date):
                    trading_day = raw_date
                else:
                    text = str(raw_date).strip()
                    parsed = None
                    for date_format in ("%Y-%m-%d", "%Y%m%d", "%Y.%m.%d"):
                        try:
                            parsed = datetime.strptime(text, date_format).date()
                            break
                        except ValueError:
                            continue
                    if parsed is None:
                        continue
                    trading_day = parsed

                result.append(
                    {
                        "date": trading_day.isoformat(),
                        "open": float(raw_open),
                        "high": float(raw_high),
                        "low": float(raw_low),
                        "close": float(raw_close),
                    }
                )
            return sorted(result, key=lambda item: str(item["date"]))
        finally:
            workbook.close()
