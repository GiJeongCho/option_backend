from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from backend.analysis.option_5x.inventory import (
    DATASET_MARKER,
    dataset_directories,
    option_file_index,
)
from backend.app.data_service import MarketDataService
from backend.app.index_minute_catalog import IndexMinuteCatalog


def _dataset(root: Path, order: str) -> Path:
    path = root / f"{DATASET_MARKER}(주문번호-{order})"
    path.mkdir(parents=True)
    return path


def _option_file(directory: Path, date: str) -> Path:
    path = directory / f"{directory.name}_{date}.csv"
    path.write_text("", encoding="cp949")
    return path


class OptionDataCatalogTest(unittest.TestCase):
    def test_new_order_directory_is_discovered_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rowdata = Path(temporary)
            old = _dataset(rowdata, "4831-2")
            new = _dataset(rowdata, "4949-1")
            unrelated = rowdata / "기타 데이터"
            unrelated.mkdir()
            _option_file(old, "20260831")
            september = _option_file(new, "20260901")
            _option_file(unrelated, "20260902")

            self.assertEqual(len(dataset_directories(rowdata)), 2)
            self.assertEqual(
                option_file_index(rowdata),
                {
                    "20260831": next(old.glob("*.csv")),
                    "20260901": september,
                },
            )

    def test_duplicate_trading_date_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rowdata = Path(temporary)
            _option_file(_dataset(rowdata, "old"), "20260901")
            _option_file(_dataset(rowdata, "new"), "20260901")

            with self.assertRaisesRegex(ValueError, "중복 거래일"):
                option_file_index(rowdata)

    def test_market_service_refreshes_dates_without_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = Path(temporary)
            rowdata = backend / "rowdata"
            dataset = _dataset(rowdata, "future")
            _option_file(dataset, "20260901")
            service = MarketDataService(backend)

            self.assertEqual(service.list_option_dates(), ["20260901"])
            _option_file(dataset, "20260902")
            self.assertEqual(
                service.list_option_dates(),
                ["20260901", "20260902"],
            )

    def test_market_context_aligns_indices_and_uses_prior_vix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = Path(temporary)
            rowdata = backend / "rowdata"
            rowdata.mkdir()
            header = "기본 주가 데이터,,,,\n날짜,시가,고가,저가,종가\n"
            (rowdata / "코스피 200현물지수_1분봉.csv").write_text(
                header
                + "2026-09-02 09:01,101,101,101,101\n"
                + "2026-09-01 09:02,100,102,99,101\n"
                + "2026-09-01 09:01,100,100,100,100\n"
                + "2026-08-31 15:20,99,99,99,99\n",
                encoding="utf-8-sig",
            )
            (rowdata / "코스피 200선물지수_1분봉.csv").write_text(
                header
                + "2026-09-02 09:01,104,104,104,104\n"
                + "2026-09-01 09:02,103,104,102,103\n"
                + "2026-09-01 09:01,102,102,102,102\n"
                + "2026-08-31 15:20,101,101,101,101\n",
                encoding="utf-8-sig",
            )
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "코스피200 변동성지수"
            sheet.append(["날짜", "시가", "고가", "저가", "종가"])
            for day, close in (
                ("2026-08-26", 20),
                ("2026-08-27", 21),
                ("2026-08-28", 22),
                ("2026-08-31", 23),
                ("2026-09-01", 99),
            ):
                sheet.append([day, close, close, close, close])
            workbook.save(rowdata / "코스피200 변동성지수.xlsx")

            context = MarketDataService(backend).market_context(
                "20260901",
                902,
            )

            self.assertEqual(len(context["spot"]), 2)
            self.assertEqual(len(context["futures"]), 2)
            self.assertEqual(context["basis"][0]["value"], 2)
            self.assertEqual(context["spot"][0]["normalizedClose"], 100)
            self.assertEqual(context["futures"][0]["normalizedClose"], 100)
            self.assertEqual(context["vix"]["knownThrough"], "2026-08-31")
            self.assertEqual(context["vix"]["snapshot"]["close"], 23)

    def test_index_xlsx_precedes_csv_and_csv_extends_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = Path(temporary)
            rowdata = backend / "rowdata"
            rowdata.mkdir()
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "1분봉통합"
            sheet.append(["분봉시작시간", "시가", "고가", "저가", "종가"])
            sheet.append(
                ["2023-08-01 09:01", 100, 101, 99, 100.5]
            )
            workbook.save(
                rowdata / "코스피200_현물지수_1분_20230801_20260930.xlsx"
            )
            (
                rowdata / "코스피 200현물지수_1분봉.csv"
            ).write_text(
                "기본 주가 데이터,,,,\n"
                "날짜,시가,고가,저가,종가\n"
                "2026-10-01 09:01,110,111,109,110.5\n"
                "2023-08-01 09:01,999,999,999,999\n",
                encoding="utf-8-sig",
            )

            catalog = IndexMinuteCatalog(rowdata)
            metadata = catalog.metadata("spot")
            service = MarketDataService(backend)
            old_bar = service.market_index_bars("20230801", "spot")[0]
            new_bar = service.market_index_bars("20261001", "spot")[0]

            self.assertEqual(old_bar["close"], 100.5)
            self.assertEqual(new_bar["close"], 110.5)
            self.assertEqual(metadata["overlapRows"], 1)
            self.assertEqual(metadata["legacyCsvFilledRows"], 1)
            self.assertEqual(metadata["mergedRows"], 2)


if __name__ == "__main__":
    unittest.main()
