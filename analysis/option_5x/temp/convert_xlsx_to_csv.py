"""2026년10월 xlsx(시트=날짜)를 기존 KRX CSV 포맷(cp949, 17열)으로 변환.

- 종목명은 그대로 둔다('위클리C'/'위클리M C' → product_kind/series 정규식이 이미 호환).
- 새 데이터셋 폴더(마커 포함)에 `..._YYYYMMDD.csv`로 저장 → inventory 자동 인식.
"""
from pathlib import Path
import csv
import openpyxl

SRC = Path(r"c:\project\option\backend\rowdata\코스피200_위클리옵션_2026년10월_1분.xlsx")
ROWDATA = Path(r"c:\project\option\backend\rowdata")
DEST_DIR = ROWDATA / "[파생 옵션 코스피200 위클리]일중 매매정보(1분)(R2-2026년10월)"
DEST_DIR.mkdir(exist_ok=True)

wb = openpyxl.load_workbook(SRC, read_only=True, data_only=True)
total = 0
for sheet in wb.sheetnames:
    if not (len(sheet) == 8 and sheet.isdigit()):
        print("skip sheet:", sheet)
        continue
    dest = DEST_DIR / f"R2_2026년10월_1분_{sheet}.csv"
    rows_written = 0
    with dest.open("w", encoding="cp949", newline="", errors="strict") as fout:
        writer = csv.writer(fout)
        for row in wb[sheet].iter_rows(values_only=True):
            out = ["" if v is None else v for v in row]
            writer.writerow(out)
            rows_written += 1
    total += rows_written - 1  # minus header
    print(f"{sheet}: {rows_written-1} rows -> {dest.name}")
wb.close()
print("TOTAL data rows:", total)
print("DEST:", DEST_DIR)
