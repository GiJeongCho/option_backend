from pathlib import Path
import openpyxl

p = Path(r"c:\project\option\backend\rowdata\코스피200_위클리옵션_2026년10월_1분.xlsx")
wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
print("SHEETS:", wb.sheetnames)
for name in ["20261001", "자료설명"]:
    ws = wb[name]
    print(f"\n=== sheet '{name}' max_row={ws.max_row} max_col={ws.max_column} ===")
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        print(row)
        if i >= 8:
            break
wb.close()
