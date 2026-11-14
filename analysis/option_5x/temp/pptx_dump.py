# -*- coding: utf-8 -*-
import sys
from pptx import Presentation

path = r"C:\project\option\backend\docs\의뢰_row\맹박사님분석\위클리옵션_만기일_가격경로_연구결과_2026_최종보완본3_연도별분석추가_흰색배경수정본.pptx"
p = Presentation(path)
out = []
kw = ("1,320", "633", "반등", "절반", "4분의", "순서", "30분", "282", "미관측", "비정상")
for i, s in enumerate(p.slides):
    texts = []
    for sh in s.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            texts.append(sh.text_frame.text.strip())
    joined = "\n".join(texts)
    if any(k in joined for k in kw):
        out.append(f"\n########## SLIDE {i} ##########\n{joined}")
sys.stdout.buffer.write("\n".join(out).encode("utf-8"))
