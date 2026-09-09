# -*- coding: utf-8 -*-
"""临时探测脚本：读题目 PDF 的第一问 + 打印 4 个附件的结构"""
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

BASE = r"c:\Users\dell\Downloads\56RoZe4zc@c3A\C题"
PDF = os.path.join(BASE, "2023C", "(C050)某商超蔬菜类商品动态定价与补货决策研究.pdf")
DATA = os.path.join(BASE, "data")

print("=" * 30, "PDF 探测", "=" * 30)
text = None
try:
    import fitz  # PyMuPDF
    doc = fitz.open(PDF)
    pages = [doc[i].get_text() for i in range(min(6, doc.page_count))]
    text = "\n".join(pages)
    print("用 PyMuPDF 提取成功，页数:", doc.page_count)
except Exception as e1:
    print("PyMuPDF 失败:", e1)
    try:
        import pdfplumber
        with pdfplumber.open(PDF) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages[:6]]
        text = "\n".join(pages)
        print("用 pdfplumber 提取成功，页数:", len(pdf.pages))
    except Exception as e2:
        print("pdfplumber 失败:", e2)
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(PDF)
            pages = [(p.extract_text() or "") for p in reader.pages[:6]]
            text = "\n".join(pages)
            print("用 PyPDF2 提取成功，页数:", len(reader.pages))
        except Exception as e3:
            print("PyPDF2 失败:", e3)

if text:
    # 打印含“问题/第一问/1.”关键字的附近文本（前 4000 字）
    print("---- PDF 前 4500 字符 ----")
    print(text[:4500])
else:
    print("!!! 没有可用的 pdf 文本提取库")

print("=" * 30, "附件结构", "=" * 30)
import pandas as pd
for i in range(1, 5):
    fn = os.path.join(DATA, f"附件{i}.xlsx")
    if not os.path.exists(fn):
        print(f"附件{i}.xlsx 不存在")
        continue
    try:
        xl = pd.ExcelFile(fn)
        print(f"\n### 附件{i}.xlsx  sheets={xl.sheet_names}")
        for sh in xl.sheet_names:
            df = xl.parse(sh, nrows=6)
            print(f"  -- sheet[{sh}] shape(前6行)={df.shape}")
            print("     列:", list(df.columns))
            print("     样例:")
            print(df.head(3).to_string())
    except Exception as e:
        print(f"附件{i}.xlsx 读取失败: {e}")
