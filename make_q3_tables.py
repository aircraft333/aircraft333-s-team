# -*- coding: utf-8 -*-
r"""从 result3.xlsx 生成问题三的三张论文表（LaTeX 形式）

对应题目原文：
    表1 微网在指定时间段的购电量及全天的购电量和购电费
    表2 储能设备在指定时间段的充放电量及 0:00 和 24:00 的储电量
    表3 微网在指定日期的紧急购电量
    「请在论文中按表 1、表 2 和表 3 的格式给出表 3 中指定日期的结果」

关键口径（本题的关键设计）：
  · 三张表都给出**表3 中指定的 4 个日期**（3-20 / 6-21 / 9-23 / 12-21），每个日期一列。
  · 表3 的「时间段」不再逐条罗列实际发生的零散区间（如 09:30-10:00），
    而是**沿用表1/表2 的时段划分**，保证三张表的行标签互相对应、表格整齐。

数据来源：直接读 result3.xlsx（q3.py 的正式交付物），
        因此数值与结果文件天然一致，不需要重新求解 LP。
输出：问题三_表格.tex（可直接 \input 进论文）
"""
import os
import re

import numpy as np
import openpyxl

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "result3.xlsx")
OUT = os.path.join(ROOT, "问题三_表格.tex")

DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
DATES_TEX = ["2025.3.20", "2025.6.21", "2025.9.23", "2025.12.21"]
OUT_START = "2025-02-01"

# 表1 的 6 个代表时段（题目指定）：10 分钟粒度，槽号 t 覆盖 [10t, 10t+10] 分钟
REP_T = [60, 72, 84, 96, 108, 120]
REP_LAB = ["10:00-10:10", "12:00-12:10", "14:00-14:10",
           "16:00-16:10", "18:00-18:10", "20:00-20:10"]

# 表2 / 表3 的 6 个 4 小时时段（题目指定）
BLK = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]
BLK_LAB = ["0:00-4:00", "4:00-8:00", "8:00-12:00",
           "12:00-16:00", "16:00-20:00", "20:00-24:00"]
BLK_H = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]

N_DEC = 4                       # 小数位数，与论文中表1/表2 的既有风格一致


def f4(x):
    return f"{float(x):.{N_DEC}f}"


# =====================================================================
# 一、读 result3.xlsx
# =====================================================================
wb = openpyxl.load_workbook(SRC)
days = [(d - np.datetime64(OUT_START)).astype(int)
        for d in np.array(DATES, dtype="datetime64[D]")]
row_of = [2 + d for d in days]                    # 计划/调整表：第 i 天在第 i+2 行

ws_p = wb["计划购电量"]
ws_a = wb["调整购电量"]
bp = np.array([[ws_p.cell(row=r, column=2 + t).value for t in range(144)]
               for r in row_of], float)
ba = np.array([[ws_a.cell(row=r, column=2 + t).value for t in range(144)]
               for r in row_of], float)
plan_tot = [float(ws_p.cell(row=r, column=146).value) for r in row_of]
adj_tot = [float(ws_a.cell(row=r, column=146).value) for r in row_of]
plan_fee = [float(ws_p.cell(row=r, column=147).value) for r in row_of]
adj_fee = [float(ws_a.cell(row=r, column=147).value) for r in row_of]

# 充放电量：第 i 天占 2+6i .. 2+6i+5 行，col3=充电 col4=放电 col5=时刻 col6=储电量
ws_c = wb["充放电量"]
chg = np.zeros((4, 6))
dis = np.zeros((4, 6))
E0 = np.zeros(4)
E24 = np.zeros(4)
for k, d in enumerate(days):
    r0 = 2 + 6 * d
    for j in range(6):
        chg[k, j] = float(ws_c.cell(row=r0 + j, column=3).value or 0.0)
        dis[k, j] = float(ws_c.cell(row=r0 + j, column=4).value or 0.0)
    E0[k] = float(ws_c.cell(row=r0, column=6).value or 0.0)
    E24[k] = float(ws_c.cell(row=r0 + 1, column=6).value or 0.0)

# 紧急购电量：逐行读，col1=日期（仅首条有值则向下继承）col2=区间 col3=电量
ws_e = wb["紧急购电量"]
emg = np.zeros((4, 6))
cur = None
for r in range(2, ws_e.max_row + 1):
    dv = ws_e.cell(row=r, column=1).value
    if dv is not None:
        cur = dv.date().isoformat() if hasattr(dv, "date") else str(dv)[:10]
    if cur not in DATES:
        continue
    seg = ws_e.cell(row=r, column=2).value
    amt = ws_e.cell(row=r, column=3).value
    if not seg or amt is None:
        continue
    k = DATES.index(cur)
    m = re.match(r"(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})", str(seg))
    assert m, "无法解析时间段：%r" % seg
    hh = int(m.group(1))                      # 按起始时刻归入 4 小时时段
    for j, (a, b) in enumerate(BLK_H):
        if a <= hh < b:
            emg[k, j] += float(amt)
            break

# =====================================================================
# 二、自检：与 q3_结果汇总.txt 的文本值逐项比对
# =====================================================================
print("=== 自检（与 q3_结果汇总.txt 的【4】节比对）===")
expect_plan = {"2025-03-20": 67231.55, "2025-06-21": 35068.86,
               "2025-09-23": 67981.24, "2025-12-21": 96126.81}
expect_adj = {"2025-03-20": 67139.02, "2025-06-21": 35225.34,
              "2025-09-23": 67856.04, "2025-12-21": 96109.56}
for k, ds in enumerate(DATES):
    d1 = abs(plan_tot[k] - expect_plan[ds])
    d2 = abs(adj_tot[k] - expect_adj[ds])
    print(f"  {ds}  全天计划 {plan_tot[k]:>12,.2f}（差 {d1:.2f}）"
          f"  全天调整 {adj_tot[k]:>12,.2f}（差 {d2:.2f}）")
    assert d1 < 0.01 and d2 < 0.01, "全天购电量与汇总文本不一致"
k0 = DATES.index("2025-03-20")
print(f"  2025-03-20 12:00-12:10 调整后购电量 = {ba[k0][72]:.2f}（汇总文本 521.27）")
assert abs(ba[k0][72] - 521.27) < 0.01
print(f"  2025-03-20 0:00-4:00 充 {chg[k0][0]:,.2f} / 放 {dis[k0][0]:,.2f}"
      f"（汇总文本 8391.43 / 132.87）")
assert abs(chg[k0][0] - 8391.43) < 0.01 and abs(dis[k0][0] - 132.87) < 0.01
print(f"  2025-03-20 0:00 储电量 {E0[k0]:,.2f} / 24:00 储电量 {E24[k0]:,.2f}"
      f"（汇总文本 1786.79 / 1799.52）")
assert abs(E0[k0] - 1786.79) < 0.01 and abs(E24[k0] - 1799.52) < 0.01
print("  紧急购电（按 4 小时时段归并）：")
for k, ds in enumerate(DATES):
    tot = emg[k].sum()
    nz = ", ".join(f"{BLK_LAB[j]}={emg[k][j]:.2f}" for j in range(6) if emg[k][j] > 0)
    print(f"    {ds}  合计 {tot:>8.2f} kWh   {nz if nz else '无紧急购电'}")
print("  -> 全部通过\n")

# =====================================================================
# 三、生成 LaTeX
# =====================================================================
L = []
A = L.append


def wide(preamble):
    """宽表用 resizebox 压到版心宽度；右花括号由调用处在 \\end{tabular} 之后闭合"""
    return "\\resizebox{\\textwidth}{!}{%\n" + preamble


# ---------------- 表1 ----------------
A("% ---------------- 表1 微网在指定时间段的购电量 ----------------")
A("\\begin{table}[h]")
A("\\centering")
A("\\caption{\\textbf{微网在指定时间段的购电量及全天的购电量和购电费"
  "（问题三，单位：kWh / 元）}}")
A("\\label{tab:q3_buy}")
A(wide("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|}")
      .rstrip("\n"))
A("\\hline")
A("\\multirow{2}{*}{时间段} & \\multicolumn{2}{c|}{" + DATES_TEX[0] + "}"
  " & \\multicolumn{2}{c|}{" + DATES_TEX[1] + "}"
  " & \\multicolumn{2}{c|}{" + DATES_TEX[2] + "}"
  " & \\multicolumn{2}{c|}{" + DATES_TEX[3] + "} \\\\")
A("\\cline{2-9}")
A(" & 计划 & 调整 & 计划 & 调整 & 计划 & 调整 & 计划 & 调整 \\\\")
A("\\hline")
for i, t in enumerate(REP_T):
    cells = []
    for k in range(4):
        cells += [f4(bp[k][t]), f4(ba[k][t])]
    A(f"{REP_LAB[i]} & " + " & ".join(cells) + " \\\\")
    A("\\hline")
cells = []
for k in range(4):
    cells += [f4(plan_tot[k]), f4(adj_tot[k])]
A("\\multicolumn{1}{|c|}{全天购电量} & " + " & ".join(cells) + " \\\\")
A("\\hline")
cells = []
for k in range(4):
    cells += [f4(plan_fee[k]), f4(adj_fee[k])]
A("\\multicolumn{1}{|c|}{全天购电费} & " + " & ".join(cells) + " \\\\")
A("\\hline")
A("\\end{tabular}")
A("}")
A("\\end{table}")
A("")

# ---------------- 表2 ----------------
A("% ---------------- 表2 储能设备在指定时间段的充放电量 ----------------")
A("\\begin{table}[h]")
A("\\centering")
A("\\caption{\\textbf{储能设备在指定时间段的充放电量及 0:00 与 24:00 的储电量"
  "（问题三，单位：kWh）}}")
A("\\label{tab:q3_soc}")
A(wide("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|}").rstrip("\n"))
A("\\hline")
A("\\multirow{2}{*}{时间段} & \\multicolumn{2}{c|}{" + DATES_TEX[0] + "}"
  " & \\multicolumn{2}{c|}{" + DATES_TEX[1] + "}"
  " & \\multicolumn{2}{c|}{" + DATES_TEX[2] + "}"
  " & \\multicolumn{2}{c|}{" + DATES_TEX[3] + "} \\\\")
A("\\cline{2-9}")
A(" & 充电量 & 放电量 & 充电量 & 放电量 & 充电量 & 放电量 & 充电量 & 放电量 \\\\")
A("\\hline")
for i, lab in enumerate(BLK_LAB):
    cells = []
    for k in range(4):
        cells += [f4(chg[k][i]), f4(dis[k][i])]
    A(f"{lab} & " + " & ".join(cells) + " \\\\")
    A("\\hline")
A("0:00 储电量 & \\multicolumn{2}{c|}{" + f4(E0[0]) + "}"
  " & \\multicolumn{2}{c|}{" + f4(E0[1]) + "}"
  " & \\multicolumn{2}{c|}{" + f4(E0[2]) + "}"
  " & \\multicolumn{2}{c|}{" + f4(E0[3]) + "} \\\\")
A("\\hline")
A("24:00 储电量 & \\multicolumn{2}{c|}{" + f4(E24[0]) + "}"
  " & \\multicolumn{2}{c|}{" + f4(E24[1]) + "}"
  " & \\multicolumn{2}{c|}{" + f4(E24[2]) + "}"
  " & \\multicolumn{2}{c|}{" + f4(E24[3]) + "} \\\\")
A("\\hline")
A("\\end{tabular}")
A("}")
A("\\end{table}")
A("")

# ---------------- 表3 ----------------
A("% ---------------- 表3 微网在指定日期的紧急购电量 ----------------")
A("% 行标签沿用表2 的 6 个 4 小时时段，使三张表的时段划分完全一致")
A("\\begin{table}[h]")
A("\\centering")
A("\\caption{\\textbf{微网在指定日期的紧急购电量（问题三，单位：kWh）}}")
A("\\label{tab:q3_emg}")
A("\\begin{tabular}{|c|c|c|c|c|}")
A("\\hline")
A("\\multirow{2}{*}{时间段} & \\multicolumn{1}{c|}{" + DATES_TEX[0] + "}"
  " & \\multicolumn{1}{c|}{" + DATES_TEX[1] + "}"
  " & \\multicolumn{1}{c|}{" + DATES_TEX[2] + "}"
  " & \\multicolumn{1}{c|}{" + DATES_TEX[3] + "} \\\\")
A("\\cline{2-5}")
A(" & 购电量 & 购电量 & 购电量 & 购电量 \\\\")
A("\\hline")
for i, lab in enumerate(BLK_LAB):
    A(f"{lab} & " + " & ".join(f4(emg[k][i]) for k in range(4)) + " \\\\")
    A("\\hline")
A("全天合计 & " + " & ".join(f4(emg[k].sum()) for k in range(4)) + " \\\\")
A("\\hline")
A("\\end{tabular}")
A("\\end{table}")

tex = "\n".join(L) + "\n"
with open(OUT, "w", encoding="utf-8") as f:
    f.write(tex)
print("已写入", OUT)
print()
print(tex)
