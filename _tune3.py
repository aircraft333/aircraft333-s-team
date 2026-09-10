# -*- coding: utf-8 -*-
"""临时：在全年的两个不同日期区块上交叉验证 hedge，并直接跑全年"""
import time

import q3
from config import *

A1 = att1_arrays(load_att1())
pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
dates, L, G = load_att2()
df3 = load_att3()
DEC = (0, 6, 12, 18)
MIX, ADAPT = 0.4, 0.2


def block(i0, i1, hedge):
    gf = q3.build_pv_forecast(df3, dates, G, DEC, MIX)
    lf = q3.build_load_forecast(L, L1, dates, DEC, q3.LOAD_LAG, ADAPT)
    E, ct, ce = SOC0, 0.0, 0.0
    for i in range(i0, i1):
        r = q3.run_day(pi, lf[:, i], gf[:, i], L[i], G[i], E, DEC, hedge)
        E = r["E24"]
        ct += r["cost_dev"]
        ce += r["cost_emg"]
    return ct + ce


log("=== 交叉验证：不同区块上 hedge 的日均费用（元/天）===")
blocks = [(120, 190, "3-5月"), (250, 320, "7-9月"), (330, 365, "11-12月")]
for i0, i1, nm in blocks:
    row = []
    for h in (0.0, 200.0, 300.0, 400.0):
        t = time.time()
        tot = block(i0, i1, h)
        row.append(f"h={h:4.0f}: {tot / (i1 - i0):8.0f} ({time.time() - t:.0f}s)")
    log(f"  {nm}({i0}~{i1})  " + "   ".join(row))
