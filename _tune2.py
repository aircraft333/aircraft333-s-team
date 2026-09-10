# -*- coding: utf-8 -*-
"""临时：细扫安全裕量与负载自适应（裕量现在同时作用于计划与调整两个阶段）"""
import time

import q3
from config import *

I0, I1 = 120, 190
A1 = att1_arrays(load_att1())
pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
dates, L, G = load_att2()
df3 = load_att3()
DEC = (0, 6, 12, 18)


def block(mix, hedge, adapt, dec=DEC):
    gf = q3.build_pv_forecast(df3, dates, G, dec, mix)
    lf = q3.build_load_forecast(L, L1, dates, dec, q3.LOAD_LAG, adapt)
    E, ct, ce, qe, qb = SOC0, 0.0, 0.0, 0.0, 0.0
    for i in range(I0, I1):
        r = q3.run_day(pi, lf[:, i], gf[:, i], L[i], G[i], E, dec, hedge)
        E = r["E24"]
        ct += r["cost_dev"]
        ce += r["cost_emg"]
        qe += r["e"].sum() * DT_H
        qb += r["b_adj"].sum() * DT_H
    return ct + ce, ce, qe, qb


log(f"=== 细扫 hedge（mix=0.4, adapt=0.2, 区块 {I0}~{I1}）===")
best = (None, 1e18)
for h in [0.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 600.0]:
    t = time.time()
    tot, ce, qe, qb = block(0.4, h, 0.2)
    mk = ""
    if tot < best[1]:
        best = (h, tot)
        mk = "  <="
    log(f"  hedge={h:6.0f}  总费用 {tot:>13,.1f}  紧急费 {ce:>11,.1f}  "
        f"紧急量 {qe:>10,.1f}  购电量 {qb:>12,.1f} kWh  ({time.time() - t:.0f}s){mk}")
log(f"  => hedge 最优 {best[0]}  费用 {best[1]:,.1f}")

log(f"\n=== 细扫 adapt（mix=0.4, hedge={best[0]}）===")
best2 = (None, 1e18)
for a in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
    t = time.time()
    tot, ce, qe, qb = block(0.4, best[0], a)
    mk = ""
    if tot < best2[1]:
        best2 = (a, tot)
        mk = "  <="
    log(f"  adapt={a:5.2f}  总费用 {tot:>13,.1f}  紧急费 {ce:>11,.1f}  "
        f"紧急量 {qe:>10,.1f}  ({time.time() - t:.0f}s){mk}")
log(f"  => adapt 最优 {best2[0]}  费用 {best2[1]:,.1f}")
