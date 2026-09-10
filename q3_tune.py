# -*- coding: utf-8 -*-
"""问题三参数寻优：光伏预报混合权重 / 安全裕量 / 负载自适应 / 决策时刻加密

在连续日期区块上做筛选（区块内储能初值连续，故与全年口径一致），
再对最优组合跑全年。用法：python q3_tune.py [起始日序号] [结束日序号]
"""
import sys
import time

import q3
from config import *

I0 = int(sys.argv[1]) if len(sys.argv) > 1 else 120
I1 = int(sys.argv[2]) if len(sys.argv) > 2 else 210
TXT = "q3_参数寻优.txt"

A1 = att1_arrays(load_att1())
pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
dates, L, G = load_att2()
df3 = load_att3()


def run_block(decide_h, mix, hedge, adapt, i0=I0, i1=I1, gf=None, lf=None):
    """在 [i0, i1) 上滚动求解，返回 (总费用, 紧急购电费, 紧急购电量)"""
    gf = q3.build_pv_forecast(df3, dates, G, decide_h, mix) if gf is None else gf
    lf = q3.build_load_forecast(L, L1, dates, decide_h, q3.LOAD_LAG, adapt) if lf is None else lf
    E, ct, ce, qe = SOC0, 0.0, 0.0, 0.0
    for i in range(i0, i1):
        r = q3.run_day(pi, lf[:, i], gf[:, i], L[i], G[i], E, decide_h, hedge)
        E = r["E24"]
        ct += r["cost_dev"]
        ce += r["cost_emg"]
        qe += r["e"].sum() * DT_H
    return ct + ce, ce, qe


def sweep(name, base, grid, key):
    """在 base 配置上扫描一个参数"""
    log(f"\n--- 扫描 {name}（区块 {I0}~{I1}，{I1 - I0} 天）---")
    best = (None, 1e18)
    for v in grid:
        cfg = dict(base)
        cfg[key] = v
        t = time.time()
        tot, ce, qe = run_block(**cfg)
        mark = ""
        if tot < best[1]:
            best = (v, tot)
            mark = "  ←"
        log(f"  {name} = {v!s:<28s} 总费用 {tot:>13,.1f} 元  紧急购电费 {ce:>11,.1f} 元"
            f"  紧急电量 {qe:>11,.1f} kWh  ({time.time() - t:.0f}s){mark}")
    log(f"  → {name} 最优：{best[0]}（总费用 {best[1]:,.1f} 元）")
    return best[0]


def main():
    rule(f"问题三参数寻优（区块 {I0}~{I1}，共 {I1 - I0} 天）")
    base = dict(decide_h=q3.DECIDE_H, mix=q3.PV_MIX, hedge=q3.HEDGE, adapt=0.0)

    t = time.time()
    tot0, ce0, qe0 = run_block(**base)
    log(f"基准配置：总费用 {tot0:,.1f} 元（紧急购电费 {ce0:,.1f} 元）"
        f"    单次区块求解 {time.time() - t:.0f} s")

    m = sweep("光伏混合权重 mix", base, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "mix")
    base["mix"] = float(m)

    h = sweep("安全裕量 hedge (kW)", base, [0.0, 200.0, 500.0, 1000.0, 2000.0], "hedge")
    base["hedge"] = float(h)

    a = sweep("负载自适应 adapt", base, [0.0, 0.1, 0.2, 0.4], "adapt")
    base["adapt"] = float(a)

    e = sweep("决策时刻 decide_h", base,
              [(0, 6, 12, 18), (0, 3, 6, 9, 12, 15, 18, 21),
               tuple(range(0, 24, 2)), tuple(range(24))], "decide_h")
    base["decide_h"] = e

    rule("最优组合（区块内）")
    log(f"  mix={base['mix']}  hedge={base['hedge']}  adapt={base['adapt']}  "
        f"decide_h={base['decide_h']}")
    tot, ce, qe = run_block(**base)
    log(f"  总费用 {tot:,.1f} 元（基准 {tot0:,.1f} 元，改善 {tot0 - tot:,.1f} 元）")
    write_report(TXT)


if __name__ == "__main__":
    main()
