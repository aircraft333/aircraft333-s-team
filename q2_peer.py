# -*- coding: utf-8 -*-
"""
对标队友模型：把电价换成他们自编的分时电价，其余口径不变，看费用差异
=====================================================================
对比矩阵（2×2）：
    电价：附件1 实际分时电价  vs  队友自编 0.35/0.65/1.15
    储能口径：按计划充放电      vs  日内实时再调度（队友做法）

只使用附件1 与附件2。
"""
import numpy as np
import pandas as pd

from config import *
from q2 import build_forecast, solve_day, solve_rt

SPEC = "wday:7|ma:3"
OUT_START = "2025-02-01"
TXT = "q2_对标队友.txt"


def peer_price():
    """队友脚本里硬编码的分时电价：0-8h 0.35；8-11/15-18/21-24 0.65；11-15/18-21 1.15"""
    p = np.zeros(N_SLOT)
    hh = np.arange(N_SLOT) / 6.0        # 每个时段中点小时数
    p[:] = 0.35
    mid = ((hh >= 8) & (hh < 11)) | ((hh >= 15) & (hh < 18)) | ((hh >= 21) & (hh < 24))
    hi = ((hh >= 11) & (hh < 15)) | ((hh >= 18) & (hh < 21))
    p[mid] = 0.65
    p[hi] = 1.15
    return p


def run(pi, L, G, F_L, F_G, mask, use_rt):
    """给定电价与储能口径，跑全年；返回 (计划购电费, 紧急购电费, 紧急购电量)"""
    E0 = SOC0
    pc = ec = qe = 0.0
    for i in range(len(L)):
        plan = solve_day(pi, F_L[i], F_G[i], E0)
        if use_rt:
            rt = solve_rt(pi, L[i], G[i], plan["b"], E0)
            deficit = rt["e"]
            E_next = rt["E"][-1]
        else:
            supply = G[i] + plan["b"] + plan["d"] - plan["c"]
            deficit = np.maximum(0.0, L[i] - supply)
            E_next = plan["E"][-1]
        if mask[i]:
            pc += float(np.sum(pi * plan["b"] * DT_H))
            ec += float(np.sum(5.0 * pi * deficit * DT_H))
            qe += float(deficit.sum() * DT_H)
        E0 = E_next
    return pc, ec, qe


def main():
    A1 = att1_arrays(load_att1())
    pi1, L1, G1 = A1["price"], A1["load"], A1["pv"]
    dates, L, G = load_att2()
    F_L, F_G = build_forecast(L, G, L1, G1, SPEC)
    mask = np.asarray(dates >= pd.Timestamp(OUT_START))
    pi2 = peer_price()

    rule("电价与储能口径的 2×2 对比（单位：万元）")
    log(f"附件1 电价：均值 {pi1.mean():.4f} 元/kWh，峰谷 {pi1.min():.4f} ~ {pi1.max():.4f}")
    log(f"队友自编电价：均值 {pi2.mean():.4f} 元/kWh，峰谷 {pi2.min():.4f} ~ {pi2.max():.4f}"
        f"（均值低 {(1 - pi2.mean() / pi1.mean()) * 100:.1f}%）\n")
    log(f"{'电价':<12s} {'储能口径':<14s} {'计划购电费':>12s} {'紧急购电费':>12s} "
        f"{'合计':>12s} {'紧急购电量kWh':>15s}")

    for pname, pi in (("附件1", pi1), ("队友自编", pi2)):
        for rname, use_rt in (("按计划执行", False), ("实时再调度", True)):
            pc, ec, qe = run(pi, L, G, F_L, F_G, mask, use_rt)
            log(f"{pname:<12s} {rname:<14s} {pc / 1e4:>12,.1f} {ec / 1e4:>12,.1f} "
                f"{(pc + ec) / 1e4:>12,.1f} {qe:>15,.0f}")

    write_report(TXT)


if __name__ == "__main__":
    main()
