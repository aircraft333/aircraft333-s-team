# -*- coding: utf-8 -*-
"""
问题二补充：储能「按计划执行」 vs 「实时再调度」
=====================================================================
目前的 q2.py 采用第一种口径：计划里定好了储能的充放电曲线，当天就照此执行，
偏差全部由紧急购电填补。由此可得

    缺口 e_t = max(0, ε_t − x_t − s_t),   ε_t = (L−G)|实 − (L−G)|预

即缺口只取决于预测误差与计划弃光，**与储能调度无关**（储能项在计划/现实两侧抵消）。

但另一种同样合理的口径是：0:00 只申报「计划购电量 b_t」，储能在日内实时再调度
（这是实际微网的标准做法），于是缺口变成「储能尽力之后仍不足的部分」：

    第二级(实时)：给定 b_t 与实际 L、G，求
        min  Σ_t 5·π_t · e_t · Δt
        s.t. e_t ≥ L_t − G_t − b_t − d_t + c_t,  e_t ≥ 0
             b_t 固定（已按计划结算）
             0 ≤ c_t, d_t ≤ P̄，SOC 区间与状态转移同前

本脚本对全年 365 天两种口径并列求解，量化差异。
"""
import numpy as np
import pandas as pd
import pulp

from config import *
from q2 import build_forecast, solve_day

SPEC = "wday:7|ma:3"
TXT = "q2_实时再调度对比.txt"
OUT_START = "2025-02-01"
HEDGES = [0.0, 300.0]            # 安全裕量档位（kW）


def solve_rt(pi, L_real, G_real, b_fixed, E_start):
    """第二级：购电量固定，实时再调度储能以最小化紧急购电

    返回 (紧急购电量 kWh, 储能末电量 kWh, 充电量, 放电量)
    """
    T = N_SLOT
    m = pulp.LpProblem("q2_rt", pulp.LpMinimize)
    c = [pulp.LpVariable(f"c{t}", lowBound=0, upBound=P_RATE) for t in range(T)]
    d = [pulp.LpVariable(f"d{t}", lowBound=0, upBound=P_RATE) for t in range(T)]
    e = [pulp.LpVariable(f"e{t}", lowBound=0) for t in range(T)]
    E = [pulp.LpVariable(f"E{t}", lowBound=SOC_MIN, upBound=SOC_MAX) for t in range(T)]

    m += pulp.lpSum(5.0 * pi[t] * e[t] * DT_H for t in range(T))
    for t in range(T):
        m += e[t] >= L_real[t] - G_real[t] - b_fixed[t] - d[t] + c[t]
        prev = E_start if t == 0 else E[t - 1]
        m += E[t] == prev + (ETA * c[t] - d[t] / ETA) * DT_H
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[m.status] != "Optimal":
        raise RuntimeError(f"实时再调度求解失败：{pulp.LpStatus[m.status]}")
    e_val = np.array([v.value() for v in e])
    return (float(e_val.sum() * DT_H),
            float(np.sum(5.0 * pi * e_val * DT_H)),
            float(E[T - 1].value()))


def main():
    A1 = att1_arrays(load_att1())
    pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
    dates, L, G = load_att2()
    F_L, F_G = build_forecast(L, G, L1, G1, SPEC)
    mask = np.asarray(dates >= pd.Timestamp(OUT_START))
    n_eval = int(mask.sum())

    rule("储能口径对比：按计划执行  vs  实时再调度")
    log(f"预测规格 = {SPEC}；评价区间 {OUT_START} 起 {n_eval} 天")

    log(f"\n{'裕量kW':>7s} | {'按计划执行: 计划费+紧急费=合计':>38s} | "
        f"{'实时再调度: 计划费+紧急费=合计':>38s} | {'节省':>10s}")
    for h in HEDGES:
        Ea = Eb = SOC0
        pa = ea = qa = pb = eb = qb = 0.0
        for i in range(len(dates)):
            plan = solve_day(pi, F_L[i] + h, F_G[i], Ea)

            # 口径 A：储能严格按计划执行
            supply = G[i] + plan["b"] + plan["d"] - plan["c"]
            deficit = np.maximum(0.0, L[i] - supply)

            # 口径 B：购电量固定，储能实时再调度
            q_rt, c_rt, E_next = solve_rt(pi, L[i], G[i], plan["b"], Eb)

            if mask[i]:
                pcost = float(np.sum(pi * plan["b"] * DT_H))
                pa += pcost
                ea += float(np.sum(5.0 * pi * deficit * DT_H))
                qa += float(deficit.sum() * DT_H)
                pb += pcost
                eb += c_rt
                qb += q_rt
            Ea = plan["E"][-1]
            Eb = E_next
            if (i + 1) % 90 == 0:
                log(f"    … 已完成 {i + 1}/{len(dates)} 天")

        log(f"{h:>7g} | {pa / 1e4:>10,.1f} + {ea / 1e4:>9,.1f} = {(pa + ea) / 1e4:>9,.1f} |"
            f" {pb / 1e4:>10,.1f} + {eb / 1e4:>9,.1f} = {(pb + eb) / 1e4:>9,.1f} |"
            f" {(pa + ea - pb - eb) / 1e4:>9,.1f}")
        log(f"{'':>7s} | 紧急购电量 {qa:>13,.0f} kWh                              |"
            f" 紧急购电量 {qb:>13,.0f} kWh")

    write_report(TXT)


if __name__ == "__main__":
    main()
