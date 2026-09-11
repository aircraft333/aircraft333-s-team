# -*- coding: utf-8 -*-
"""临时检验：q3.solve_adjust 的 LP 与逐槽执行是否一致（批量统计）"""
import numpy as np
import pulp

import q2
import q3
from config import *

A1 = att1_arrays(load_att1())
pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
dates, L, G = load_att2()
df3 = load_att3()
EPS = q3.EPS_CUR


def lp_full(pi, Lf, Gf, bp, E_start, t_from):
    """与 q3.solve_adjust 同构，额外返回 LP 内部的 e 与目标值"""
    T, n = N_SLOT, N_SLOT - t_from
    m = pulp.LpProblem("chk", pulp.LpMinimize)
    b = [pulp.LpVariable(f"b{k}", lowBound=0) for k in range(n)]
    c = [pulp.LpVariable(f"c{k}", lowBound=0, upBound=P_RATE) for k in range(n)]
    d = [pulp.LpVariable(f"d{k}", lowBound=0, upBound=P_RATE) for k in range(n)]
    s = [pulp.LpVariable(f"s{k}", lowBound=0) for k in range(n)]
    e = [pulp.LpVariable(f"e{k}", lowBound=0) for k in range(n)]
    E = [pulp.LpVariable(f"E{k}", lowBound=SOC_MIN, upBound=SOC_MAX) for k in range(n)]
    dp = [pulp.LpVariable(f"dp{k}", lowBound=0) for k in range(n)]
    dm = [pulp.LpVariable(f"dm{k}", lowBound=0) for k in range(n)]
    m += pulp.lpSum((1.5 * pi[t] * dp[k] - 0.5 * pi[t] * dm[k]
                     + 5.0 * pi[t] * e[k] + EPS * s[k]) * DT_H
                    for k, t in enumerate(range(t_from, T)))
    for k, t in enumerate(range(t_from, T)):
        m += b[k] == bp[t] + dp[k] - dm[k]
        m += d[k] - c[k] - s[k] - e[k] == Lf[t] - Gf[t] - b[k]
        m += s[k] <= Gf[t]
        prev = E_start if k == 0 else E[k - 1]
        m += d[k] <= (prev - SOC_MIN) * ETA / DT_H
        m += c[k] <= (SOC_MAX - prev) / (ETA * DT_H)
        m += E[k] == prev + (ETA * c[k] - d[k] / ETA) * DT_H
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    clip = lambda a: np.maximum(np.array([v.value() for v in a]), 0.0)
    bb = bp.copy()
    bb[t_from:] = clip(b)
    return dict(b=bb, e=clip(e))


def dev_cost(pi, bp, ba, t0):
    """t ≥ t0 段的调整相关购电费（含计划部分）"""
    p, q = bp[t0:], ba[t0:]
    return float(np.sum(pi[t0:] * (np.minimum(p, q)
                                   + 0.5 * np.maximum(p - q, 0)
                                   + 1.5 * np.maximum(q - p, 0)) * DT_H))


Gf = q3.build_pv_forecast(df3, dates, G, q3.DECIDE_H, q3.PV_MIX)
Lf = q3.build_load_forecast(L, L1, dates, q3.LOAD_LAG and dates, q3.DECIDE_H, 0.0) \
    if False else q3.build_load_forecast(L, L1, dates, q3.DECIDE_H, q3.LOAD_LAG, 0.0)
t0 = 36
I0 = 40

acc = dict(lp_e=0.0, gf_e=0.0, ac_e=0.0, lp_c=0.0, gf_c=0.0, ac_c=0.0, n=0, nd=0)
E_start = SOC0
for i, day in enumerate(dates):
    bp = q2.solve_day(pi, Lf[0, i], Gf[0, i], E_start)["b"]
    ba = bp
    if i >= I0:
        E36 = simulate_dispatch(L[i], G[i], bp, E_start, 0, t0)["E"][-1]
        r = lp_full(pi, Lf[1, i], Gf[1, i], bp, E36, t0)
        ba = r["b"]
        ea = r["e"]
        sf = simulate_dispatch(Lf[1, i], Gf[1, i], ba, E36, t0, N_SLOT)["e"]
        sa = simulate_dispatch(L[i], G[i], ba, E36, t0, N_SLOT)["e"]
        dc = dev_cost(pi, bp, ba, t0)
        acc["lp_e"] += ea.sum() * DT_H
        acc["gf_e"] += sf.sum() * DT_H
        acc["ac_e"] += sa.sum() * DT_H
        acc["lp_c"] += dc + float(np.sum(5 * pi[t0:] * ea * DT_H))
        acc["gf_c"] += dc + float(np.sum(5 * pi[t0:] * sf * DT_H))
        acc["ac_c"] += dc + float(np.sum(5 * pi[t0:] * sa * DT_H))
        acc["n"] += 1
        acc["nd"] += 1 if ea.sum() > 1e-6 else 0
    E_start = simulate_dispatch(L[i], G[i], ba, E_start)["E"][-1]

rule(f"调整 LP 与逐槽执行的一致性（第 {I0}~{len(dates) - 1} 天，共 {acc['n']} 天）")
log(f"【A】LP 内部假设的紧急购电   {acc['lp_e']:>11,.1f} kWh   t≥36 段总费用 {acc['lp_c']:>13,.1f} 元")
log(f"【B】同 b + 逐槽 + 同一预报   {acc['gf_e']:>11,.1f} kWh   t≥36 段总费用 {acc['gf_c']:>13,.1f} 元")
log(f"【C】同 b + 逐槽 + 实际数据   {acc['ac_e']:>11,.1f} kWh   t≥36 段总费用 {acc['ac_c']:>13,.1f} 元")
log("")
log(f"A − B = {acc['lp_e'] - acc['gf_e']:+,.1f} kWh；费用差 "
    f"{acc['lp_c'] - acc['gf_c']:+,.1f} 元")
log(f"LP 主动掏紧急购电（e>0）的天数 = {acc['nd']}/{acc['n']}")
if acc["lp_c"] < acc["gf_c"] - 1.0:
    log("⇒ ✗ 不一致：LP 假设自己能在电池有电时留电以压低费用，逐槽执行做不到，")
    log("  于是 LP 挑出的 b 是『为错误模型优化的』。")
else:
    log("⇒ ✓ 基本一致：LP 的最优解与逐槽行为吻合。")
