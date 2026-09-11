# -*- coding: utf-8 -*-
"""Q2 执行口径诊断：逐槽法 vs 每日最优（完全信息 LP）

目的：看清两者的差别究竟来自哪里 —— 是"储能吞吐总量"还是"紧急购电的时段选择"。
输出：各口径的费用、紧急购电量按峰/平/谷拆解、紧急购电的逐小时分布。
"""
import numpy as np
import pulp

import q2
from config import *

A1 = att1_arrays(load_att1())
pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
dates, L, G = load_att2()
F_L, F_G = q2.build_forecast(L, G, L1, G1, q2.FC_SPEC)
tier, lo, hi = classify_tiers(pi)
log(f"电价：谷 ≤ {lo:.4f}，峰 ≥ {hi:.4f}，日均 {pi.mean():.4f}")

# 各档的时段数与总时长
log("时段构成：" + "  ".join(
    f"{k} {int((tier == k).sum())} 段（{(tier == k).sum() * DT_H:.1f} h）"
    for k in (TIER_V, TIER_F, TIER_P)))
log("各档紧急购电单价 5π：" + "  ".join(
    f"{k} {5 * pi[tier == k].mean():.3f} 元/kWh" for k in (TIER_V, TIER_F, TIER_P)))


def solve_rt(pi, L_real, G_real, b_fixed, E_start):
    """每日最优（完全信息 LP）：已知当天实际 L/G，最优安排储能以最小化 5π 紧急购电"""
    T = N_SLOT
    m = pulp.LpProblem("rt", pulp.LpMinimize)
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
        raise RuntimeError(pulp.LpStatus[m.status])
    return {k: np.array([v.value() for v in a])
            for k, a in (("e", e), ("c", c), ("d", d), ("E", E))}


def run(mode):
    E_start = SOC0
    tot_e = np.zeros(N_SLOT)
    q_ch = q_dis = 0.0
    recs = []
    for i, day in enumerate(dates):
        plan = q2.solve_day(pi, F_L[i], F_G[i], E_start)
        if mode == "greedy":
            sim = simulate_dispatch(L[i], G[i], plan["b"], E_start)
            e, c, d, E24 = sim["e"], sim["c"], sim["d"], sim["E"][-1]
        else:
            r = solve_rt(pi, L[i], G[i], plan["b"], E_start)
            e, c, d, E24 = r["e"], r["c"], r["d"], r["E"][-1]
        recs.append((day, plan["b"], e, c, d))
        E_start = E24
    # 统一按输出起始日期筛选（否则会把 1 月的热身期也算进去）
    out = [r for r in recs if r[0] >= pd.Timestamp("2025-02-01")]
    for _d, _b, e, c, d in out:
        tot_e += e * DT_H
        q_ch += c.sum() * DT_H
        q_dis += d.sum() * DT_H
    q_emg = sum(e.sum() * DT_H for _d, _b, e, _c, _d2 in out)
    return dict(q_emg=q_emg,
                cost_emg=sum(float(np.sum(5.0 * pi * e * DT_H)) for _d, _b, e, _c, _dd in out),
                q_plan=sum(b.sum() * DT_H for _d, b, _e, _c, _dd in out),
                cost_plan=sum(float(np.sum(pi * b * DT_H)) for _d, b, _e, _c, _dd in out),
                n_days=len(out), tot_e=tot_e, q_ch=q_ch, q_dis=q_dis)


log("\n" + "=" * 78)
log("正在跑两种口径的全年对比 …")
R = {m: run(m) for m in ("greedy", "best")}

for m, nm in (("greedy", "逐槽法（因果，无前瞻）"), ("best", "每日最优（完全信息 LP）")):
    r = R[m]
    log(f"\n【{nm}】（2025.2.1-12.31，{r['n_days']} 天）")
    log(f"  计划购电量 {r['q_plan']:>14,.1f} kWh   计划购电费 {r['cost_plan']:>14,.1f} 元")
    log(f"  紧急购电量 {r['q_emg']:>14,.1f} kWh   紧急购电费 {r['cost_emg']:>14,.1f} 元")
    log(f"  总购电费 {r['cost_plan'] + r['cost_emg']:>16,.1f} 元")
    log(f"  充电 {r['q_ch']:>12,.1f} kWh   放电 {r['q_dis']:>12,.1f} kWh"
        f"（净 {r['q_ch'] - r['q_dis']:+,.1f}）")
    log("  紧急购电量按电价档拆分：" + "  ".join(
        f"{k} {r['tot_e'][tier == k].sum() / r['q_emg']:>6.1%}" for k in (TIER_V, TIER_F, TIER_P)))
    log("  紧急购电费按电价档拆分：" + "  ".join(
        f"{k} {5 * pi[tier == k].dot(r['tot_e'][tier == k]) / r['cost_emg']:>6.1%}"
        for k in (TIER_V, TIER_F, TIER_P)))

a, b = R["greedy"], R["best"]
log("\n" + "=" * 78)
log("差距（每日最优 相对 逐槽法）：")
log(f"  总购电费  {a['cost_plan'] + a['cost_emg'] - b['cost_plan'] - b['cost_emg']:>14,.1f} 元")
log(f"  紧急购电费 {a['cost_emg'] - b['cost_emg']:>13,.1f} 元")
log(f"  紧急购电量 {a['q_emg'] - b['q_emg']:>13,.1f} kWh（负值 = 每日最优反而多买了电量）")
log(f"  储能吞吐   充电 {a['q_ch'] - b['q_ch']:+,.1f} / 放电 {a['q_dis'] - b['q_dis']:+,.1f} kWh")

log("\n紧急购电量的逐小时分布（kWh，按小时汇总 6 个时段）")
log("  小时    逐槽法      每日最优     单价(元/kWh)")
for h in range(24):
    x, y = a["tot_e"][6 * h:6 * h + 6].sum(), b["tot_e"][6 * h:6 * h + 6].sum()
    if x + y > 1:
        pr = 5 * pi[6 * h:6 * h + 6].mean()
        bar = "#" * int(x / 8000)
        log(f"  {h:2d}:00 {x:>10,.0f} {y:>12,.0f}      {pr:>5.2f}   {bar}")
