# -*- coding: utf-8 -*-
"""
问题二补充分析：逐日 LP  vs  多日联合 LP
=====================================================================
问题：把「每天 0:00 独立求解单日 LP」换成「多日联合 LP」，结果会更好吗？

关键点
    问题2 的电价每天完全相同（都用附件1 那条曲线），因此不存在
    「今天便宜买、明天贵了卖」的跨日套利。多日 LP 能多榨出来的只有：
        ① 夜间谷价的细微差别：22:00-24:00 约 0.42 元/kWh
           vs 次日 0:00-5:20 约 0.43~0.44 元/kWh
        ② 跨日的储能水位调度（单日 LP 因无终端约束，每天末必然放空到 SOC 下限）

本脚本在若干个代表性 7 天窗口上做对照实验：
    A. 逐日 LP（现状）：每天独立求解，储能初值逐日传递
    B. 多日联合 LP   ：窗口内 7 天一次性求解，储能跨日连续
    两者都令窗口首端 = 1200 kWh、末端 = 1200 kWh，保证可比

费用口径与 q2.py 一致：
    总费用 = Σ π_t·b_t·Δt  +  Σ 5·π_t·max(0, L_t^实 − G_t^实 − b_t − d_t + c_t)·Δt
"""
import numpy as np
import pulp

from config import *
from q2 import build_forecast, solve_day

# ==================== 实验配置 ====================
SPEC = "wday:7|ma:3"        # q2_tune 寻优得到的最优预测规格
HEDGE = 300.0               # 最优安全裕量 (kW)
WIN_DAYS = 7                # 每个窗口的天数
WIN_START = ["2025-01-15", "2025-03-17", "2025-06-18", "2025-09-20", "2025-12-18"]
TXT = "q2_多日LP对比.txt"


def solve_multiday(pi, Lp, Gp, E_start, E_end):
    """多日联合 LP：一次求解 n 天的购电与充放电计划

        min  Σ_i Σ_t π_t · b_{i,t} · Δt
        s.t. b + G^p + d − c − s = L^p                      逐时段功率平衡
             E = E_prev + (η·c − d/η)·Δt                     储电量跨日连续
             SOC_min ≤ E ≤ SOC_max
             E(窗口首) = E_start, E(窗口末) = E_end

    Lp, Gp : (n, T) 每天的计划依据（预测值）
    返回 (b, c, d, s)，均为 (n, T)
    """
    n, T = Lp.shape
    m = pulp.LpProblem("q2_multi", pulp.LpMinimize)
    b = [[pulp.LpVariable(f"b{i}_{t}", lowBound=0) for t in range(T)] for i in range(n)]
    c = [[pulp.LpVariable(f"c{i}_{t}", lowBound=0, upBound=P_RATE) for t in range(T)] for i in range(n)]
    d = [[pulp.LpVariable(f"d{i}_{t}", lowBound=0, upBound=P_RATE) for t in range(T)] for i in range(n)]
    s = [[pulp.LpVariable(f"s{i}_{t}", lowBound=0) for t in range(T)] for i in range(n)]
    E = [[pulp.LpVariable(f"E{i}_{t}", lowBound=SOC_MIN, upBound=SOC_MAX) for t in range(T)]
         for i in range(n)]

    m += pulp.lpSum(pi[t] * b[i][t] * DT_H for i in range(n) for t in range(T))
    prev = E_start
    for i in range(n):
        for t in range(T):
            m += b[i][t] + Gp[i][t] + d[i][t] - c[i][t] - s[i][t] == Lp[i][t]
            m += E[i][t] == (prev if t == 0 else E[i][t - 1]) \
                + (ETA * c[i][t] - d[i][t] / ETA) * DT_H
        prev = E[i][T - 1]
    m += E[n - 1][T - 1] == E_end

    m.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[m.status] != "Optimal":
        raise RuntimeError(f"多日 LP 求解失败：{pulp.LpStatus[m.status]}")
    pick = lambda X: np.array([[v.value() for v in row] for row in X])
    return pick(b), pick(c), pick(d), pick(s)


def cost_of(pi, L, G, b, c, d):
    """按实际负载/光伏结算：计划购电费 + 5 倍价紧急购电费"""
    supply = G + b + d - c
    deficit = np.maximum(0.0, L - supply)
    return (float(np.sum(pi[None, :] * b * DT_H)),
            float(np.sum(5.0 * pi[None, :] * deficit * DT_H)))


def main():
    A1 = att1_arrays(load_att1())
    pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
    dates, L, G = load_att2()
    F_L, F_G = build_forecast(L, G, L1, G1, SPEC)

    rule("多日联合 LP  vs  逐日 LP")
    log(f"预测规格 = {SPEC}，安全裕量 = {HEDGE:g} kW，窗口长度 = {WIN_DAYS} 天")
    log(f"电价：每天相同（附件1 曲线），峰谷 {pi.min():.4f} ~ {pi.max():.4f} 元/kWh")
    log("说明：两者都令窗口首端/末端储电量 = 1200 kWh，保证口径可比")

    tot_daily = tot_multi = tot_plan_d = tot_plan_m = 0.0
    tot_emg_d = tot_emg_m = 0.0
    log(f"\n{'窗口起始':<12s} {'逐日LP总费用':>14s} {'多日LP总费用':>14s} {'多日节省':>12s} {'相对':>8s}")
    for ds in WIN_START:
        i0 = int(np.argmax(dates >= np.datetime64(ds)))
        sl = slice(i0, i0 + WIN_DAYS)
        Lp = F_L[sl] + HEDGE
        Gp = F_G[sl]

        # --- A. 逐日 LP ---
        E_start = SOC_MIN
        bd = np.empty_like(Lp); cd = np.empty_like(Lp); dd = np.empty_like(Lp)
        for k in range(WIN_DAYS):
            plan = solve_day(pi, Lp[k], Gp[k], E_start)
            bd[k], cd[k], dd[k] = plan["b"], plan["c"], plan["d"]
            E_start = plan["E"][-1]
        p_d, e_d = cost_of(pi, L[sl], G[sl], bd, cd, dd)

        # --- B. 多日联合 LP ---
        bm, cm, dm, _ = solve_multiday(pi, Lp, Gp, SOC_MIN, SOC_MIN)
        p_m, e_m = cost_of(pi, L[sl], G[sl], bm, cm, dm)

        tot_plan_d += p_d; tot_emg_d += e_d
        tot_plan_m += p_m; tot_emg_m += e_m
        tot_daily += p_d + e_d
        tot_multi += p_m + e_m
        log(f"{ds:<12s} {p_d + e_d:>14,.0f} {p_m + e_m:>14,.0f} "
            f"{p_d + e_d - p_m - e_m:>12,.0f} {(p_d + e_d - p_m - e_m) / (p_d + e_d):>8.3%}")

    log(f"\n{'合计':<12s} {tot_daily:>14,.0f} {tot_multi:>14,.0f} "
        f"{tot_daily - tot_multi:>12,.0f} {(tot_daily - tot_multi) / tot_daily:>8.3%}")
    log(f"\n细分：逐日 LP  计划购电费 {tot_plan_d:>12,.0f} 元，紧急购电费 {tot_emg_d:>11,.0f} 元")
    log(f"      多日 LP  计划购电费 {tot_plan_m:>12,.0f} 元，紧急购电费 {tot_emg_m:>11,.0f} 元")
    log(f"      → 差别全部来自计划购电费（紧急购电由预测偏差决定，与多日 LP 无关）")

    n_eval = len(WIN_START) * WIN_DAYS
    log(f"\n按 {n_eval} 天窗口外推全年 334 天：")
    log(f"      逐日 LP 约 {tot_daily / n_eval * 334:,.0f} 元；"
        f"多日 LP 约 {tot_multi / n_eval * 334:,.0f} 元；"
        f"节省约 {(tot_daily - tot_multi) / n_eval * 334:,.0f} 元/年")
    write_report(TXT)


if __name__ == "__main__":
    main()
