# -*- coding: utf-8 -*-
"""
问题二：日前预测方法与安全裕量的参数寻优
=====================================================================
目标：在「每天 0:00 只能使用历史数据」的约束下，搜索使全年总购电费
      （计划购电费 + 5 倍价紧急购电费）最小的预测方案。

搜索维度
    ① 预测方法：MA(n)（回看前 n 天移动平均，n=1 即持续性预测）、EWMA(α)、WDAY(lag)
    ② 安全裕量 m（kW）：加到负载预测上。紧急购电电价是计划购电的 5 倍，
       所以最优先验上应当「多买一点」对冲缺口风险 ——
       报童模型判据：缺口发生概率 > 1/5 时就值得提前多买 1 kWh。

为什么能用代理指标快速筛选（省掉上千次 LP）
    计划满足  b_t + G_t^p + d_t − c_t − s_t = L_t^p
    即        b_t + d_t − c_t = net_t^p + s_t          (net = L − G)
    实际缺口  e_t = max(0, net_t^实 − net_t^p − m − s_t) ≤ max(0, net_t^实 − net_t^p − m)
    右边即「净负荷预测偏差的正部」，是缺口的一个紧上界，且完全不依赖储能调度，
    可向量化 O(1) 批量算，用来给上千个组合排序；再对前 K 名跑完整 LP 取真值。

两阶段
    阶段一：全网格快筛（代理指标）
    阶段二：前 K 名跑完整 365 天 LP，输出真实总费用，选出最优参数
"""
import itertools

import numpy as np
import pandas as pd

from config import *
from q2 import build_forecast, solve_day        # 复用预测构造与单日 LP

# ==================== 寻优配置 ====================
TXT_TUNE = "q2_参数寻优.txt"
OUT_START = "2025-02-01"        # 与 result2.xlsx 的输出区间一致
TOP_K = 3                       # 阶段二跑完整 LP 的候选个数

GRID_MA = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 21, 25, 30]
GRID_ALPHA = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
GRID_LAG = [7, 14]
GRID_HEDGE = [0, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500]   # kW
EXCLUDE_RANK = {"perfect"}      # 用了当天实际值，不参与排序（仅作理论下界）


def build_specs():
    """生成全部候选预测规格（spec 语法见 q2.build_forecast）

    负载与光伏的主导规律不同（负载有强周周期、光伏看天气），
    因此除了「同一方法用于两者」，也试几种混搭（左负载 | 右光伏）。
    """
    specs = [f"ma:{n}" for n in GRID_MA]
    specs += [f"ewma:{a}" for a in GRID_ALPHA]
    specs += [f"wday:{lag}" for lag in GRID_LAG]
    for sl in ("wday:7", "ma:1", "ma:7", "ewma:0.9"):
        for sg in ("ma:1", "ma:3", "wday:7"):
            if sl != sg:
                specs.append(f"{sl}|{sg}")
    specs += ["att1:0", "perfect"]
    return specs


def run_full(pi, L, G, F_L, F_G, hedge, eval_mask):
    """对给定预测方案跑完整 365 天 LP，只统计 eval_mask 为 True 的天数

    返回 (计划购电量, 计划购电费, 紧急购电量, 紧急购电费, 出现紧急购电的天数)
    """
    D = L.shape[0]
    E_start = SOC0
    tb = tc = te = tce = 0.0
    n_day = 0
    for i in range(D):
        plan = solve_day(pi, F_L[i] + hedge, F_G[i], E_start)
        supply = G[i] + plan["b"] + plan["d"] - plan["c"]
        deficit = np.maximum(0.0, L[i] - supply)
        if eval_mask[i]:
            tb += plan["b"].sum() * DT_H
            tc += float(np.sum(pi * plan["b"] * DT_H))
            te += deficit.sum() * DT_H
            tce += float(np.sum(5.0 * pi * deficit * DT_H))
            if deficit.sum() > 1e-9:
                n_day += 1
        E_start = plan["E"][-1]
    return tb, tc, te, tce, n_day


def main():
    # ---------- 数据 ----------
    A1 = att1_arrays(load_att1())
    pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
    dates, L, G = load_att2()
    D, T = L.shape
    net = L - G
    eval_mask = np.asarray(dates >= pd.Timestamp(OUT_START))   # DatetimeIndex 比较直接得 ndarray
    pi_row = pi[None, :]
    net_ev = net[eval_mask]
    n_eval = int(eval_mask.sum())

    rule("【1】候选预测方案（严格只用历史数据）")
    specs = build_specs()
    pool = []                                    # [(spec, F_L, F_G), ...]
    log(f"共 {len(specs)} 种预测规格 × {len(GRID_HEDGE)} 档安全裕量 "
        f"= {len(specs) * len(GRID_HEDGE)} 个组合；评价区间 {OUT_START} 起共 {n_eval} 天")
    log("（'|' 左侧为负载预测方法，右侧为光伏预测方法）\n")
    log(f"{'预测规格':<20s} {'负载MAE':>9s} {'光伏MAE':>9s} {'净负荷MAE':>10s}")
    for sp in specs:
        FL, FG = build_forecast(L, G, L1, G1, sp)
        pool.append((sp, FL, FG))
        mae_l = np.abs(L[eval_mask] - FL[eval_mask]).mean()
        mae_g = np.abs(G[eval_mask] - FG[eval_mask]).mean()
        mae_n = np.abs(net_ev - (FL - FG)[eval_mask]).mean()
        log(f"{sp:<20s} {mae_l:9.1f} {mae_g:9.1f} {mae_n:10.1f}")

    # ---------- 阶段一：代理指标快筛 ----------
    rule("【2】阶段一：代理指标快筛")
    rows = []
    for sp, FL, FG in pool:
        if sp in EXCLUDE_RANK:                   # 完美预测用未来数据，不参与排序
            continue
        np_ev = (FL - FG)[eval_mask]
        for m in GRID_HEDGE:
            plan_net = np.maximum(0.0, np_ev + m)
            cost_plan = float(np.sum(pi_row * plan_net * DT_H)) / n_eval
            deficit = np.maximum(0.0, net_ev - np_ev - m)
            cost_emg = float(np.sum(5.0 * pi_row * deficit * DT_H)) / n_eval
            rows.append({"spec": sp, "hedge": m, "plan": cost_plan,
                         "emg": cost_emg, "total": cost_plan + cost_emg})
    rows.sort(key=lambda r: r["total"])
    log("（日均费用，元/天；代理指标不含储能套利，仅用于排序）")
    log(f"{'排名':>4s} {'预测规格':<20s} {'裕量kW':>7s} {'计划购电费':>12s} "
        f"{'紧急购电费':>12s} {'合计':>12s}")
    for k, r in enumerate(rows[:15], 1):
        log(f"{k:>4d} {r['spec']:<20s} {r['hedge']:>7g} {r['plan']:>12,.0f} "
            f"{r['emg']:>12,.0f} {r['total']:>12,.0f}")

    # 最优规格下，安全裕量对费用的影响
    best_spec = rows[0]["spec"]
    log(f"\n最优规格「{best_spec}」的安全裕量敏感性（日均，元/天）：")
    log(f"{'裕量kW':>7s} {'计划购电费':>12s} {'紧急购电费':>12s} {'合计':>12s} {'较最优':>10s}")
    sub = sorted([r for r in rows if r["spec"] == best_spec], key=lambda r: r["hedge"])
    base = min(r["total"] for r in sub)
    for r in sub:
        mark = "  ←最优" if abs(r["total"] - base) < 1e-9 else ""
        log(f"{r['hedge']:>7g} {r['plan']:>12,.0f} {r['emg']:>12,.0f} "
            f"{r['total']:>12,.0f} {r['total'] - base:>+10,.0f}{mark}")

    # ---------- 阶段二：前 K 名跑完整 LP ----------
    rule(f"【3】阶段二：前 {TOP_K} 名跑完整 365 天 LP（真实费用）")
    top = rows[:TOP_K]
    # 去重：同一 (method, param) 只保留首个（裕量不同仍需各跑一次）
    log(f"{'预测规格':<20s} {'裕量kW':>7s} {'计划购电量':>14s} {'计划购电费':>14s} "
        f"{'紧急购电量':>13s} {'紧急购电费':>14s} {'总购电费':>14s}")
    results = []
    for r in top:
        FL, FG = build_forecast(L, G, L1, G1, r["spec"])
        tb, tc, te, tce, nd = run_full(pi, L, G, FL, FG, r["hedge"], eval_mask)
        results.append({**r, "tb": tb, "tc": tc, "te": te, "tce": tce,
                        "real_total": tc + tce, "days": nd})
        log(f"{r['spec']:<20s} {r['hedge']:>7g} {tb:>14,.0f} {tc:>14,.0f} "
            f"{te:>13,.0f} {tce:>14,.0f} {tc + tce:>14,.0f}")
    results.sort(key=lambda r: r["real_total"])
    best = results[0]

    rule("【4】最优参数")
    log(f"预测规格 = {best['spec']}，安全裕量 = {best['hedge']:g} kW")
    log(f"计划购电量 = {best['tb']:,.1f} kWh；计划购电费 = {best['tc']:,.1f} 元")
    log(f"紧急购电量 = {best['te']:,.1f} kWh（占计划购电量 {best['te'] / best['tb']:.2%}）；"
        f"紧急购电费 = {best['tce']:,.1f} 元")
    log(f"总购电费 = {best['real_total']:,.1f} 元（紧急购电占 "
        f"{best['tce'] / best['real_total']:.1%}）；出现紧急购电 {best['days']}/{n_eval} 天")
    log(f"\n复现命令：python q2.py '{best['spec']}' {best['hedge']:g}")
    write_report(TXT_TUNE)


if __name__ == "__main__":
    main()
