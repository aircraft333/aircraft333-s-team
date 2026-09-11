# -*- coding: utf-8 -*-
"""Q2 执行口径诊断：逐槽被动平衡 vs 每日完全信息最优

目的：看清两者的差别究竟来自哪里 —— 是「储能吞吐总量」还是「紧急购电的时段选择」。
输出：各口径的费用、紧急购电量按峰/平/谷拆解、紧急购电的逐小时分布。
运行：python q2_diag.py（结果同时写入 q2_口径诊断.txt）
"""
import numpy as np

import q2
from config import *

solve_rt = q2.solve_rt          # 对照口径的 LP 直接复用 q2.py，避免两份实现漂移

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


def run(mode, X):
    E_start = SOC0
    tot_e = np.zeros(N_SLOT)
    q_ch = q_dis = 0.0
    recs = []
    for i, day in enumerate(dates):
        plan = q2.solve_day(pi, F_L[i] + X[i], F_G[i], E_start)
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


# =====================================================================
# 三、2×2 对照：裕量口径 × 储能执行方式
# ---------------------------------------------------------------------
# 【口径提醒】早期版本这里只跑「无裕量」口径，与本仓库现行主口径
# （报童分位数裕量）不一致，容易在论文里被误读成主方案的下界。
# 现改为两种裕量口径并列，主口径那一行可与 q2.py 主结果直接对照。
# =====================================================================
err = (L - G) - (F_L - F_G)                 # 净负荷预测误差（与 q2.py 同口径）
X_MAIN = q2.build_hedge(err, q2.HEDGE_MODE, q2.HEDGE_PARAM, q2.HEDGE_WIN)
X_ZERO = np.zeros((len(dates), N_SLOT))
HEDGES = [("无裕量 hedge=0", X_ZERO),
          (f"主口径 q={q2.HEDGE_PARAM:g}/win={q2.HEDGE_WIN}", X_MAIN)]
MODES = [("greedy", "逐槽被动平衡（因果、无前瞻）"),
         ("best", "每日完全信息最优（事后下界）")]

rule("储能执行口径诊断：裕量口径 × 执行方式")
log(f"预测规格 {q2.FC_SPEC}；输出区间 {q2.OUT_START} 起；逐日储能初值连续")
log(f"主口径裕量 = 逐时段历史净负荷误差的 {q2.HEDGE_PARAM:.2f} 分位数"
    f"（回看 {q2.HEDGE_WIN} 天；全年日均 {X_MAIN.mean():.0f} kW，最大 {X_MAIN.max():.0f} kW）")

R = {}
for hname, X in HEDGES:
    for m, mname in MODES:
        r = R[(hname, m)] = run(m, X)
        log(f"  {hname:<24s} | {mname:<26s} | 计划 {r['cost_plan']:>13,.1f} | "
            f"紧急 {r['cost_emg']:>11,.1f} | 合计 {r['cost_plan'] + r['cost_emg']:>13,.1f} 元")

a = R[(HEDGES[1][0], "greedy")]
b = R[(HEDGES[1][0], "best")]
log("")
log(f"主口径自检：逐槽法应为 q2.py 主结果 13,949,108.5 元 → 实得 "
    f"{a['cost_plan'] + a['cost_emg']:,.1f} 元，"
    f"差 {a['cost_plan'] + a['cost_emg'] - 13949108.5:+,.1f} 元")

rule("主口径下两组结果的明细")
for m, nm in MODES:
    r = R[(HEDGES[1][0], m)]
    log(f"\n【{nm}】（2025.2.1-12.31，{r['n_days']} 天）")
    log(f"  计划购电量 {r['q_plan']:>14,.1f} kWh   计划购电费 {r['cost_plan']:>14,.1f} 元")
    log(f"  紧急购电量 {r['q_emg']:>14,.1f} kWh   紧急购电费 {r['cost_emg']:>14,.1f} 元")
    log(f"  总购电费 {r['cost_plan'] + r['cost_emg']:>16,.1f} 元")
    log(f"  充电 {r['q_ch']:>12,.1f} kWh   放电 {r['q_dis']:>12,.1f} kWh"
        f"（净 {r['q_ch'] - r['q_dis']:+,.1f}）")
    if r["q_emg"] > 0:
        log("  紧急购电量按电价档拆分：" + "  ".join(
            f"{k} {r['tot_e'][tier == k].sum() / r['q_emg']:>6.1%}"
            for k in (TIER_V, TIER_F, TIER_P)))
        log("  紧急购电费按电价档拆分：" + "  ".join(
            f"{k} {5 * pi[tier == k].dot(r['tot_e'][tier == k]) / r['cost_emg']:>6.1%}"
            for k in (TIER_V, TIER_F, TIER_P)))

rule("差距（逐槽法 相对 每日完全信息最优）")
log(f"  总购电费   {a['cost_plan'] + a['cost_emg'] - b['cost_plan'] - b['cost_emg']:>14,.1f} 元"
    f"（{(a['cost_plan'] + a['cost_emg']) / (b['cost_plan'] + b['cost_emg']) - 1:+.3%}）")
log(f"  紧急购电费 {a['cost_emg'] - b['cost_emg']:>14,.1f} 元")
log(f"  紧急购电量 {a['q_emg'] - b['q_emg']:>14,.1f} kWh（负值 = 每日最优反而多买了电量）")
log(f"  储能吞吐   充电 {a['q_ch'] - b['q_ch']:+,.1f} / 放电 {a['q_dis'] - b['q_dis']:+,.1f} kWh")

log("")
log("紧急购电量的逐小时分布（kWh，按小时汇总 6 个时段；主口径）")
log("  小时    逐槽法      每日最优     单价(元/kWh)")
for h in range(24):
    x, y = a["tot_e"][6 * h:6 * h + 6].sum(), b["tot_e"][6 * h:6 * h + 6].sum()
    if x + y > 1:
        pr = 5 * pi[6 * h:6 * h + 6].mean()
        bar = "#" * int(x / 8000)
        log(f"  {h:2d}:00 {x:>10,.0f} {y:>12,.0f}      {pr:>5.2f}   {bar}")

write_report(resolve("q2_口径诊断.txt"))

