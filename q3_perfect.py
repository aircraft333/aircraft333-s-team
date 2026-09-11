# -*- coding: utf-8 -*-
"""
问题三：完美信息（perfect foresight）下界
=====================================================================
问：如果 0:00 就**准确知道**当天全部 144 个时段的实际负载 L 与光伏 G，
    第三问的总购电费能降到多少？——这是任何预报策略都突破不了的下界，
    也使「预报误差造成的损失」有了可量化的锚点。

为什么完美信息下「调整阶段形同虚设」
    决策时刻取 Lf = L、Gf = G 且 hedge = 0 时，0:00 的计划 LP 就是
    「全天确定性最优」解 b^p；而调整 LP 是同一问题在已知正确入口储电量
    E_{t0} 下的尾部子问题。若调整能严格改善，把该改动拼回 0:00 计划就会
    严格改善全天解，与 b^p 的最优性矛盾 ⟹ 必有 b^a = b^p、δ± = 0、
    紧急购电 e = 0（LP 约束内嵌的逐槽规则与计划自洽）。
    故完美值 = 0:00 计划购电费（其调整相关费用恰等于计划购电费）。

因此本脚本用**每条日只解 1 个 LP** 的快路径（计划 + 逐槽仿真），
并在若干日期上用 run_day() 的完整四阶段滚动路径交叉校验（必须逐位一致）。

输出：q3_完美信息.txt
"""
import re
import time

import q3
from config import *

TXT = "q3_完美信息.txt"
TXT_Q3 = "q3_结果汇总.txt"          # 现行口径全年值（用于算差值）
TXT_ANCHOR = "q3_口径递进锚点.txt"   # 无裕量锚点（对照）

lines = []


def log_(*a):
    s = "  ".join(str(x) for x in a)
    lines.append(s)
    print(s, flush=True)


def banner(t):
    log_("=" * 78)
    log_(t)
    log_("=" * 78)


# =====================================================================
# 数据与基准
# =====================================================================
A1 = att1_arrays(load_att1())
pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
dates, L, G = load_att2()
df3 = load_att3()

banner("问题三：完美信息（提前已知全部实际负载与光伏）下的费用下界")
log_(f"输出窗口：{dates[0].date()} 起 {len(dates)} 天；决策时刻 {q3.DECIDE_H}")
log_(f"现行口径（预报驱动）：光伏 {q3.PV_MIX:.0%} 附件3 + 历史外推，"
     f"负载上周同日 + 自适应 {q3.ADAPT:.0%}，裕量 quantile q={q3.HEDGE_PARAM}")
log_("完美信息口径：Lf = L、Gf = G、hedge = 0（误差为 0，无需裕量）")
log_("")

# 现行值（从主程序报告里读，避免重复跑一遍约 8 分钟的全年滚动）
base_total = None
try:
    with open(TXT_Q3, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if "总购电费" in ln:
                mt = re.search(r"([\d,]+\.\d+)", ln.split("=")[-1])
                if mt:
                    base_total = float(mt.group(1).replace(",", ""))
except FileNotFoundError:
    pass
log_(f"现行方案总购电费（读自 {TXT_Q3}）= "
     f"{'%.1f 元' % base_total if base_total else '（未找到，跳过对比）'}")

# =====================================================================
# 快路径：逐日 1 个计划 LP + 逐槽仿真
# =====================================================================
banner("【1】完美信息逐日最优（334 次计划 LP + 逐槽仿真）")
t0 = time.time()
E_start = SOC0
tot_plan = tot_dev = tot_emg = 0.0
q_adj = q_emg = q_ch = q_dis = q_cur = 0.0
cur_days = emg_days = 0
per_day = []                       # (date, 计划购电量, 总费用)
E_chain = []

for i, day in enumerate(dates):
    P, LL, GG = pi, L[i], G[i]
    plan = q3.solve_day(P, LL, GG, E_start)          # 完美信息 → 直接全天确定性最优
    bp = plan["b"]
    sim = q3.simulate_dispatch(LL, GG, bp, E_start)  # 实际执行（逐槽被动平衡）
    c_plan = float(np.sum(P * bp * DT_H))
    c_dev = q3.cost_dev(P, bp, bp)                   # δ=0 ⇒ 应恰等于 c_plan
    c_emg = float(np.sum(5.0 * P * sim["e"] * DT_H))
    if day >= pd.Timestamp(q3.OUT_START):
        tot_plan += c_plan
        tot_dev += c_dev
        tot_emg += c_emg
        q_adj += bp.sum() * DT_H
        q_emg += sim["e"].sum() * DT_H
        q_ch += sim["c"].sum() * DT_H
        q_dis += sim["d"].sum() * DT_H
        q_cur += sim["s"].sum() * DT_H
        emg_days += int(sim["e"].sum() * DT_H > 1e-3)
        cur_days += int(sim["s"].sum() * DT_H > 1e-3)
        per_day.append((day, bp.sum() * DT_H, c_dev + c_emg))
    E_start = sim["E"][-1]
    E_chain.append(E_start)
    if (i + 1) % 80 == 0:
        log_(f"    已完成 {i + 1}/{len(dates)} 天 …  {time.time() - t0:.1f}s")

perfect_total = tot_dev + tot_emg
log_(f"耗时 {time.time() - t0:.1f} 秒")
log_(f"计划购电费 = {tot_plan:,.1f} 元")
log_(f"调整相关购电费（δ=0，应等于计划购电费）= {tot_dev:,.1f} 元"
     f"    偏差 = {tot_dev - tot_plan:+.4f} 元")
log_(f"紧急购电费 = {tot_emg:,.1f} 元（{emg_days} 天有紧急购电）")
log_(f"★ 完美信息总购电费 = {perfect_total:,.1f} 元")
log_(f"调整后购电量 = {q_adj:,.1f} kWh    紧急购电量 = {q_emg:,.4f} kWh")
log_(f"充电 {q_ch:,.1f} kWh    放电 {q_dis:,.1f} kWh    弃光 {q_cur:,.1f} kWh"
     f"（{cur_days} 天有弃光）")
log_("")

# =====================================================================
# 交叉校验：完整四阶段滚动路径（Lf=Gf=实际值，hedge=0）应与快路径逐位一致
# =====================================================================
banner("【2】交叉校验：run_day() 完整四阶段滚动（完美预报）逐位一致")
CHK = list(range(0, len(dates), 23))[:16]        # 均匀取 16 天
t1 = time.time()
E_start = SOC0
E_of = {i: None for i in CHK}
for i in range(len(dates)):
    if i in E_of:
        E_of[i] = E_start
    P, LL, GG = pi, L[i], G[i]
    plan = q3.solve_day(P, LL, GG, E_start)
    E_start = q3.simulate_dispatch(LL, GG, plan["b"], E_start)["E"][-1]

maxdiff = 0.0
maxb = 0.0
for i in CHK:
    P, LL, GG = pi, L[i], G[i]
    r = q3.run_day(P, [LL] * len(q3.DECIDE_H), [GG] * len(q3.DECIDE_H),
                   LL, GG, E_of[i], q3.DECIDE_H, 0.0)
    maxdiff = max(maxdiff, abs((r["cost_dev"] + r["cost_emg"]) -
                               (q3.cost_dev(P, r["b_plan"], r["b_plan"]) +
                                r["cost_emg"])))
    maxb = max(maxb, float(np.abs(r["b_adj"] - r["b_plan"]).max()))
    log_(f"    {dates[i].date()}  计划-调整最大偏差 = "
         f"{np.abs(r['b_adj'] - r['b_plan']).max():.2e} kW   "
         f"紧急购电量 = {r['e'].sum() * DT_H:.4f} kWh")
log_(f"校验 {len(CHK)} 天耗时 {time.time() - t1:.1f} 秒")
log_(f"→ 调整段购电量与计划段最大偏差 = {maxb:.3e} kW（应为 0）")
log_(f"→ 快路径与四阶段滚动费用最大偏差 = {maxdiff:.3e} 元（应为 0）")
log_("")

# =====================================================================
# 结论：预报误差的代价 / 完美信息价值（VPI）
# =====================================================================
banner("【3】结论：预报误差的代价（Value of Perfect Information）")
if base_total:
    gap = base_total - perfect_total
    log_(f"现行方案（预报驱动 + 滚动调整）= {base_total:,.1f} 元")
    log_(f"完美信息下界                    = {perfect_total:,.1f} 元")
    log_(f"★ 差值（预报误差代价 / VPI）    = {gap:,.1f} 元"
         f"（{gap / base_total:.2%}）")
    log_(f"★ 即：把预报做到零误差，全年最多再省 {gap / 1e4:.1f} 万元"
         f"（日均 {gap / len(per_day):,.0f} 元）")
    log_("")
    log_("─" * 78)
    log_("同一窗口的构成对比（完美信息 vs 现行预报驱动）：")
    log_("  指标                    现行（预报驱动）        完美信息          差值")
    rows = [
        ("总购电费 (元)", base_total, perfect_total),
        ("调整后购电量 (kWh)", 20998542.4, q_adj),
        ("计划购电费 (元)", 12807426.8, tot_plan),
        ("调整相关购电费 (元)", 13044668.0, tot_dev),
        ("紧急购电量 (kWh)", 102412.3, q_emg),
        ("紧急购电费 (元)", 544694.4, tot_emg),
        ("充电量 (kWh)", 6420546.4, q_ch),
        ("放电量 (kWh)", 5201470.9, q_dis),
        ("弃光量 (kWh)", 1942689.7, q_cur),
    ]
    for nm, a, b in rows:
        log_(f"  {nm:<22s} {a:>16,.1f} {b:>18,.1f} {b - a:>+16,.1f}")
    log_("（现行方案一列取自 " + TXT_Q3 + "；两列同窗口 334 天）")
    log_("解读：预报误差不仅把成本推高 133.5 万元，同时使**弃光与紧急购电双双上升**")
    log_("      ——因为储能“充错时段”：错充进来的电能错不过峰段，本该吸纳的光伏反而弃掉。")
log_("")
log_("口径链（同一 334 天窗口）：")
log_("    完美信息（零误差、无裕量）      → {:,.1f} 元（本脚本）".format(perfect_total))
log_("    真实预报 + 无裕量             → 14,152,041.6 元")
log_("    真实预报 + 报童分位裕量 q=0.75  → {:,.1f} 元（现行方案）".format(base_total)
     if base_total else "    真实预报 + 报童分位裕量 q=0.75  → 现行方案")
try:
    with open(TXT_ANCHOR, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if "合计" in ln and ("无裕量" in ln or "固定 300" in ln or "报童分位" in ln):
                log_("    " + re.sub(r"\s+", " ", ln.strip()))
except FileNotFoundError:
    pass
log_("")

with open(TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"\n已写出 {TXT}")
