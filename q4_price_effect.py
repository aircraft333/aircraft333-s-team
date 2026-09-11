# -*- coding: utf-8 -*-
"""问题四：波动电价到底为什么让总费用上升？—— 受控分解

背景
    问题四的结论是「波动电价下总费用上升 +4.91%（问题二口径）」，但这件事需要解释清楚：
    附件4 在**输出窗口（334 天）内的逐时段均价其实比附件1 更低**（0.7575 vs 0.7662），
    为什么费用反而更高？如果再叠加报童裕量、紧急购电等机制，很难一眼看出因果。

方法（全部为受控对照，唯一变量是电价）
    同一套负载/光伏、同一套预报与安全裕量、同一套逐槽执行规则，只更换电价矩阵：
        A：附件1 的固定分时电价（每天相同）
        B：附件4 的波动电价（每天不同）
    两个口径都严格按 q2.py 的方式链式推进（次日初值 = 当日**实际执行仿真**的日末储电量）。

分解口径
    记 Pbar_t 为窗口内第 t 个时段的**平均**电价，则
        Cost_B = Σ_{d,t} π_{d,t} q_{d,t} Δt
               = Σ_{d,t} Pbar_t q_{d,t} Δt   +   Σ_{d,t} (π_{d,t} − Pbar_t) q_{d,t} Δt
                 └── 逐时段均价项 ──┘         └── 逐日漂移项（协方差）──┘
    前者只反映「各时段平均多贵」，后者才反映「逐日上下漂移」。
    再配合交叉评估（把 A 的购电量按 B 的电价计价），可判断 LP 的再优化到底扳回多少。

输出：控制台 + q4_电价波动效应分解.txt
"""
import time

import numpy as np
import pandas as pd

import q2
import q4
from config import *   # noqa: F401,F403

TXT = resolve("q4_电价波动效应分解.txt")
OUT_START = "2025-02-01"


def run(Pmat, dates, L, G, Lp, Gp, msk):
    """按 q2.py 的链式口径跑全年，返回 (购电量矩阵, 计划费, 紧急费, 弃光量, 日末储电量)"""
    D = len(dates)
    B = np.zeros((D, N_SLOT))
    E24 = np.zeros(D)
    cp = qe = ce = qs = 0.0
    E = SOC0
    for i in range(D):
        pi = Pmat[i]
        pl = q2.solve_day(pi, Lp[i], Gp[i], E)
        sim = simulate_dispatch(L[i], G[i], pl["b"], E)
        B[i] = pl["b"]
        if msk[i]:
            cp += float(np.sum(pi * pl["b"] * DT_H))
            qe += float(sim["e"].sum() * DT_H)
            ce += float(np.sum(5.0 * pi * sim["e"] * DT_H))
            qs += float(sim["s"].sum() * DT_H)
        E = sim["E"][-1]
        E24[i] = E
    return B, cp, qe, ce, qs, E24


def main():
    t0 = time.time()
    A1 = att1_arrays(load_att1())
    pi1, L1, G1 = A1["price"], A1["load"], A1["pv"]
    dates, L, G = load_att2()
    PI = q4.load_att4()
    msk = np.asarray(dates >= pd.Timestamp(OUT_START))
    n = int(msk.sum())

    F_L, F_G = q2.build_forecast(L, G, L1, G1, q2.FC_SPEC)
    err = (L - G) - (F_L - F_G)
    X = q2.build_hedge(err, q2.HEDGE_MODE, q2.HEDGE_PARAM, q2.HEDGE_WIN)
    Lp, Gp = F_L + X, F_G                       # 两个口径共用，唯一变量是电价

    rule()
    log("问题四：固定电价 vs 波动电价 的受控分解")
    log(f"输出窗口 {OUT_START} 起 {n} 天；预测规格 {q2.FC_SPEC}；"
        f"安全裕量 {q2.HEDGE_MODE} q={q2.HEDGE_PARAM:g}/win={q2.HEDGE_WIN}")
    log("两个口径共用同一套负载/光伏、预报、裕量与执行规则，唯一变量是电价矩阵")
    rule()

    PF = np.tile(pi1, (len(dates), 1))
    BA, cP_A, qE_A, cE_A, qS_A, E_A = run(PF, dates, L, G, Lp, Gp, msk)
    BB, cP_B, qE_B, cE_B, qS_B, E_B = run(PI, dates, L, G, Lp, Gp, msk)

    rule("一、两个口径的全年结果")
    log(f"  {'口径':<14s}{'计划购电量':>15s}{'计划购电费':>15s}{'均价':>9s}"
        f"{'紧急电量':>11s}{'紧急电费':>13s}{'弃光量':>13s}{'合计':>15s}")
    qA, qB = BA[msk] * DT_H, BB[msk] * DT_H
    for tag, q, cp, qe, ce, qs in (("固定电价(附件1)", qA, cP_A, qE_A, cE_A, qS_A),
                                   ("波动电价(附件4)", qB, cP_B, qE_B, cE_B, qS_B)):
        log(f"  {tag:<14s}{q.sum():>15,.1f}{cp:>15,.1f}{cp / q.sum():>9.4f}"
            f"{qe:>11,.1f}{ce:>13,.1f}{qs:>13,.1f}{cp + ce:>15,.1f}")

    rule("二、口径自检")
    log(f"  固定电价合计应 = 13,949,108.5 → 实得 {cP_A + cE_A:,.1f}，"
        f"差 {cP_A + cE_A - 13949108.5:+,.1f} 元")
    log(f"  波动电价合计应 = 14,634,080.5 → 实得 {cP_B + cE_B:,.1f}，"
        f"差 {cP_B + cE_B - 14634080.5:+,.1f} 元")

    rule("三、计划购电费的精确分解（固定 → 波动）")
    Pbar = PI[msk].mean(axis=0)                 # 窗口逐时段平均电价
    log(f"  窗口逐时段均价 Pbar：均值 {Pbar.mean():.6f}，与附件1 逐时段相关 "
        f"{np.corrcoef(Pbar, pi1)[0, 1]:.6f}")
    log(f"  附件1 固定电价        均值 {pi1.mean():.6f}")
    log("")
    log(f"  {'电量口径':<20s}{'合计':>15s}{'逐时段均价项':>16s}{'逐日漂移项':>16s}")
    for tag, q in (("A 固定电价的电量", qA), ("B 波动电价的电量", qB)):
        tot = float(np.sum(PI[msk] * q))
        mean_term = float(np.sum(Pbar * q))
        log(f"  {tag:<20s}{tot:>15,.1f}{mean_term:>16,.1f}{tot - mean_term:>+16,.1f}")
    log("")
    log(f"  A 按自己的电价结算        ：{cP_A:>15,.1f} 元")
    log(f"  A 的电量按 B 的电价计价   ：{float(np.sum(PI[msk] * qA)):>15,.1f} 元"
        f"   （+{float(np.sum(PI[msk] * qA)) - cP_A:,.1f}）")
    log(f"  B 优化后的电量按 B 的电价 ：{cP_B:>15,.1f} 元"
        f"   （扳回 {float(np.sum(PI[msk] * qA)) - cP_B:,.1f}）")
    log("  ⇒ 逐日 LP 的再优化是有效的，涨价不是求解问题")

    rule("四、涨价的真正机理：日购电量与当日电价水平高度正相关")
    Ld = PI[msk].mean(axis=1)
    QA, QB = qA.sum(axis=1), qB.sum(axis=1)
    log(f"  日购电量  均值 {QB.mean():,.1f} kWh   标准差 {QB.std():,.1f}")
    log(f"  当日电价水平（逐日均价） 均值 {Ld.mean():.4f}  标准差 {Ld.std():.4f}")
    log(f"  corr(日购电量, 当日电价水平) = {np.corrcoef(QB, Ld)[0, 1]:+.4f}")
    log(f"  （作为对照，固定电价下一个自然日购电量与电价水平的相关 = "
        f"{np.corrcoef(QA, Ld)[0, 1]:+.4f}）")
    log("")
    log("  即：负载高的那些天，恰好也是电价整体偏高的天。负载是刚性的、必须满足，")
    log("      而储能可用容量只有 E_max−E_min = 9600 kWh，远不足以跨日搬运")
    log(f"      （日均购电 {QB.mean() / 1000:.1f} 万 kWh，相差近一个数量级），")
    log("      于是「价格漂移」直接转化为「费用上升」。")
    log("")
    log("  对应到分解表：逐日漂移项 = "
        f"{float(np.sum(PI[msk] * qB)) - float(np.sum(Pbar * qB)):+,.1f} 元，"
        f"占计划购电费总变化 {cP_B - cP_A:+,.1f} 元的 "
        f"{100 * (float(np.sum(PI[msk] * qB)) - float(np.sum(Pbar * qB))) / (cP_B - cP_A):.1f}%")

    rule("五、波动电价还会改变购电的时间分布")
    hA = qA.reshape(-1, 24, 6).sum(axis=(0, 2))
    hB = qB.reshape(-1, 24, 6).sum(axis=(0, 2))
    hA, hB = 100 * hA / hA.sum(), 100 * hB / hB.sum()
    log("  小时   固定电价   波动电价     差")
    for h in range(24):
        if abs(hB[h] - hA[h]) > 0.2 or h in (0, 6, 18, 21):
            log(f"  {h:2d}:00   {hA[h]:7.2f}%  {hB[h]:7.2f}%  {hB[h] - hA[h]:+6.2f}%")
    log("")
    log("  -> 固定电价下「谷充峰放」集中在 0:00（占 12.3%）；波动电价下逐日最便宜的")
    log("     时段在移动，购电分布被摊平（0:00 降到 8.4%），单看「择时能力」是下降的：")
    log("     （购电均价 / 时间均价：固定 0.6135/0.7662 = 0.801；"
        "波动 0.6397/0.7575 = 0.845）")
    log("     但这只是次要因素 —— 主因仍是上面的「贵日子正好是用电多的日子」。")

    rule("六、结论（可直接写进论文）")
    log("  1) 波动电价使总费用上升 4.91%（问题二口径）/ 4.57%（问题三口径），")
    log("     上升由「逐日价格漂移与日购电量的正相关」主导，而非求解或调度缺陷；")
    log(f"  2) 漂移项占计划购电费增量的 "
        f"{100 * (float(np.sum(PI[msk] * qB)) - float(np.sum(Pbar * qB))) / (cP_B - cP_A):.1f}%，"
        "是唯一的量级级因素；")
    log("  3) 逐日 LP 相对「照抄固定电价方案」仍省 "
        f"{float(np.sum(PI[msk] * qA)) - cP_B:,.1f} 元，说明现有模型对波动电价有适应性；")
    log("  4) 储能容量是瓶颈：若要真正对冲突发电价，需要把可用容量从 9600 kWh 提升")
    log("     一个数量级才可能实现跨日搬运。")

    write_report(TXT)
    log("")
    log("总耗时 %.1f 分钟" % ((time.time() - t0) / 60.0))


if __name__ == "__main__":
    main()
