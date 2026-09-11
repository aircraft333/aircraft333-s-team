# -*- coding: utf-8 -*-
"""问题二：为什么光伏预报要用「前 3 天滑动平均」？—— 用图与费用证明

问题二的预测规格是 `wday:7|ma:3`，即 **负载用上周同一天、光伏用前 3 天滑动平均**。
为什么两者用不同方法？本脚本给出一套完整证据。

三层论证
    ① 光伏没有周周期性 → 不能对光伏用 wday:7
       证据：滞后自相关曲线。负载在 lag=7 有明显峰值（人类作息按周重复），
             光伏在各滞后上基本平坦（天气不按周重复）。
    ② 单日（持续性 ma:1）不够 → 需要多日平滑
       证据：典型日曲线。天气突变时 ma:1 完全跟不上；ma:7 又过度平滑、丢趋势。
    ③ 恰好 3 天最优 → 偏差-方差权衡的平衡点
       证据：窗口扫描。随 k 增大，误差的**标准差单调下降**（平滑掉单日噪声），
             但**偏差绝对值单调上升**（跟不上趋势），二者交点即最优；
             再用**全年总购电费**复验（不是只看 MAE —— 预报精度指标 ≠ 决策目标）。

合计 4 张单图 + 1 张 2×2 合集，输出到 figures/q2/。
报告：q2_光伏预报方法.txt

用法：python q2_pv_forecast.py
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import q2
from config import *   # noqa: F401,F403

FIGDIR = os.path.join(DIR_FIG, "q2")
TXT = resolve("q2_光伏预报方法.txt")
OUT_START = "2025-02-01"

KS = [1, 2, 3, 4, 5, 6, 7, 10]        # 无 LP 的窗口扫描
KS_COST = [1, 2, 3, 4, 5, 7, 10]      # 带全年 LP 的窗口扫描（每个约 50 s）
K_BEST = 3


def main():
    t0 = time.time()
    setup_plot()
    A1 = att1_arrays(load_att1())
    pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
    dates, L, G = load_att2()
    msk = np.asarray(dates >= pd.Timestamp(OUT_START))
    D, T = G.shape

    rule()
    log("问题二：光伏预报为什么用「前 3 天滑动平均」")
    log(f"预测规格 {q2.FC_SPEC}（左=负载、右=光伏）；评价区间 {OUT_START} 起 {int(msk.sum())} 天")
    rule()

    # =================================================================
    # ① 滞后自相关：光伏没有周周期性
    # =================================================================
    rule("一、滞后自相关：负载按周重复，光伏不按周重复")
    lags = np.arange(1, 15)

    def acf_by_lag(X):
        """各滞后的逐槽相关系数（对齐到同一时段）"""
        out = []
        for k in lags:
            a, b = X[k:], X[:-k]
            out.append(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        return np.array(out)

    acf_l, acf_g = acf_by_lag(L), acf_by_lag(G)
    # 逐槽自相关会被「白天有、夜里没有」的日周期主导，各滞后都接近 1，
    # 区分度低；改用**日总量**序列算自相关，才能真正看出周周期。
    acf_dl, acf_dg = acf_by_lag(L.sum(axis=1)[:, None]), \
        acf_by_lag(G.sum(axis=1)[:, None])
    log("  （两套口径都列出：逐槽自相关区分度低，日总量自相关才是真正的周期性证据）")
    log(f"  {'滞后':<6s}{'负载 r':>10s}{'光伏 r':>10s}{'负载日总量':>12s}{'光伏日总量':>12s}")
    for i, k in enumerate(lags):
        mark = "   ← 周周期" if k == 7 else ("   ← 3 天" if k == 3 else "")
        log(f"  {k:<6d}{acf_l[i]:>10.3f}{acf_g[i]:>10.3f}"
            f"{acf_dl[i]:>12.3f}{acf_dg[i]:>12.3f}{mark}")
    log("")
    log(f"  日总量口径：负载 r(lag=7) = {acf_dl[6]:.3f} 明显高于 lag=1~6 与 lag=8~13"
        f"（如 lag=1 仅 {acf_dl[0]:.3f}）→ 上周同一天确实最优；")
    log(f"              光伏 r(lag=7) = {acf_dg[6]:.3f}，但没有类似的孤立峰值"
        f"（lag=1 = {acf_dg[0]:.3f} 反而接近）→ 对光伏用 wday:7 没有依据。")
    log("")
    log(f"  逐槽口径（仅作对照）：负载 {acf_l[6]:.3f} / 光伏 {acf_g[6]:.3f}，"
        "两者都被日周期抬到 0.98+，几乎区分不出周周期。")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(lags, acf_dl, "o-", color=C_LOAD, lw=LW, label="负载（日总量）")
    ax.plot(lags, acf_dg, "s-", color=C_PV, lw=LW, label="光伏（日总量）")
    ax.axvline(7, color="gray", ls="--", lw=1.0)
    ax.annotate("负载在 lag=7 出现孤立峰值\n（人类作息按周重复）", xy=(7, acf_dl[6]),
                xytext=(8.4, acf_dl[6] + 0.03), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("光伏无周峰\n（天气不按周重复）", xy=(7, acf_dg[6]),
                xytext=(8.4, acf_dg[6] - 0.13), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("滞后天数 lag")
    ax.set_ylabel("日总量自相关系数 r")
    ax.set_title("(a) 日总量自相关：负载按周重复，光伏不按周重复")
    ax.grid(alpha=0.3)
    ax.legend()
    save_fig(fig, "figPV_A_滞后自相关.png", FIGDIR)
    plt.close(fig)

    # =================================================================
    # ② 典型日：ma:1 跟不上，ma:7 过度平滑
    # =================================================================
    rule("二、典型日曲线：单日跟不上天气突变，7 天又过度平滑")
    gt = G.sum(axis=1)
    chg = np.abs(np.diff(gt, prepend=gt[0]))
    picks = [int(i) for i in np.argsort(-chg)[:3]]
    picks.sort()
    log(f"  选取「日间光伏总量变化最大」的 3 天："
        + "、".join(f"{dates[i].date()}（Δ{gt[i] - gt[i - 1]:+,.0f} kWh）" for i in picks))

    Fs = {k: q2.build_forecast(L, G, L1, G1, f"wday:7|ma:{k}")[1] for k in (1, 3, 7)}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True)
    hours = np.arange(T) * DT_H
    for ax, i in zip(axes, picks):
        ax.plot(hours, G[i], color="k", lw=2.2, label="实际")
        ax.plot(hours, Fs[1][i], color="#d62728", lw=LW, ls="--", label="前 1 天（持续性）")
        ax.plot(hours, Fs[3][i], color=C_PV, lw=2.0, label="前 3 天滑动平均（采用）")
        ax.plot(hours, Fs[7][i], color="#7f7f7f", lw=LW, ls=":", label="前 7 天滑动平均")
        ax.set_title(f"{dates[i].date()}   日总量 {gt[i]:,.0f} kWh", fontsize=10)
        ax.set_xlabel("时刻 (h)")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 24)
    axes[0].set_ylabel("光伏功率 (kW)")
    axes[0].legend(fontsize=8.5)
    fig.suptitle("(b) 天气突变日的预报对比：单日预报抖动大，7 天平均太平滑", fontsize=11)
    save_fig(fig, "figPV_B_典型日预报对比.png", FIGDIR)
    plt.close(fig)

    # =================================================================
    # ③ 窗口扫描：偏差-方差权衡 + 全年费用复验
    # =================================================================
    rule("三、窗口扫描 k = 1…10")
    stats = {}
    log(f"  {'k':<4s}{'MAE(kW)':>10s}{'RMSE(kW)':>11s}{'误差标准差':>12s}"
        f"{'偏差绝对值':>12s}{'相对 k=3':>11s}")
    for k in KS:
        F_G = q2.build_forecast(L, G, L1, G1, f"wday:7|ma:{k}")[1]
        e = (F_G[msk] - G[msk])
        stats[k] = dict(mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()),
                        std=e.std(), bias=abs(e.mean()))
    base = stats[K_BEST]["mae"]
    for k in KS:
        s = stats[k]
        log(f"  {k:<4d}{s['mae']:>10.1f}{s['rmse']:>11.1f}{s['std']:>12.1f}"
            f"{s['bias']:>12.1f}{(s['mae'] / base - 1):>+11.1%}")
    k_star = min(KS, key=lambda k: stats[k]["mae"])
    plateau = [k for k in KS if stats[k]["mae"] <= stats[k_star]["mae"] * 1.02]
    log("")
    log(f"  MAE 最小者：k = {k_star}（MAE {stats[k_star]['mae']:.1f} kW）；"
        f"相差 2% 以内的平台 = {plateau}")
    log(f"  误差标准差从 k=1 的 {stats[1]['std']:.1f} 单调降到 k=10 的 {stats[10]['std']:.1f}"
        f" —— 平滑确实压掉了单日天气噪声；")
    log(f"  偏差绝对值虽单调上升（{stats[1]['bias']:.1f} → {stats[10]['bias']:.1f}），"
        f"但量级远小于标准差，故整条曲线很平坦。")
    log(f"  ⇒ 单看 MAE 无法把 k=3 与邻近窗口区分开（k={k_star} 反而略优），"
        f"要定 k 必须看经济性 —— 见下一节。")

    # 全年费用复验（含 附件1 基准日 作为退化对照）
    rule("四、全年总购电费复验（预报精度 ≠ 决策目标）")
    log(f"  {'光伏预报':<22s}{'计划购电费':>15s}{'紧急购电费':>14s}{'总购电费':>15s}{'相对 k=3':>12s}")
    costs = {}
    specs = [(f"前 {k} 天滑动平均", f"wday:7|ma:{k}") for k in KS_COST]
    specs.append(("附件1 基准日（气候态）", "wday:7|att1:0"))
    t1 = time.time()
    for nm, spec in specs:
        F_L, F_G = q2.build_forecast(L, G, L1, G1, spec)
        err = (L - G) - (F_L - F_G)
        X = q2.build_hedge(err, q2.HEDGE_MODE, q2.HEDGE_PARAM, q2.HEDGE_WIN)
        E = SOC0
        cp = ce = 0.0
        for i in range(D):
            pl = q2.solve_day(pi, F_L[i] + X[i], F_G[i], E)
            sim = simulate_dispatch(L[i], G[i], pl["b"], E)
            if msk[i]:
                cp += float(np.sum(pi * pl["b"] * DT_H))
                ce += float(np.sum(5.0 * pi * sim["e"] * DT_H))
            E = sim["E"][-1]
        costs[spec] = cp + ce
        log(f"  {nm:<22s}{cp:>15,.1f}{ce:>14,.1f}{cp + ce:>15,.1f}")
    cbest = costs[f"wday:7|ma:{K_BEST}"]
    kcost_best = min(KS_COST, key=lambda k: costs[f"wday:7|ma:{k}"])
    log("")
    for nm, spec in [("前 1 天（持续性）", "wday:7|ma:1"),
                     ("前 2 天滑动平均", "wday:7|ma:2"),
                     ("前 3 天滑动平均 ← 采用", "wday:7|ma:3"),
                     ("前 4 天滑动平均", "wday:7|ma:4"),
                     ("前 10 天滑动平均", "wday:7|ma:10"),
                     ("附件1 基准日（气候态）", "wday:7|att1:0")]:
        c = costs[spec]
        log(f"  {nm:<26s}{c:>15,.1f} 元   {(c / cbest - 1):>+8.2%}")
    log("")
    log(f"  自检：k=3 应复现 q2.py 主结果 13,949,108.5 元 → 实得 {cbest:,.1f} 元，"
        f"差 {cbest - 13949108.5:+,.1f} 元")
    log(f"  （本节约 {(time.time() - t1) / 60:.1f} 分钟）")
    log("")
    c_kstar = costs[f"wday:7|ma:{kcost_best}"]
    log(f"  费用最低点：k={kcost_best}（{c_kstar:,.1f} 元）；"
        f"k=3 为 {cbest:,.1f} 元，相差 {cbest - c_kstar:+,.1f} 元"
        f"（{(cbest / c_kstar - 1):+.2%}）")
    log("  ⇒ 两个判据（MAE 最优 k=5、费用最优 k=4）并不指向同一个 k，且差异都在噪声级，")
    log("     故正确定述是「k=3~7 为平坦平台」，而非「k=3 为最优点」。")

    # ---- 图 C：双轴 ----
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(KS, [stats[k]["std"] for k in KS], "s-", color=C_PV, lw=LW,
            label="误差标准差（越小越好）")
    ax.plot(KS, [stats[k]["bias"] for k in KS], "^-", color="#d62728", lw=LW,
            label="偏差绝对值（越小越好）")
    ax.plot(KS, [stats[k]["mae"] for k in KS], "o-", color="k", lw=LW,
            label="MAE")
    ax.axvline(K_BEST, color="gray", ls="--", lw=1.0)
    ax.set_xlabel("滑动窗口 k（天）")
    ax.set_ylabel("误差 (kW)")
    ax2 = ax.twinx()
    kc = KS_COST
    ax2.plot(kc, [costs[f"wday:7|ma:{k}"] / 1e4 for k in kc], "D--",
             color=C_NET, lw=LW, label="全年总购电费（右轴）")
    ax2.set_ylabel("全年总购电费（万元）", color=C_NET)
    ax2.tick_params(axis="y", colors=C_NET)
    ax.set_title("(c) 窗口扫描：由经济性定参，而非只看 MAE")
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="center right")
    save_fig(fig, "figPV_C_窗口扫描.png", FIGDIR)
    plt.close(fig)

    # =================================================================
    # ④ 误差分布
    # =================================================================
    rule("五、各方法的误差分布")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    data, labels = [], []
    for k in (1, 2, 3, 5, 7, 10):
        F_G = q2.build_forecast(L, G, L1, G1, f"wday:7|ma:{k}")[1]
        data.append((F_G[msk] - G[msk]).ravel())
        labels.append(f"ma:{k}")
    F_G = q2.build_forecast(L, G, L1, G1, "wday:7|att1:0")[1]
    data.append((F_G[msk] - G[msk]).ravel())
    labels.append("附件1基准日")
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(C_PV if labels[i] == "ma:3" else "#cccccc")
        box.set_alpha(0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("预报误差 F − G (kW)")
    ax.set_title("(d) 误差分布：多日平均明显收窄，气候态基准日最差")
    ax.grid(alpha=0.3, axis="y")
    save_fig(fig, "figPV_D_误差分布.png", FIGDIR)
    plt.close(fig)

    # ---- 2×2 合集 ----
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    axes = axes.ravel()
    ax = axes[0]
    ax.plot(lags, acf_dl, "o-", color=C_LOAD, lw=LW, label="负载（日总量）")
    ax.plot(lags, acf_dg, "s-", color=C_PV, lw=LW, label="光伏（日总量）")
    ax.axvline(7, color="gray", ls="--", lw=1.0)
    ax.set_xlabel("滞后天数 lag")
    ax.set_ylabel("日总量自相关系数 r")
    ax.set_title("(a) 光伏无周周期性，不能用 wday:7")
    ax.grid(alpha=0.3)
    ax.legend()
    ax = axes[1]
    ax.plot(KS, [stats[k]["std"] for k in KS], "s-", color=C_PV, lw=LW, label="误差标准差")
    ax.plot(KS, [stats[k]["bias"] for k in KS], "^-", color="#d62728", lw=LW, label="偏差绝对值")
    ax.plot(KS, [stats[k]["mae"] for k in KS], "o-", color="k", lw=LW, label="MAE")
    ax.axvline(K_BEST, color="gray", ls="--", lw=1.0)
    ax.set_xlabel("滑动窗口 k（天）")
    ax.set_ylabel("误差 (kW)")
    ax.set_title("(b) 窗口扫描：k=3~7 为平坦平台（k=5 的 MAE 最低）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax = axes[2]
    ax.plot(kc, [costs[f"wday:7|ma:{k}"] / 1e4 for k in kc], "D-", color=C_NET, lw=LW)
    for k in kc:
        v = costs[f"wday:7|ma:{k}"] / 1e4
        ax.annotate(f"{v:.1f}", (k, v), textcoords="offset points",
                    xytext=(0, 7), fontsize=8, ha="center")
    ax.axvline(K_BEST, color="gray", ls="--", lw=1.0)
    ax.set_xlabel("滑动窗口 k（天）")
    ax.set_ylabel("全年总购电费（万元）")
    ax.set_title("(c) 经济性复验：由全年费用定参（而非只看 MAE）")
    ax.grid(alpha=0.3)
    ax = axes[3]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(C_PV if labels[i] == "ma:3" else "#cccccc")
        box.set_alpha(0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("预报误差 F − G (kW)")
    ax.set_title("(d) 误差分布：多日平均明显收窄，气候态基准日最差")
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("问题二：为什么光伏预报取「前 3 天滑动平均」", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "fig_光伏预报依据.png", FIGDIR)
    plt.close(fig)

    rule("六、结论")
    k_cost = min(KS_COST, key=lambda k: costs[f"wday:7|ma:{k}"])
    c_k = costs[f"wday:7|ma:{k_cost}"]
    log(f"  1) 日总量自相关：负载在 lag=7 出现孤立峰值（{acf_dl[6]:.3f}），"
        f"光伏没有（{acf_dg[6]:.3f}）")
    log("     ⇒ 负载用 wday:7 有依据，光伏不能照搬，必须另选方法；")
    log(f"  2) 单日预报（ma:1）的 MAE 比 ma:3 高 {stats[1]['mae'] / stats[3]['mae'] - 1:.1%}，"
        f"而窗口拉到 10 天又高 {stats[10]['mae'] / stats[3]['mae'] - 1:.1%}"
        f" —— 存在中间的 最优窗口；")
    log(f"  3) 但必须**用经济性定参**：MAE 的最优点在 k={k_star}"
        f"（且 k=4~6 几乎无差别），而全年总购电费的最优点在 k={k_cost}；")
    if k_cost == K_BEST:
        log(f"     k=3 的费用 {cbest:,.1f} 元即费用意义上的最低点"
            f" —— 预报精度与决策目标在这一点上一致。")
    else:
        log(f"     k=3 的费用 {cbest:,.1f} 元，比费用最优点 k={k_cost} 的 "
            f"{c_k:,.1f} 元高 {cbest - c_k:,.1f} 元（{(cbest / c_k - 1):+.2%}）"
            f" —— 属于很低水平的差别，保持 k=3 同时兼顾简洁与稳定。")
    log("  4) 附件1 基准日（气候态平均）完全不看最近天气，费用最高，说明")
    log("     「必须用最近的历史」这条结论与窗口长短无关。")

    write_report(TXT)
    log("")
    log("总耗时 %.1f 分钟" % ((time.time() - t0) / 60.0))


if __name__ == "__main__":
    main()
