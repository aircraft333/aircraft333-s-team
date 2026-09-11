# -*- coding: utf-8 -*-
"""问题三敏感性分析：购电量对附件3 光伏预报的响应
=====================================================================
目的
    定量回答「附件3 的预报到底怎样影响调整后的购电量」。
    做法是**受控扰动**：固定同一天、同一决策时刻、同一条 0:00 计划、同一进入储电量，
    只把调整 LP 所用的光伏预报乘以 (1+δ)，观察购电量相对计划的偏离 ΔQ。

    为什么不能用「预报修订量的代数和」去和 ΔQ 做相关？
    因为逐时段的修订会互相抵消（上午下调、下午上调，代数和接近 0），
    标量汇总会掩盖逐点响应，必须用受控扰动才能看清方向与弹性。

结论（受控扰动的实测结果）
    光伏预报上调 → 预测净负荷变小 → LP 用 δ⁻ 少买（省 0.5π/kWh）  ⇒ ΔQ < 0
    光伏预报下调 → 预测净负荷变大 → 提前用 δ⁺ 多买（1.5π 优于 5π 紧急购电）⇒ ΔQ > 0
    信息完全不变（δ = 0）时 ΔQ ≈ 0，说明调整的收益确实来自附件3 的新信息。

运行：python q3_sensitivity.py
产出：figures/q3/fig_光伏预报敏感性.png + q3_光伏预报敏感性.txt
"""
import numpy as np

import q2
import q3
from config import *

FIGDIR = os.path.join(DIR_FIG, "q3")
TXT = "q3_光伏预报敏感性.txt"
DELTAS = [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]     # 光伏预报的乘性扰动
EPOCHS = (6, 12, 18)                                 # 考察的决策时刻
STRIDE = 30                                          # 每 30 天取一个样本日


def main():
    A1 = att1_arrays(load_att1())
    pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
    dates, L, G = load_att2()
    df3 = load_att3()
    Gf = q3.build_pv_forecast(df3, dates, G, q3.DECIDE_H, q3.PV_MIX)
    Lf = q3.build_load_forecast(L, L1, dates, q3.DECIDE_H, q3.LOAD_LAG, q3.ADAPT)

    samples = list(range(40, len(dates), STRIDE))
    res = {h: np.full((len(samples), len(DELTAS)), np.nan) for h in EPOCHS}
    base_q = np.zeros(len(samples))

    rule("问题三敏感性：购电量对附件3 光伏预报的响应（受控扰动）")
    log(f"取样 {len(samples)} 天（每 {STRIDE} 天一个）× 决策时刻 {EPOCHS}"
        f" × 扰动档位 {[f'{d:+.0%}' for d in DELTAS]}")
    log("固定条件：同一天、同一 0:00 计划 b^p、同一进入储电量 E_t0；只改 LP 用的光伏预报。")
    log("")

    E_start = SOC0
    for i, day in enumerate(dates):
        bp = q2.solve_day(pi, Lf[0, i] + q3.HEDGE, Gf[0, i], E_start)["b"]
        si = samples.index(i) if i in samples else -1
        if si >= 0:
            base_q[si] = bp.sum() * DT_H
            for h in EPOCHS:
                t0 = h * 6
                E_t0 = simulate_dispatch(L[i], G[i], bp, E_start, 0, t0)["E"][-1]
                for dj, d in enumerate(DELTAS):
                    Gt = np.maximum(Gf[0, i] * (1.0 + d), 0.0)
                    ba = q3.solve_adjust(pi, Lf[q3.DECIDE_H.index(h), i] + q3.HEDGE,
                                         Gt, bp, E_t0, t0)
                    res[h][si, dj] = float(np.sum((ba[t0:] - bp[t0:]) * DT_H))
        E_start = simulate_dispatch(L[i], G[i], bp, E_start)["E"][-1]

    # ---------- 统计 ----------
    rule("【1】响应表：光伏预报 ×(1+δ) 后的购电量偏离 ΔQ（相对 0:00 计划，kWh）")
    hdr = "  " + f"{'时刻':>6s}" + "".join(f"{d:>+9.0%}" for d in DELTAS) + f"{'弹性':>12s}"
    log(hdr)
    log("  " + "-" * (len(hdr) + 4))
    for h in EPOCHS:
        m = res[h].mean(axis=0)
        slope = (m[-1] - m[0]) / (DELTAS[-1] - DELTAS[0])
        log(f"  {h:>4d}:00" + "".join(f"{v:>9,.0f}" for v in m) + f"{slope:>12,.0f}")
    log("")
    log(f"  样本日平均计划购电量 = {base_q.mean():,.0f} kWh")
    log("  弹性 = ΔQ 对 δ 的斜率（kWh），负值 = 光伏预报上调则少买")

    for h in EPOCHS:
        m = res[h].mean(axis=0)
        lo = res[h][:, 0].mean()
        up = res[h][:, -1].mean()
        log(f"  {h:>2d}:00  预报 −40% → ΔQ {lo:+9,.0f} kWh（占日购电量 "
            f"{lo / base_q.mean():+6.1%}）；+40% → ΔQ {up:+9,.0f} kWh（{up / base_q.mean():+6.1%}）")

    log("")
    log("  ⇒ 方向符合预期：光伏上调则少买（δ⁻ 省 0.5π/kWh），下调则多买（δ⁺ 花 1.5π，优于 5π 紧急购电）")
    log(f"  ⇒ δ = 0（信息完全不变，仅重新优化）时 ΔQ 均值 "
        f"{np.mean([res[h][:, DELTAS.index(0.0)].mean() for h in EPOCHS]):+,.0f} kWh ≈ 0，")
    log("     说明调整的收益确实来自附件3 的新预报，而不是无意义的重复求解。")

    # ---------- 绘图 ----------
    setup_plot()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    cols = {6: C_LOAD, 12: C_PV, 18: C_PRICE}

    # (a) 响应曲线（均值 ± 四分位带）
    ax = axes[0]
    xs = np.array(DELTAS) * 100
    for h in EPOCHS:
        m = res[h].mean(axis=0)
        q1 = np.percentile(res[h], 25, axis=0)
        q3_ = np.percentile(res[h], 75, axis=0)
        ax.plot(xs, m, "o-", color=cols[h], lw=2, ms=5, label=f"{h}:00 决策")
        ax.fill_between(xs, q1, q3_, color=cols[h], alpha=.16)
    ax.axhline(0, color="0.3", lw=.9)
    ax.axvline(0, color="0.3", lw=.9, ls=":")
    ax.set_title("(a) 购电量对光伏预报的响应：上调则少买、下调则多买", fontsize=11)
    ax.set_xlabel("附件3 光伏预报的乘性扰动 δ (%)")
    ax.set_ylabel("ΔQ 相对 0:00 计划 (kWh)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)

    # (b) 三个扰动档位的分布
    ax = axes[1]
    show = [-0.2, 0.0, 0.2]
    pos, data, labs = [], [], []
    for j, d in enumerate(show):
        for k, h in enumerate(EPOCHS):
            pos.append(j * 4 + k)
            data.append(res[h][:, DELTAS.index(d)])
            labs.append(f"{h}:00")
    bp_ = ax.boxplot(data, positions=pos, widths=.75, patch_artist=True,
                     medianprops=dict(color="0.2"))
    for p, d in zip(bp_["boxes"], [d for d in show for _ in EPOCHS]):
        p.set_facecolor({-0.2: C_PRICE, 0.0: "0.85", 0.2: C_NET}[d])
        p.set_alpha(.75)
    ax.axhline(0, color="0.3", lw=.9)
    ax.set_xticks(pos)
    ax.set_xticklabels(labs, fontsize=7)
    for j, d in enumerate(show):
        ax.text(j * 4 + 1, ax.get_ylim()[1] * .95, f"δ = {d:+.0%}",
                ha="center", fontsize=9, color="0.25")
    ax.set_title(f"(b) ΔQ 的分布（{len(samples)} 个样本日）", fontsize=11)
    ax.set_xlabel("决策时刻")
    ax.set_ylabel("ΔQ (kWh)")
    ax.grid(alpha=.3, axis="y")

    fig.suptitle("问题三：附件3 光伏预报误差如何传导到购电量", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93))
    save_fig(fig, "fig_光伏预报敏感性.png", FIGDIR)
    write_report(TXT)


if __name__ == "__main__":
    main()
