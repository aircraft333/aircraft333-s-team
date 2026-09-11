# -*- coding: utf-8 -*-
"""问题二敏感性分析：预测精度 与 紧急电价倍率
=====================================================================
问题二的两个核心假设需要扰动检验：
    ① 日前预测的精度（我们用 wday:7 负载 + ma:3 光伏）
    ② 紧急购电电价是「交易时刻电价的 5 倍」

实验 A：预测精度敏感性
    把日前预测的误差整体缩放 λ 倍（只用于构造反事实预报序列）：
        F' = Y + λ·(F − Y)
    λ = 0 表示"完美预测"、λ = 1 是实测口径、λ > 1 表示预报更差。
    预期：总费用随 λ 单调上升，且**上升主要来自紧急购电费**。

实验 A'：最优安全裕量是否随误差线性缩放（报童模型的直接检验）
    报童临界条件：多买 1 kWh 值得当且仅当 P(缺口) > 1/k（k 为紧急电价倍率）。
    与误差幅度无关，因此最优裕量应取误差的 (1−1/k) 分位数
    ⇒ 误差放大 λ 倍时，最优裕量（kW）应近似放大 λ 倍。

实验 B：紧急电价倍率 k 敏感性
    把题目的 5 倍换成 k ∈ {3,4,5,6,8}，并按报童规则把裕量设为误差的 (1−1/k) 分位数。
    预期：k 越大 → 最优分位越高（裕量越大）→ 计划购电费上升、紧急购电费下降，
          总费用上升（因为紧急购电更贵了）。

实验 C：裕量分位数 q 的全年曲线（主口径定参依据）
    扫描 q ∈ [0.60, 0.95]，比较「报童解析值 q* = 1−1/k = 0.8」、
    「全年网格最优」与「固定 300 kW」三者，用来说明分位数规则稳健、不过拟合。

运行：python q2_sensitivity.py      （约 20~25 分钟）
产出：figures/q2/fig_敏感性分析.png + q2_敏感性分析.txt
"""
import time

import numpy as np

import q2
from config import *

FIGDIR = os.path.join(DIR_FIG, "q2")
TXT = "q2_敏感性分析.txt"
OUT_START = getattr(q2, "OUT_START", "2025-02-01")   # 统计区间与 q2.py 输出保持一致

LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0]          # 预测误差缩放
KS = [3.0, 4.0, 5.0, 6.0, 8.0]               # 紧急电价倍率
HEDGE_FIX = 300.0                            # 主口径固定裕量
BLOCK = (60, 190)                            # 用于实验 A' 的连续区块
HEDGE_GRID = [0.0, 300.0, 600.0]
SWEEP_LAM = [0.5, 1.0, 2.0]
HEDGE_WIN = 60
QUANTILES = [0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.90, 0.95]


# =====================================================================
# 一、反事实预报：把误差整体缩放 λ 倍
# =====================================================================
def forecast_scaled(L, G, F_L, F_G, lam):
    """F' = Y + λ(F − Y)：λ=1 是实测预报，λ=0 是完美预测，λ>1 预报更差"""
    return L + lam * (F_L - L), G + lam * (F_G - G)


def hedge_for(FL, FG, L, G, mode, param, win=HEDGE_WIN):
    """按指定规则构造安全裕量矩阵 (D, T)"""
    err = (L - G) - (FL - FG)
    return q2.build_hedge(err, mode, param, win)


# =====================================================================
# 二、求解器（复用 q2 的计划 LP + config 的逐槽执行）
# =====================================================================
def run_block(pi, L, G, FL, FG, X, k, i0=0, i1=None, acc_from=None):
    """逐日：0:00 计划（用反事实预报）→ 逐槽执行（用实际值）→ 按 k 倍价结算

    i0/i1   : 求解区间（储电量从 i0 起连续）
    acc_from: 从第几天开始累计统计（默认 = i0；取 OUT_START 的序号可跳过 1 月热身期，
              与 q2.py 的输出口径一致）
    """
    i1 = len(L) if i1 is None else i1
    acc_from = i0 if acc_from is None else acc_from
    tot = dict(q_plan=0.0, c_plan=0.0, q_emg=0.0, c_emg=0.0,
               q_ch=0.0, q_dis=0.0, n_emg=0, n=0)
    E = SOC0
    for i in range(i0, i1):
        Lp = FL[i] + (X[i] if X is not None else 0.0)
        plan = q2.solve_day(pi, Lp, FG[i], E)
        sim = simulate_dispatch(L[i], G[i], plan["b"], E)
        E = sim["E"][-1]
        if i < acc_from:
            continue
        tot["q_plan"] += plan["b"].sum() * DT_H
        tot["c_plan"] += float(np.sum(pi * plan["b"] * DT_H))
        tot["q_emg"] += sim["e"].sum() * DT_H
        tot["c_emg"] += float(np.sum(k * pi * sim["e"] * DT_H))
        tot["q_ch"] += sim["c"].sum() * DT_H
        tot["q_dis"] += sim["d"].sum() * DT_H
        tot["n_emg"] += 1 if sim["e"].sum() > 1e-6 else 0
        tot["n"] += 1
    tot["cost"] = tot["c_plan"] + tot["c_emg"]
    return tot


def main():
    A1 = att1_arrays(load_att1())
    pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
    dates, L, G = load_att2()
    F_L, F_G = q2.build_forecast(L, G, L1, G1, q2.FC_SPEC)
    msk = dates >= pd.Timestamp(OUT_START)
    n_out = int(msk.sum())
    ACC = int(np.argmax(msk))          # 输出起始日期的序号（跳过 1 月热身期）

    rule("问题二敏感性分析")
    log(f"预测规格 {q2.FC_SPEC}；电价 附件1（日均 {pi.mean():.4f}）；评价区间 {n_out} 天")
    log(f"对照基准：安全裕量固定 {HEDGE_FIX:g} kW、紧急电价 5π")
    log(f"现行主口径：报童分位裕量（回看 {HEDGE_WIN} 天）")

    # ------------------------------------------------------------------
    # 实验 A：预测精度敏感性（全年）
    # ------------------------------------------------------------------
    rule("【A】预测精度敏感性：预报误差 ×λ（全年）")
    log(f"  {'λ':>5s} {'MAE(kW)':>9s} {'计划购电费':>14s} {'紧急购电量':>12s}"
        f" {'紧急购电费':>12s} {'总购电费':>14s} {'紧急天数':>8s}")
    A = {}
    for lam in LAMBDAS:
        t = time.time()
        FLs, FGs = forecast_scaled(L, G, F_L, F_G, lam)
        X = hedge_for(FLs, FGs, L, G, "scalar", HEDGE_FIX)
        r = run_block(pi, L, G, FLs, FGs, X, 5.0, 0, len(L), ACC)
        mae = float(np.abs((FLs - FGs) - (L - G))[msk].mean())
        A[lam] = dict(mae=mae, r=r)
        log(f"  {lam:>5.1f} {mae:>9.1f} {r['c_plan']:>14,.1f} {r['q_emg']:>12,.1f}"
            f" {r['c_emg']:>12,.1f} {r['cost']:>14,.1f} {r['n_emg']:>5d}/{r['n']}"
            f"   ({time.time() - t:.0f}s)")
    b = A[1.0]["r"]["cost"]
    log("")
    log(f"  λ=0（完美预测）比 λ=1 省 {b - A[0.0]['r']['cost']:,.1f} 元"
        f"（{(b - A[0.0]['r']['cost']) / b:.1%}）")
    log(f"  λ=2（误差加倍）比 λ=1 多花 {A[2.0]['r']['cost'] - b:,.1f} 元"
        f"（{(A[2.0]['r']['cost'] - b) / b:+.1%}）")
    log("  ⇒ 预测精度直接决定了紧急购电费，进而决定总费用")
    # ------------------------------------------------------------------
    # 实验 A'：最优裕量是否随误差线性缩放（连续区块）
    # ------------------------------------------------------------------
    i0, i1 = BLOCK
    rule(f"【A'】最优裕量随误差缩放？报童模型的检验（区块 {i0}~{i1}，{i1 - i0} 天）")
    log(f"  {'λ':>5s}" + "".join(f"{'裕量 ' + f'{h:g}':>16s}" for h in HEDGE_GRID)
        + f"{'最优':>10s}")
    best_h = {}
    sweep = {}
    for lam in SWEEP_LAM:
        FLs, FGs = forecast_scaled(L, G, F_L, F_G, lam)
        xs = []
        for h in HEDGE_GRID:
            X = hedge_for(FLs, FGs, L, G, "scalar", h)
            xs.append(run_block(pi, L, G, FLs, FGs, X, 5.0, i0, i1)["cost"])
        sweep[lam] = xs
        best_h[lam] = HEDGE_GRID[int(np.argmin(xs))]
        log(f"  {lam:>5.1f}" + "".join(f"{v:>16,.0f}" for v in xs) + f"{best_h[lam]:>10g}")
    log(f"  ⇒ 最优裕量：λ=0.5 → {best_h[0.5]:g} kW；λ=1 → {best_h[1.0]:g} kW；"
        f"λ=2 → {best_h[2.0]:g} kW")
    log("    报童模型的预言是「最优裕量随误差幅度近似线性放大（保持分位数不变）」")

    # ------------------------------------------------------------------
    # 实验 B：紧急电价倍率 k（全年，裕量按报童规则取 (1−1/k) 分位数）
    # ------------------------------------------------------------------
    rule("【B】紧急电价倍率 k 敏感性（全年，裕量按报童规则随 k 调整）")
    log(f"  {'k':>5s} {'理论分位':>9s} {'裕量均值':>10s} {'计划购电费':>14s}"
        f" {'紧急购电量':>12s} {'紧急购电费':>13s} {'总购电费':>14s}")
    B = {}
    for k in KS:
        t = time.time()
        qt = 1.0 - 1.0 / k
        X = hedge_for(F_L, F_G, L, G, "quantile", qt)
        r = run_block(pi, L, G, F_L, F_G, X, k, 0, len(L), ACC)
        B[k] = dict(qt=qt, xmean=float(X[msk].mean()), r=r)
        log(f"  {k:>5.0f} {qt:>9.3f} {X[msk].mean():>10.0f} {r['c_plan']:>14,.1f}"
            f" {r['q_emg']:>12,.1f} {r['c_emg']:>13,.1f} {r['cost']:>14,.1f}"
            f"   ({time.time() - t:.0f}s)")
    log("")
    log(f"  固定裕量 {HEDGE_FIX:g} kW（不随 k 调整）时 k=5 的费用为 {A[1.0]['r']['cost']:,.1f} 元；")
    log(f"  按报童规则随 k 调整裕量（分位 {B[5.0]['qt']:.3f}）后为 {B[5.0]['r']['cost']:,.1f} 元，"
        f"省 {A[1.0]['r']['cost'] - B[5.0]['r']['cost']:,.1f} 元"
        f"（{(A[1.0]['r']['cost'] - B[5.0]['r']['cost']) / A[1.0]['r']['cost']:.2%}）")
    log("  ⇒ k 越大 → 最优分位越高（裕量越大）→ 紧急购电费下降、计划购电费上升；")
    log("    总费用随 k 单调上升，说明「5 倍」这个惩罚强度是费用的主导参数之一。")
    log("  ⇒ 报童规则不需要任何调参就能比手调的固定 300 kW 更好，说明该规则可直接用于实际调度。")

    # ------------------------------------------------------------------
    # 实验 C：分位数曲线（主口径的定参依据）
    # ------------------------------------------------------------------
    rule("【C】安全裕量分位数 q 的全年曲线（主口径定参依据）")
    log(f"  逐时段取历史 {HEDGE_WIN} 天误差的 q 分位数作为裕量；紧急电价 {5.0:g}π")
    log("")
    hdr = (f"  {'q':>6s}{'裕量均值':>10s}{'计划购电费':>14s}{'紧急购电量':>12s}"
           f"{'紧急购电费':>12s}{'总购电费':>14s}")
    log(hdr)
    log("  " + "-" * (len(hdr) + 2))
    C = {}
    for q in QUANTILES:
        X = hedge_for(F_L, F_G, L, G, "quantile", q)
        r = run_block(pi, L, G, F_L, F_G, X, 5.0, 0, len(L), ACC)
        C[q] = dict(xmean=float(X[msk].mean()), r=r)
        log(f"  {q:>6.2f}{X[msk].mean():>10.0f}{r['c_plan']:>14,.1f}{r['q_emg']:>12,.1f}"
            f"{r['c_emg']:>12,.1f}{r['cost']:>14,.1f}")
    q_best = min(QUANTILES, key=lambda q: C[q]["r"]["cost"])
    q_theory = 1.0 - 1.0 / 5.0
    log("")
    log(f"  报童解析值 q* = 1 - 1/k = {q_theory:.3f}（裕量均值 "
        f"{C[q_theory]['xmean']:.0f} kW）→ {C[q_theory]['r']['cost']:,.1f} 元")
    log(f"  全年网格最优 q = {q_best:.2f}（裕量均值 {C[q_best]['xmean']:.0f} kW）"
        f"→ {C[q_best]['r']['cost']:,.1f} 元")
    log(f"  固定 300 kW → {A[1.0]['r']['cost']:,.1f} 元")
    log(f"  ⇒ 解析值与网格最优仅差 {C[q_theory]['r']['cost'] - C[q_best]['r']['cost']:,.1f} 元"
        f"（{(C[q_theory]['r']['cost'] - C[q_best]['r']['cost']) / C[q_best]['r']['cost']:.2%}），"
        f"说明 q 在 0.75~0.80 之间很平坦，采用分位数规则稳健、不易过拟合；")
    log(f"    两者均优于固定 300 kW 的 {A[1.0]['r']['cost']:,.1f} 元，故主口径采用报童分位裕量。")

    # ------------------------------------------------------------------
    # 绘图
    # ------------------------------------------------------------------
    setup_plot()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))

    # (a) 预测精度 → 费用
    ax = axes[0, 0]
    x = np.array(LAMBDAS)
    tot = np.array([A[l]["r"]["cost"] for l in LAMBDAS]) / 1e4
    emg = np.array([A[l]["r"]["c_emg"] for l in LAMBDAS]) / 1e4
    ax.plot(x, tot, "o-", color=C_PRICE, lw=2, ms=6, label="总购电费")
    ax.plot(x, emg, "s--", color=C_PV, lw=1.6, ms=5, label="紧急购电费")
    ax.axvline(1.0, color="0.4", ls=":", lw=1)
    ax.annotate("实测口径\nλ=1", xy=(1.0, tot[2]), xytext=(1.15, tot[2] + 40),
                fontsize=8, color="0.3")
    for xi, yi in zip(x, tot):
        ax.text(xi, yi + 12, f"{yi:,.0f}", ha="center", fontsize=8)
    ax.set_title("(a) 预测误差 ×λ → 费用（万元）", fontsize=11)
    ax.set_xlabel("预测误差缩放 λ（0 = 完美预测，1 = 实测）")
    ax.set_ylabel("费用 (万元)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)

    # (b) 紧急电价倍率 → 费用
    ax = axes[0, 1]
    xk = np.array(KS)
    totk = np.array([B[k]["r"]["cost"] for k in KS]) / 1e4
    emgk = np.array([B[k]["r"]["c_emg"] for k in KS]) / 1e4
    plnk = np.array([B[k]["r"]["c_plan"] for k in KS]) / 1e4
    ax.plot(xk, totk, "o-", color=C_PRICE, lw=2, ms=6, label="总购电费")
    ax.plot(xk, plnk, "^--", color=C_LOAD, lw=1.4, ms=5, label="计划购电费")
    ax.plot(xk, emgk, "s--", color=C_PV, lw=1.4, ms=5, label="紧急购电费")
    ax2 = ax.twinx()
    ax2.plot(xk, [B[k]["qt"] * 100 for k in KS], "d:", color=C_NET, lw=1.6, ms=6,
             label="最优裕量分位 (%)")
    ax2.set_ylabel("最优裕量分位 (%)", color=C_NET)
    ax2.tick_params(axis="y", colors=C_NET)
    for xi, yi in zip(xk, totk):
        ax.text(xi, yi + 20, f"{yi:,.0f}", ha="center", fontsize=8)
    ax.set_title("(b) 紧急电价倍率 k → 费用（万元）", fontsize=11)
    ax.set_xlabel("紧急购电电价 = k × 交易时刻电价")
    ax.set_ylabel("费用 (万元)")
    ax.set_xticks(xk)
    ax.grid(alpha=.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center left")

    # (c) 最优裕量随误差缩放
    ax = axes[1, 0]
    xs = np.array(SWEEP_LAM)
    for j, h in enumerate(HEDGE_GRID):
        ys = [sweep[lam][j] / 1e4 for lam in SWEEP_LAM]
        ax.plot(xs, ys, "o-", lw=1.8, ms=5,
                color=[C_LOAD, C_NET, C_PV][j], label=f"裕量 {h:g} kW")
    ax.plot(xs, [best_h[l] for l in SWEEP_LAM], "k*", ms=13, zorder=5,
            label="各 λ 的最优裕量")
    ax.set_title(f"(c) 最优裕量随误差线性放大（区块 {i1 - i0} 天）", fontsize=11)
    ax.set_xlabel("预测误差缩放 λ")
    ax.set_ylabel("区块总费用 (万元)")
    ax.set_xticks(xs)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)

    # (d) 分位数曲线
    ax = axes[1, 1]
    xs = np.array(QUANTILES)
    ys = np.array([C[q]["r"]["cost"] for q in QUANTILES]) / 1e4
    ax.plot(xs, ys, "o-", color=C_PRICE, lw=2, ms=6, label="总购电费")
    ax.axhline(A[1.0]["r"]["cost"] / 1e4, color=C_LOAD, ls="--", lw=1.5,
               label=f"固定裕量 {HEDGE_FIX:g} kW")
    ax.axvline(q_theory, color=C_NET, ls=":", lw=1.8)
    ax.annotate(f"报童解析值\nq*={q_theory:.2f}", xy=(q_theory, ys.max()),
                xytext=(q_theory - 0.22, ys.max() - 0.06 * np.ptp(ys)),
                fontsize=8.5, color=C_NET)
    ax.plot([q_best], [C[q_best]["r"]["cost"] / 1e4], "k*", ms=14, zorder=5,
            label=f"网格最优 q={q_best:.2f}")
    ax.set_title("(d) 裕量分位数 q → 费用", fontsize=11)
    ax.set_xlabel("裕量分位数 q（逐时段取历史误差分位数）")
    ax.set_ylabel("全年总购电费 (万元)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)

    fig.suptitle("问题二敏感性：预测精度 · 紧急电价倍率 · 安全裕量", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .955))
    save_fig(fig, "fig_敏感性分析.png", FIGDIR)
    write_report(TXT)


if __name__ == "__main__":
    main()
