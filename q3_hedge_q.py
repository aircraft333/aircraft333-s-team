# -*- coding: utf-8 -*-
"""问题三：安全裕量升级为报童分位数规则（区块寻优 + 全年验证 + 可视化）
=====================================================================
背景：问题二已证明「报童分位裕量」优于手调的固定 300 kW；
      而问题三的 `HEDGE = 300.0` 至今仍是手调的固定值。

问题三与问题二的两点差别：
  ① 裕量加在**负载预报**上（光伏另走附件3 预报渠道），所以误差应取 L − F_L；
  ② 边际结构不同 —— 计划阶段是「π 对 5π」，报童临界分位 1−1/k = 0.80；
     但调整阶段多买要多付 1.5π、少买可退 0.5π，边际变成「1.5π 对 5π」，
     临界分位 (5−1.5)/((5−1.5)+1.5) = 0.70。
     两阶段共用同一个裕量，所以实测最优点应落在 0.70~0.80 之间。

流程：
  【1】区块 60~190（130 天）：固定 0 / 固定 300 / 分位 q ∈ {0.60 … 0.90}（win=60）
  【2】同区块：扫回看窗口 win ∈ {20, 30, 45, 90}（固定上一步最优 q）
  【3】全年 334 天验证：固定 300 vs 分位最优 vs 报童解析值
  【4】绘图：figures/q3/fig_裕量寻优.png

运行：python q3_hedge_q.py      （约 12~15 分钟）
产出：q3_裕量寻优.txt + figures/q3/fig_裕量寻优.png
"""
import time

import numpy as np

import q2
import q3
from config import *

FIGDIR = os.path.join(DIR_FIG, "q3")
TXT = "q3_裕量寻优.txt"
BLOCK = (60, 190)                 # 与 q3 参数寻优同款区块（130 天）
QS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
WINS_B = [20, 30, 45, 90]
FIX = [0.0, 300.0]
WIN0 = 60
Q_THEORY = 0.80                   # 计划阶段报童解析值 1 − 1/k
Q_THEORY_ADJ = 0.70               # 调整阶段临界分位 (5−1.5)/(5−1.5+1.5)
OKABE = "#0072B2"


def hedge3(L, L1, dates, q, win=WIN0):
    """问题三的分位数裕量：对「负载预报误差 L − F_L」逐时段取滚动分位数 → (D, 144)"""
    F0 = q3.build_load_forecast(L, L1, dates, q3.DECIDE_H, q3.LOAD_LAG, q3.ADAPT)[0]
    return q2.build_hedge(L - F0, "quantile", q, win)


def run_span(dates, L, G, pi, Lf, Gf, X, i0, i1, acc_from):
    """在 [i0, i1) 上滚动调度，从 acc_from 起累计；X 可为标量或逐日矩阵"""
    tot = dict(c_plan=0.0, c_dev=0.0, c_emg=0.0,
               q_plan=0.0, q_adj=0.0, q_emg=0.0, n=0)
    E = SOC0
    for i in range(i0, i1):
        h = X if np.isscalar(X) else X[i]
        r = q3.run_day(pi, Lf[:, i], Gf[:, i], L[i], G[i], E, q3.DECIDE_H, h)
        E = r["E24"]
        if i < acc_from:
            continue
        tot["c_plan"] += r["cost_plan"]
        tot["c_dev"] += r["cost_dev"]
        tot["c_emg"] += r["cost_emg"]
        tot["q_plan"] += float(r["b_plan"].sum() * DT_H)
        tot["q_adj"] += float(r["b_adj"].sum() * DT_H)
        tot["q_emg"] += float(r["e"].sum() * DT_H)
        tot["n"] += 1
    # 注意：q3.cost_dev 已包含「计划部分」π·min(b^p,b^a)，
    # 与 q3.summarize 一致，合计 = c_dev + c_emg（不能再加 c_plan，否则重复计入）
    tot["c_adj"] = tot["c_dev"] - tot["c_plan"]      # 调整带来的净增（可为负）
    tot["cost"] = tot["c_dev"] + tot["c_emg"]
    return tot


def main():
    t0 = time.time()
    A1 = att1_arrays(load_att1())
    pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
    dates, L, G = load_att2()
    df3 = load_att3()
    Lf = q3.build_load_forecast(L, L1, dates, q3.DECIDE_H, q3.LOAD_LAG, q3.ADAPT)
    Gf = q3.build_pv_forecast(df3, dates, G, q3.DECIDE_H, q3.PV_MIX)
    msk = dates >= pd.Timestamp(q3.OUT_START)
    ACC = int(np.argmax(msk))
    i0, i1 = BLOCK
    R = {}

    rule("问题三安全裕量升级：固定值 vs 报童分位数")
    log(f"其余参数取 q3.py 默认：mix={q3.PV_MIX:g}, adapt={q3.ADAPT:g}, "
        f"决策时刻 {q3.DECIDE_H}, 现行固定裕量 {q3.HEDGE:g} kW")
    log(f"区块 {i0}~{i1}（{i1 - i0} 天）；全年输出区间 {q3.OUT_START} 起 {int(msk.sum())} 天")

    errs = L - q3.build_load_forecast(L, L1, dates, q3.DECIDE_H,
                                      q3.LOAD_LAG, 0.0)[0]
    log(f"负载预报误差：均值 {errs.mean():.1f} kW，标准差 {errs.std():.1f} kW，"
        f"q50 {np.percentile(errs, 50):.0f}，q70 {np.percentile(errs, 70):.0f}，"
        f"q80 {np.percentile(errs, 80):.0f}，q90 {np.percentile(errs, 90):.0f}")

    def show(tag, r, xm, ref):
        log(f"  {tag:>22s}{xm:>10.0f}{r['c_plan']:>13,.0f}{r['c_adj']:>12,.0f}"
            f"{r['c_emg']:>12,.0f}{r['cost']:>13,.0f}{r['cost'] - ref:>+12,.0f}")

    # ---------------- 【1】区块：扫分位 q ----------------
    rule(f"【1】区块 {i0}~{i1}：固定值 vs 分位数 q（win={WIN0}）")
    hdr = (f"  {'方案':>22s}{'日均裕量':>10s}{'计划购电费':>13s}{'调整净增':>12s}"
           f"{'紧急购电费':>12s}{'区块合计':>13s}{'相对基线':>12s}")
    log(hdr)
    log("  " + "-" * (len(hdr) + 2))
    base = None
    for name, X in ([("固定 0 kW", 0.0), (f"固定 {FIX[1]:g} kW", FIX[1])]
                    + [(f"分位 q={q:.2f}", hedge3(L, L1, dates, q)) for q in QS]):
        t = time.time()
        r = run_span(dates, L, G, pi, Lf, Gf, X, i0, i1, i0)
        xm = float(X) if np.isscalar(X) else float(X[i0:i1].mean())
        base = r["cost"] if base is None else base
        R[name] = dict(r=r, xm=xm)
        show(name, r, xm, base)
        log(f"{'':>44s}({time.time() - t:.0f}s)")
    q_names = [f"分位 q={q:.2f}" for q in QS]
    q_best = min(q_names, key=lambda k: R[k]["r"]["cost"])
    qb = float(q_best.split("=")[1])
    log("")
    log(f"  ⇒ 区块最优：{q_best} → {R[q_best]['r']['cost']:,.0f} 元"
        f"（日均裕量 {R[q_best]['xm']:.0f} kW）")
    for nm in (f"固定 {FIX[1]:g} kW", "固定 0 kW"):
        log(f"     相对 {nm}：{R[q_best]['r']['cost'] - R[nm]['r']['cost']:+,.0f} 元"
            f"（{(R[q_best]['r']['cost'] - R[nm]['r']['cost']) / R[nm]['r']['cost']:+.2%}）")

    # ---------------- 【2】区块：扫回看窗口 ----------------
    rule(f"【2】区块 {i0}~{i1}：扫回看窗口 win（q={qb:.2f}）")
    log(hdr)
    log("  " + "-" * (len(hdr) + 2))
    RB = {WIN0: dict(r=R[q_best]["r"], xm=R[q_best]["xm"])}
    for w in WINS_B:
        t = time.time()
        X = hedge3(L, L1, dates, qb, w)
        r = run_span(dates, L, G, pi, Lf, Gf, X, i0, i1, i0)
        RB[w] = dict(r=r, xm=float(X[i0:i1].mean()))
        show(f"分位 win={w} 天", r, RB[w]["xm"], R["固定 0 kW"]["r"]["cost"])
        log(f"{'':>44s}({time.time() - t:.0f}s)")
    w_best = min(RB, key=lambda k: RB[k]["r"]["cost"])
    log("")
    log(f"  ⇒ 区块最优窗口 win = {w_best} 天 → {RB[w_best]['r']['cost']:,.0f} 元"
        f"（相对 win={WIN0}：{RB[w_best]['r']['cost'] - RB[WIN0]['r']['cost']:+,.0f} 元）")

    # ---------------- 【3】全年验证 ----------------
    rule(f"【3】全年验证（{q3.OUT_START} 起 {int(msk.sum())} 天）")
    log(f"  {'方案':>22s}{'日均裕量':>10s}{'计划购电费':>14s}{'调整净增':>12s}"
        f"{'紧急购电费':>12s}{'全年合计':>14s}")
    log("  " + "-" * 88)
    cands = ((f"固定 {FIX[1]:g} kW", FIX[1]),
             (f"分位 q={qb:.2f}, win={w_best}", hedge3(L, L1, dates, qb, w_best)),
             (f"分位 q={Q_THEORY:.2f}, win={WIN0}", hedge3(L, L1, dates, Q_THEORY)))
    full = {}
    for name, X in cands:
        t = time.time()
        r = run_span(dates, L, G, pi, Lf, Gf, X, 0, len(L), ACC)
        xm = float(X) if np.isscalar(X) else float(X[ACC:].mean())
        full[name] = dict(r=r, xm=xm)
        log(f"  {name:>22s}{xm:>10.0f}{r['c_plan']:>14,.1f}{r['c_adj']:>12,.1f}"
            f"{r['c_emg']:>12,.1f}{r['cost']:>14,.1f}   ({time.time() - t:.0f}s)")
    fb = min(full, key=lambda k: full[k]["r"]["cost"])
    for nm in full:
        log(f"  {nm:>22s}  相对最优：{full[nm]['r']['cost'] - full[fb]['r']['cost']:+,.1f} 元"
            f"（{(full[nm]['r']['cost'] - full[fb]['r']['cost']) / full[fb]['r']['cost']:+.3%}）")
    log("")
    log(f"  ⇒ 全年结论：采用「{fb}」→ {full[fb]['r']['cost']:,.1f} 元"
        f"（现行固定 {FIX[1]:g} kW 为 {full[f'固定 {FIX[1]:g} kW']['r']['cost']:,.1f} 元）")
    log("")
    log("  理论参照：计划阶段报童临界分位 = 1-1/k = 0.80；")
    log("            调整阶段因多付 1.5π / 退 0.5π，临界分位 = (5-1.5)/(5-1.5+1.5) = 0.70；")
    log("            两阶段共用同一裕量，因此实测最优点应落在 0.70~0.80 之间。")

    # ---------------- 【4】绘图 ----------------
    rule("绘图")
    setup_plot()
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2))

    # (a) q 扫描
    ax = axes[0, 0]
    ys = [R[f"分位 q={q:.2f}"]["r"]["cost"] / 1e4 for q in QS]
    k = int(np.argmin(ys))
    ax.plot(QS, ys, "o-", ms=6, lw=LW, color=C_PRICE, label="报童分位裕量")
    ax.plot([QS[k]], [ys[k]], "*", ms=16, color=C_PRICE, zorder=5)
    for lab, col, ls in ((f"固定 {FIX[1]:g} kW", OKABE, "--"), ("固定 0 kW", C_LOAD, ":")):
        yv = R[lab]["r"]["cost"] / 1e4
        ax.axhline(yv, color=col, ls=ls, lw=LW * 0.9, label=f"{lab}（{yv:,.0f} 万）")
    for xv, txt, dx in ((Q_THEORY_ADJ, "调整阶段\n临界分位 0.70", -66),
                        (Q_THEORY, "计划阶段\n临界分位 0.80", 8)):
        ax.axvline(xv, color=C_NET, ls=":", lw=LW)
        ax.annotate(txt, (xv, min(ys)), textcoords="offset points",
                    xytext=(dx, -34), fontsize=8, color=C_NET)
    ax.set_xlabel("安全裕量分位数 $q$")
    ax.set_ylabel("区块总费用（万元）")
    ax.set_title(f"(a) 分位水平 $q$（区块 {i1 - i0} 天，win={WIN0}）")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="upper center")

    # (b) win 扫描
    ax = axes[0, 1]
    ws = sorted(RB)
    yw = [RB[w]["r"]["cost"] / 1e4 for w in ws]
    kw = int(np.argmin(yw))
    ax.plot(ws, yw, "o-", ms=6, lw=LW, color=C_PRICE)
    ax.plot([ws[kw]], [yw[kw]], "*", ms=16, color=C_PRICE, zorder=5)
    ax.annotate(f"最优 {ws[kw]} 天\n{yw[kw]:,.1f} 万元", (ws[kw], yw[kw]),
                textcoords="offset points", xytext=(8, 10), fontsize=9, color=C_PRICE)
    ax.axhline(R[f"固定 {FIX[1]:g} kW"]["r"]["cost"] / 1e4, color=OKABE, ls="--",
               lw=LW * 0.9, label=f"固定 {FIX[1]:g} kW")
    ax.set_xticks(ws)
    ax.set_xlabel("回看窗口长度 win（天）")
    ax.set_ylabel("区块总费用（万元）")
    ax.set_title(f"(b) 回看窗口（$q$={qb:.2f}）")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)

    # (c) 裕量形状
    ax = axes[1, 0]
    iday = len(dates) - 1
    hh = np.arange(N_SLOT) / 6.0
    X_best = hedge3(L, L1, dates, qb, w_best)
    ax.plot(hh, np.full(N_SLOT, FIX[1]), "--", color=OKABE, lw=LW,
            label=f"固定 {FIX[1]:g} kW（均匀加码）")
    ax.plot(hh, X_best[iday], "-", color=C_PRICE, lw=LW,
            label=f"分位 q={qb:.2f}, win={w_best}（按误差形状加码）")
    ax.plot(hh, np.maximum(L[iday] - Lf[0, iday], 0), "-", color="0.55", lw=1.2,
            label="当日实际负载预报误差（正偏差部分）")
    ax.set_xlabel("时刻（小时）")
    ax.set_ylabel("裕量（kW）")
    ax.set_title(f"(c) 裕量的全天形状（{dates[iday].date()}）")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)

    # (d) 全年三方案成本构成
    ax = axes[1, 1]
    names = list(full)
    plan = np.array([full[n]["r"]["c_plan"] for n in names]) / 1e4
    dev = np.array([full[n]["r"]["c_adj"] for n in names]) / 1e4
    emg = np.array([full[n]["r"]["c_emg"] for n in names]) / 1e4
    xpos = np.arange(len(names))
    wbar = 0.26
    ax.bar(xpos - wbar, plan, wbar, color=C_LOAD, label="计划购电费")
    ax.bar(xpos, dev, wbar, color=C_NET, label="调整净增费")
    ax.bar(xpos + wbar, emg, wbar, color=C_PV, label="紧急购电费")
    for i, n in enumerate(names):
        tot = full[n]["r"]["cost"] / 1e4
        ax.text(i, max(plan[i], dev[i], emg[i]) + 24, f"合计 {tot:,.0f}",
                ha="center", fontsize=8.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels([n.replace(", ", ",\n") for n in names], fontsize=8)
    ax.set_ylim(0, 1500)
    ax.set_ylabel("费用（万元）")
    ax.set_title("(d) 全年费用构成（334 天）")
    ax.grid(alpha=.3, axis="y")
    ax.legend(fontsize=8)

    fig.suptitle("问题三安全裕量升级：固定值 vs 报童分位数规则", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .955))
    save_fig(fig, "fig_裕量寻优.png", FIGDIR)
    log(f"\n总耗时 {time.time() - t0:.1f} 秒")
    write_report(resolve(TXT))


if __name__ == "__main__":
    main()
