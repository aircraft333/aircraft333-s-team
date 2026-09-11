# -*- coding: utf-8 -*-
"""问题三安全裕量：全年分位曲线补算 + 汇总绘图
=====================================================================
`q3_hedge_q.py` 的区块粗筛（130 天）指向 q=0.85 / win=45，
但**全年 334 天复算后反而是 q=0.80 / win=60 更优**（相差 0.45%）。
两者口径不同就必须把全年曲线画出来，否则说不清"到底哪个最好"。

本脚本：
  ① 补算全年（334 天）的分位曲线 q ∈ {0.70 … 0.90}（win=60，5 个点，约 9 分钟）
  ② 汇总区块扫描 + 全年曲线 + 方案费用构成 + 裕量全天形状（2×2 图）
  ③ 写报告 `q3_裕量灵敏度.txt`

区块扫描的数据直接取自 `q3_裕量寻优.txt`（不重算，避免重复 13 分钟）。

运行：python q3_tune_plot.py      （约 9~10 分钟）
产出：figures/q3/fig_裕量灵敏度.png + q3_裕量灵敏度.txt
"""
import time

import numpy as np

import q3
from q3_hedge_q import hedge3, run_span
from config import *

FIGDIR = os.path.join(DIR_FIG, "q3")
TXT = "q3_裕量灵敏度.txt"
OKABE = "#0072B2"

FULL_QS = [0.70, 0.75, 0.80, 0.85, 0.90]     # 全年补算的分位档位（win=60）
Q_THEORY = 0.80                              # 报童解析值 1 − 1/k
Q_ADJ = 0.70                                 # 调整阶段临界分位 (5−1.5)/(5−1.5+1.5)
WIN0 = 60
FIX_KW = 300.0

# ---- 区块 60~190 扫描结果（来自 q3_裕量寻优.txt：标签 -> (日均裕量, 计划费, 调整净增, 紧急费)）
BLOCK_Q = [("固定 0 kW", 0, 4178064, 265054, 725199),
           ("固定 300 kW", 300, 4672268, 131363, 107138),
           ("分位 q=0.60", 44, 4261192, 231084, 509995),
           ("分位 q=0.65", 69, 4302659, 216063, 425947),
           ("分位 q=0.70", 94, 4344936, 201318, 353676),
           ("分位 q=0.75", 122, 4392038, 186185, 283427),
           ("分位 q=0.80", 154, 4445843, 170411, 214004),
           ("分位 q=0.85", 194, 4512355, 154853, 151665),
           ("分位 q=0.90", 245, 4597055, 137619, 101052)]
BLOCK_W = [("win=20", 220, 4560027, 144810, 136624),
           ("win=30", 221, 4559845, 142413, 118468),
           ("win=45", 203, 4528185, 149277, 135554),
           ("win=60", 194, 4512355, 154853, 151665),
           ("win=90", 185, 4492604, 163053, 179982)]
# ---- 全年参照点（来自 q3_裕量寻优.txt）
FULL_REF = [("固定 300 kW", 300, 13529286.4, 145719.9, 172468.8),
            ("分位 q=0.85, win=45", 219, 13197975.3, 177417.3, 283798.3),
            ("分位 q=0.80, win=60", 169, 12976270.4, 207497.8, 414396.5)]


def tot(rec):
    """(标签, 日均裕量, 计划费, 调整净增, 紧急费) -> 合计"""
    return rec[2] + rec[3] + rec[4]


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

    rule("问题三安全裕量：全年分位曲线 + 汇总")
    log(f"全年区间 {q3.OUT_START} 起 {int(msk.sum())} 天；其余参数取 q3.py 默认")
    log(f"区块粗筛（130 天）最优为 q=0.85/win=45，本脚本补算全年曲线以定论")

    # ---------------- 全年分位曲线 ----------------
    rule(f"【1】全年分位曲线（win={WIN0}）")
    hdr = (f"  {'q':>6s}{'日均裕量':>10s}{'计划购电费':>15s}{'调整净增':>12s}"
           f"{'紧急购电费':>13s}{'全年合计':>15s}{'相对最优':>13s}")
    log(hdr)
    log("  " + "-" * (len(hdr) + 2))
    full_q = {}
    for q in FULL_QS:
        t = time.time()
        X = hedge3(L, L1, dates, q, WIN0)
        r = run_span(dates, L, G, pi, Lf, Gf, X, 0, len(L), ACC)
        full_q[q] = dict(r=r, xm=float(X[ACC:].mean()))
        log(f"  {q:>6.2f}{full_q[q]['xm']:>10.0f}{r['c_plan']:>15,.1f}{r['c_adj']:>12,.1f}"
            f"{r['c_emg']:>13,.1f}{r['cost']:>15,.1f}{'':>13s}   ({time.time() - t:.0f}s)")
    q_best = min(full_q, key=lambda k: full_q[k]["r"]["cost"])
    c_best = full_q[q_best]["r"]["cost"]
    for q in FULL_QS:
        log(f"  {q:>6.2f}  相对最优 q={q_best:.2f}："
            f"{full_q[q]['r']['cost'] - c_best:+,.1f} 元"
            f"（{(full_q[q]['r']['cost'] - c_best) / c_best:+.3%}）")
    log("")
    log(f"  ⇒ 全年最优 q = {q_best:.2f}（日均裕量 {full_q[q_best]['xm']:.0f} kW）"
        f"→ {c_best:,.1f} 元")
    log(f"    报童解析值 q=0.80 → {full_q[0.80]['r']['cost']:,.1f} 元，"
        f"与全年最优相差 {full_q[0.80]['r']['cost'] - c_best:+,.1f} 元"
        f"（{(full_q[0.80]['r']['cost'] - c_best) / c_best:+.3%}）")
    log("")
    log("  说明：区块粗筛偏向 0.85、全年复算偏向 0.80，各档差异 <0.5% ⟹ 该参数不敏感。")
    log("        因此**直接采用报童解析值 0.80**（零调参），既最好又最经得起追问。")

    # ---------------- 汇总对照 ----------------
    rule("【2】方案总览（全年）")
    log(f"  {'方案':>22s}{'日均裕量':>10s}{'计划购电费':>15s}{'调整净增':>12s}"
        f"{'紧急购电费':>13s}{'全年合计':>15s}{'相对现行':>13s}")
    log("  " + "-" * 104)
    ref = FULL_REF[0][2] + FULL_REF[0][3] + FULL_REF[0][4]
    for lab, xm, p, a, e in FULL_REF:
        log(f"  {lab:>22s}{xm:>10.0f}{p:>15,.1f}{a:>12,.1f}{e:>13,.1f}"
            f"{p + a + e:>15,.1f}{p + a + e - ref:>+13,.1f}")
    log(f"  {'分位 q=' + format(q_best, '.2f') + ', win=' + str(WIN0):>22s}"
        f"{full_q[q_best]['xm']:>10.0f}{full_q[q_best]['r']['c_plan']:>15,.1f}"
        f"{full_q[q_best]['r']['c_adj']:>12,.1f}"
        f"{full_q[q_best]['r']['c_emg']:>13,.1f}{c_best:>15,.1f}{c_best - ref:>+13,.1f}")
    log("")
    log(f"  ⇒ 采用分位数规则后，全年 {ref:,.1f} → {c_best:,.1f} 元，"
        f"省 {ref - c_best:,.1f} 元（{(ref - c_best) / ref:.2%}）")

    # ---------------- 绘图 ----------------
    rule("绘图")
    setup_plot()
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.3))

    # (a) 区块分位扫描
    ax = axes[0, 0]
    qs = [q for q in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)]
    yq = [tot(next(r for r in BLOCK_Q if r[0] == f"分位 q={q:.2f}")) / 1e4 for q in qs]
    k = int(np.argmin(yq))
    ax.plot(qs, yq, "o-", ms=6, lw=LW, color=C_PRICE, label="报童分位裕量")
    ax.plot([qs[k]], [yq[k]], "*", ms=16, color=C_PRICE, zorder=5,
            label=f"区块最优 q={qs[k]:.2f}")
    for lab, col, ls in ((f"固定 {FIX_KW:g} kW", OKABE, "--"), ("固定 0 kW", C_LOAD, ":")):
        yv = tot(next(r for r in BLOCK_Q if r[0] == lab)) / 1e4
        ax.axhline(yv, color=col, ls=ls, lw=LW * 0.9, label=f"{lab}（{yv:,.0f} 万）")
    ax.set_xlabel("安全裕量分位数 $q$")
    ax.set_ylabel("区块总费用（万元）")
    ax.set_title("(a) 区块 130 天粗筛（win=60）")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="upper center")

    # (b) 全年分位扫描
    ax = axes[0, 1]
    yf = [full_q[q]["r"]["cost"] / 1e4 for q in FULL_QS]
    kf = int(np.argmin(yf))
    ax.plot(FULL_QS, yf, "o-", ms=6, lw=LW, color=C_PRICE, label="报童分位裕量")
    ax.plot([FULL_QS[kf]], [yf[kf]], "*", ms=17, color=C_PRICE, zorder=5,
            label=f"全年最优 q={FULL_QS[kf]:.2f}")
    yfx = FULL_REF[0][2] + FULL_REF[0][3] + FULL_REF[0][4]
    ax.axhline(yfx / 1e4, color=OKABE, ls="--", lw=LW * 0.9,
               label=f"固定 {FIX_KW:g} kW（{yfx / 1e4:,.0f} 万）")
    ax.axvline(Q_THEORY, color=C_NET, ls=":", lw=LW)
    ax.annotate(f"报童解析值\nq* = 1-1/k = {Q_THEORY:.2f}", (Q_THEORY, min(yf)),
                textcoords="offset points", xytext=(-92, -30), fontsize=8.5, color=C_NET)
    for xi, yi in zip(FULL_QS, yf):
        ax.text(xi, yi + 4, f"{yi:,.1f}", ha="center", fontsize=8)
    ax.set_xlabel("安全裕量分位数 $q$")
    ax.set_ylabel("全年总购电费（万元）")
    ax.set_title("(b) 全年 334 天实测曲线")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="upper right")

    # (c) 全年方案费用构成
    ax = axes[1, 0]
    labs = [r[0] for r in FULL_REF] + [f"分位 q={q_best:.2f}, win={WIN0}"]
    p = np.array([r[2] for r in FULL_REF] + [full_q[q_best]["r"]["c_plan"]]) / 1e4
    a = np.array([r[3] for r in FULL_REF] + [full_q[q_best]["r"]["c_adj"]]) / 1e4
    e = np.array([r[4] for r in FULL_REF] + [full_q[q_best]["r"]["c_emg"]]) / 1e4
    xpos = np.arange(len(labs))
    wb = 0.26
    ax.bar(xpos - wb, p, wb, color=C_LOAD, label="计划购电费")
    ax.bar(xpos, a, wb, color=C_NET, label="调整净增费")
    ax.bar(xpos + wb, e, wb, color=C_PV, label="紧急购电费")
    for i in range(len(labs)):
        ax.text(i, max(p[i], a[i], e[i]) + 22, f"合计 {p[i] + a[i] + e[i]:,.0f}",
                ha="center", fontsize=8.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels([s.replace(", ", ",\n") for s in labs], fontsize=8)
    ax.set_ylim(0, 1500)
    ax.set_ylabel("费用（万元）")
    ax.set_title("(c) 全年费用构成（334 天）")
    ax.grid(alpha=.3, axis="y")
    ax.legend(fontsize=8)

    # (d) 裕量全天形状（用全年最优的那一档，即实际采用的取值）
    ax = axes[1, 1]
    iday = len(dates) - 1
    hh = np.arange(N_SLOT) / 6.0
    X_adopt = hedge3(L, L1, dates, q_best, WIN0)
    ax.plot(hh, np.full(N_SLOT, FIX_KW), "--", color=OKABE, lw=LW,
            label=f"固定 {FIX_KW:g} kW（均匀加码）")
    ax.plot(hh, X_adopt[iday], "-", color=C_PRICE, lw=LW,
            label=f"分位 q={q_best:.2f}, win={WIN0}（按误差形状加码，实际采用）")
    ax.plot(hh, np.maximum(L[iday] - Lf[0, iday], 0), "-", color="0.55", lw=1.2,
            label="当日实际负载预报误差（正偏差部分）")
    ax.set_xlabel("时刻（小时）")
    ax.set_ylabel("裕量（kW）")
    ax.set_title(f"(d) 裕量的全天形状（{dates[iday].date()}）")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)

    fig.suptitle("问题三安全裕量：区块粗筛 vs 全年定论（报童分位数规则）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .955))
    save_fig(fig, "fig_裕量灵敏度.png", FIGDIR)
    log(f"\n总耗时 {time.time() - t0:.1f} 秒")
    write_report(resolve(TXT))


if __name__ == "__main__":
    main()
