# -*- coding: utf-8 -*-
"""问题三：「四个时刻已足够」——三张独立插图（各自 PNG + 独立 PDF）

只保留三个论据各一张图，每张单独成文件：
    fig_问题三_影响窗口.png/pdf      ← 结构：插入 tau 只能影响 [tau, t_{k+1})，窗口随 tau 递减
    fig_问题三_弹性衰减.png/pdf      ← 实测：弹性 vs 可影响时段数；晚间光伏恒为 0 是原因
    fig_问题三_信息价值上限.png/pdf  ← 上限：完美信息可再省 9.82%

输出目录：figures/q3/
运行：python fig_q3_epochs_figs.py
"""
import os

import numpy as np

from config import *          # noqa: F401,F403

FIGDIR = os.path.join(DIR_FIG, "q3")
OUT_START = "2025-02-01"
DEC = (0, 6, 12, 18)
ELAS = {6: 37051.0, 12: 16745.0, 18: 146.0}
SPAN = {6: 18, 12: 12, 18: 6}
CEIL_YUAN = 1334596.7
Q3_TOTAL = 13589362.4


# =====================================================================
# 图 1：影响窗口
# =====================================================================
def fig_window():
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.set_xlim(0, 24); ax.set_ylim(0, 1); ax.axis("off")
    ax.plot([0, 24], [0.44, 0.44], color="0.45", lw=1.6)
    for h in DEC + (24,):
        ax.plot([h, h], [0.34, 0.54], color="#3b6ea5", lw=1.7)
        ax.text(h, 0.24, f"{h}:00", ha="center", va="top", fontsize=8.6)
    for i, (tau, c) in enumerate([(6.4, "#2ca02c"), (8.0, "#ff7f0e"),
                                  (10.5, "#d62728")]):
        tn, yy = 12.0, 0.62 + 0.11 * i
        ax.plot([tau, tau], [0.44, yy], color=c, lw=1.8, ls="--")
        ax.annotate("", xy=(tn, yy), xytext=(tau, yy),
                    arrowprops=dict(arrowstyle="<|-|>", color=c, lw=1.7,
                                    mutation_scale=9))
        ax.text((tau + tn) / 2, yy + 0.035, f"{tn - tau:g} h", ha="center",
                fontsize=8.8, color=c)
    ax.text(6.15, 0.56, "$t_k$", fontsize=10)
    ax.text(12.1, 0.56, "$t_{k+1}$", fontsize=10)
    ax.text(0, 0.03, "插入 $\\tau$ 只影响 $[\\tau,\\,t_{k+1})$："
                     "越靠近 $t_{k+1}$ 窗口越短，最长不超过 6 h",
            fontsize=9.2, color="0.20")
    fig.tight_layout()
    save_fig(fig, "fig_问题三_影响窗口.png", FIGDIR)


# =====================================================================
# 图 2：弹性衰减（＋晚间光伏恒为 0）
# =====================================================================
def fig_elasticity():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))

    ax = axes[0]
    hs = [6, 12, 18]
    e = np.array([ELAS[h] for h in hs], float); e /= e[0]
    sp = np.array([SPAN[h] for h in hs], float); sp /= sp[0]
    x = np.arange(3)
    ax.bar(x - .19, e, .36, color="#d62728", label="购电量弹性（kWh）")
    ax.bar(x + .19, sp, .36, color="#1f77b4", label="可影响时段数（h）")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 4)
    for xi, h in enumerate(hs):
        ax.annotate(f"{ELAS[h]:,.0f}", (xi - .19, e[xi]),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=8.8, color="#d62728")
        ax.annotate(f"{SPAN[h]} h", (xi + .19, sp[xi]),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=8.8, color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels([f"{h}:00" for h in hs], fontsize=10)
    ax.set_ylabel("以 6:00 归一化（对数轴）", fontsize=9.5)
    ax.grid(alpha=.3, axis="y", which="both")
    ax.legend(fontsize=9, loc="lower left")
    ax.text(0.97, 0.93, "$253:114:1$  vs  $3:2:1$", transform=ax.transAxes,
            ha="right", fontsize=11, color="0.15")

    ax = axes[1]
    dates, L, G = load_att2()
    m = dates >= np.datetime64(OUT_START)
    prof = G[m].mean(axis=0)
    tt = (np.arange(N_SLOT) + 1) / 6.0
    ax.plot(tt, prof, color="#2ca02c", lw=2.2)
    ax.axvspan(18, 24, color="#d62728", alpha=.13)
    ax.axhline(0, color="0.4", lw=1)
    ax.annotate("光伏恒为 0", (21, prof.max() * 0.62), ha="center",
                fontsize=10, color="#d62728")
    ax.set_xlim(0, 24); ax.set_xticks(DEC + (24,))
    ax.set_xlabel("时刻", fontsize=9.5)
    ax.set_ylabel("光伏出力（kW，全年平均）", fontsize=9.5)
    ax.grid(alpha=.3)
    ax.text(0.02, 0.95, "18:00 的预报只覆盖 18:00$\\sim$24:00", 
            transform=ax.transAxes, fontsize=9, color="0.25", va="top")

    fig.tight_layout()
    save_fig(fig, "fig_问题三_弹性衰减.png", FIGDIR)


# =====================================================================
# 图 3：信息价值上限
# =====================================================================
def fig_ceiling():
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar([0, 1], [Q3_TOTAL / 1e4, CEIL_YUAN / 1e4], .5,
           color=["#1f77b4", "#d62728"], alpha=.9)
    ax.annotate(f"{Q3_TOTAL/1e4:,.2f} 万元", (0, Q3_TOTAL / 1e4),
                textcoords="offset points", xytext=(0, 6), ha="center",
                fontsize=9.5, color="#1f77b4")
    ax.annotate(f"{CEIL_YUAN/1e4:,.2f} 万元\n（9.82%）", (1, CEIL_YUAN / 1e4),
                textcoords="offset points", xytext=(0, 6), ha="center",
                fontsize=9.5, color="#d62728")
    ax.annotate("新增时刻只能从中再切一块", (0.5, Q3_TOTAL / 1e4 * 0.55),
                ha="center", fontsize=9.5, color="0.20")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["全年购电费", "完美信息可再省\n（24 h 全部信息）"],
                       fontsize=9.5)
    ax.set_ylim(0, Q3_TOTAL / 1e4 * 1.22)
    ax.set_ylabel("万元", fontsize=9.5)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    save_fig(fig, "fig_问题三_信息价值上限.png", FIGDIR)


def main():
    arm_report("q3_四时刻已足够.txt")
    rule("问题三：四个时刻已足够（三张独立插图）")
    setup_plot()
    fig_window()
    fig_elasticity()
    fig_ceiling()
    log("    弹性 6:00/12:00/18:00 = 37,051 / 16,745 / 146 kWh（253:114:1）")
    log("    可影响时段数 = 18 / 12 / 6 h（3:2:1）")
    log("    完美信息上限 1,334,596.7 元 = 全年费用的 9.82%")
    write_report("q3_四时刻已足够.txt")


if __name__ == "__main__":
    main()
