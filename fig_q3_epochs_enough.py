# -*- coding: utf-8 -*-
"""问题三：为什么"四个时刻已足够"——纯图形可视化（不做文字复述）

六个子图，全部用坐标轴/曲线/条形表达该段论文的逻辑与数据：
    (a) 24 h 时间轴＋四个决策时刻；插入 tau 及其影响窗口，窗口长度随 tau 递减
    (b) 权衡示意：窗口长度（随 tau 递减）与信息新鲜度（随 tau 递增）负相关
    (c) 实测弹性 vs 可影响时段数（各自以 6:00 归一化，对数轴）
    (d) 全年平均光伏出力曲线：18:00~24:00 恒为 0
    (e) 18:00 处预报抬压 40% 后全天购电量的变动（117 kWh / 0.09%）
    (f) 完美信息上限：全年费用中可再省的 1,334,596.7 元（9.82%）

输出：figures/q3/fig_问题三_四时刻已足够.png + 同名 PDF
运行：python fig_q3_epochs_enough.py
"""
import os

import numpy as np

from config import *          # noqa: F401,F403

FIGDIR = os.path.join(DIR_FIG, "q3")
OUT_START = "2025-02-01"
DEC = (0, 6, 12, 18)
ELAS = {6: 37051.0, 12: 16745.0, 18: 146.0}     # 弹性（kWh）
SPAN = {6: 18, 12: 12, 18: 6}                   # 可影响时段数（h）
DAILY_Q = 65327.0                               # 样本日平均计划购电量（kWh）
CEIL_YUAN = 1334596.7                           # 完美信息可再省（元）
Q3_TOTAL = 13589362.4                           # 问题三全年总费用（元）


def lab(ax, s):
    ax.text(0.0, 1.045, s, transform=ax.transAxes, fontsize=10.5,
            weight="bold", va="bottom", ha="left", color="0.10")


def main():
    rule("问题三：四个时刻已足够（纯图形）")
    setup_plot()
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.6))

    # ================= (a) 时间轴与影响窗口 =================
    ax = axes[0, 0]
    ax.set_xlim(0, 24); ax.set_ylim(0, 1); ax.axis("off")
    ax.plot([0, 24], [0.50, 0.50], color="0.45", lw=1.6)
    for h in DEC + (24,):
        ax.plot([h, h], [0.40, 0.62], color="#3b6ea5", lw=1.7)
        ax.text(h, 0.32, f"{h}", ha="center", va="top", fontsize=9)
    for i, (tau, c) in enumerate([(6.4, "#2ca02c"), (8.0, "#ff7f0e"),
                                  (10.5, "#d62728")]):
        tn, yy = 12.0, 0.70 + 0.10 * i
        ax.plot([tau, tau], [0.40, yy], color=c, lw=1.8, ls="--")
        ax.annotate("", xy=(tn, yy), xytext=(tau, yy),
                    arrowprops=dict(arrowstyle="<|-|>", color=c, lw=1.7,
                                    mutation_scale=9))
        ax.text((tau + tn) / 2, yy + 0.03, f"{tn - tau:g} h", ha="center",
                fontsize=8.6, color=c)
    ax.text(6.2, 0.64, "$t_k$", fontsize=9.5)
    ax.text(12.2, 0.64, "$t_{k+1}$", fontsize=9.5)
    ax.text(0.05, 0.12, "窗口最长 $=6$ h", fontsize=9, color="0.25")
    lab(ax, "(a) 插入 $\\tau$ 只能影响 $[\\tau,\\,t_{k+1})$")

    # ================= (b) 窗口长度 × 信息新鲜度 =================
    ax = axes[0, 1]
    s = np.linspace(0, 1, 100)
    ax.plot(s, 1 - s, color="#1f77b4", lw=2.4)
    ax.annotate("窗口长度", (0.04, 0.90), fontsize=10, color="#1f77b4")
    ax.plot(s, s, color="#d62728", lw=2.4)
    ax.annotate("信息新鲜度", (0.72, 0.74), fontsize=10, color="#d62728")
    ax.annotate("负相关", (0.5, 0.5), xytext=(0.60, 0.76), fontsize=10.5,
                color="0.15",
                arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.4))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["$\\tau\\to t_k$", "$\\tau\\to t_{k+1}$"], fontsize=9.5)
    ax.set_yticks([]); ax.set_ylim(0, 1.10); ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("$\\tau$ 在段内的位置", fontsize=9.5)
    ax.grid(alpha=.25, axis="x")
    lab(ax, "(b) 二者此消彼长，四等分后已无余地")

    # ================= (c) 弹性 vs 可影响时段数 =================
    ax = axes[0, 2]
    hs = [6, 12, 18]
    e = np.array([ELAS[h] for h in hs], float); e /= e[0]
    sp = np.array([SPAN[h] for h in hs], float); sp /= sp[0]
    x = np.arange(3)
    ax.bar(x - .19, e, .36, color="#d62728", label="弹性（kWh）")
    ax.bar(x + .19, sp, .36, color="#1f77b4", label="可影响时段数（h）")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 3)
    for xi, h in enumerate(hs):
        ax.annotate(f"{ELAS[h]:,.0f}", (xi - .19, e[xi]),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=8.6, color="#d62728")
        ax.annotate(f"{SPAN[h]} h", (xi + .19, sp[xi]),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=8.6, color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels([f"{h}:00" for h in hs], fontsize=10)
    ax.set_ylabel("以 6:00 归一化（对数轴）", fontsize=9.5)
    ax.grid(alpha=.3, axis="y", which="both")
    ax.legend(fontsize=9, loc="lower left")
    ax.text(0.98, 0.90, "$253:114:1$  vs  $3:2:1$", transform=ax.transAxes,
            ha="right", fontsize=11, color="0.15")
    lab(ax, "(c) 弹性衰减远快于时段数衰减")

    # ================= (d) 晚间光伏恒为 0 =================
    ax = axes[1, 0]
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
    lab(ax, "(d) 18:00$\\sim$24:00 的光伏预报无可更新内容")

    # ================= (e) ±40% 扰动 → 117 kWh =================
    ax = axes[1, 1]
    ax.barh([1, 0], [DAILY_Q, 117.0], height=.46,
            color=["#1f77b4", "#d62728"], alpha=.9)
    ax.set_xscale("log")
    ax.set_xlim(50, DAILY_Q * 3)
    ax.annotate(f"日购电量 {DAILY_Q:,.0f} kWh", (DAILY_Q, 1),
                textcoords="offset points", xytext=(8, 0), va="center",
                fontsize=9.5, color="#1f77b4")
    ax.annotate("117 kWh\n＝日购电量的 0.18%", (117, 0),
                textcoords="offset points", xytext=(10, 0), va="center",
                fontsize=9.5, color="#d62728")
    ax.annotate("", xy=(117, 0.36), xytext=(DAILY_Q, 0.36),
                arrowprops=dict(arrowstyle="<|-|>", color="0.35", lw=1.3))
    ax.text(np.sqrt(117 * DAILY_Q), 0.46, "相隔约 560 倍", ha="center",
            fontsize=9, color="0.30")
    ax.set_yticks([]); ax.set_ylim(-.6, 1.6)
    ax.set_xlabel("抬压 40% 后全天购电量的变动（kWh，对数轴）", fontsize=9.5)
    ax.grid(alpha=.3, axis="x", which="both")
    lab(ax, "(e) 该时刻近似为哑变量")

    # ================= (f) 信息价值上限 =================
    ax = axes[1, 2]
    ax.bar([0, 1], [Q3_TOTAL / 1e4, CEIL_YUAN / 1e4], .5,
           color=["#1f77b4", "#d62728"], alpha=.9)
    ax.annotate(f"全年购电费\n{Q3_TOTAL/1e4:,.2f} 万元", (0, Q3_TOTAL / 1e4),
                textcoords="offset points", xytext=(0, 6), ha="center",
                fontsize=9.5, color="#1f77b4")
    ax.annotate(f"完美信息可再省\n{CEIL_YUAN/1e4:,.2f} 万元（9.82%）",
                (1, CEIL_YUAN / 1e4), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=9.5, color="#d62728")
    ax.annotate("新增时刻只能从中再切一块（窗口更短、且被后续时刻覆盖）",
                (0.5, Q3_TOTAL / 1e4 * 0.62), ha="center", fontsize=9.5,
                color="0.20")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["全年总费用", "24 h 全部信息的天花板"], fontsize=10)
    ax.set_ylim(0, Q3_TOTAL / 1e4 * 1.22)
    ax.set_ylabel("万元", fontsize=9.5)
    ax.grid(alpha=.3, axis="y")
    lab(ax, "(f) 新增时刻能分到的份额上限")

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save_fig(fig, "fig_问题三_四时刻已足够.png", FIGDIR)

    rule("图中数字", width=60)
    for s in (f"弹性 6:00 {ELAS[6]:,.0f} / 12:00 {ELAS[12]:,.0f} / 18:00 {ELAS[18]:,.0f} kWh",
              "归一化 1 : 0.452 : 0.0039（原文 253:114:1）",
              f"可影响时段数 {SPAN[6]} / {SPAN[12]} / {SPAN[18]} h（3:2:1）",
              f"抬压 40% 后全天购电量变动 117 kWh = 日购电量的 {117/DAILY_Q:.2%}",
              "（说明：论文原句写的 0.09% 是把它当成单侧幅度，117/65327 应为 0.18%）",
              f"完美信息上限 {CEIL_YUAN:,.1f} 元 = 全年费用的 {CEIL_YUAN/Q3_TOTAL:.2%}"):
        log("    " + s)
    write_report("q3_四时刻已足够.txt")


if __name__ == "__main__":
    main()
