# -*- coding: utf-8 -*-
"""问题四 思维导图（模仿团队手绘的「问题三思维导图」版式）

版式要素与问题三那张一致：
    虚线标题框 → 向下箭头 → 分支（两个并列子框）→ 汇合 → 右侧附加框
    → 三级竖直链 → 左侧方括号括住后三段 + 红色方法名 → 右侧「第四问建模结果」

内容依据（全部来自交付文件）：
    · 唯一变量是电价矩阵：附件 1 固定分时曲线 → 附件 4 逐日波动电价
    · 其余口径（负载 wday:7、光伏 ma:3 / 附件3 混合 m=0.4、报童分位裕量、逐槽执行）完全不变
    · 结果：对应问题二 14,634,080.5 元（+4.91%）；对应问题三 14,210,954.8 元（+4.57%）
    · 涨价归因：逐时段均价项 2.8% / 逐日漂移项 97.2%（受控分解）

输出：figures/q4/fig_第四问思维导图.png + 同名 PDF
运行：python fig_q4_mindmap.py
"""
import os

from matplotlib.patches import FancyArrowPatch, Rectangle

from config import *          # noqa: F401,F403

FIGDIR = os.path.join(DIR_FIG, "q4")
BLUE = "#2b6cb0"
EDGE = "#3b6ea5"
RED = "#c0392b"
DASH = (0, (5, 3))


def box(ax, x, y, w, h, text, fs=10.5, ec=EDGE, fc="white", lw=1.4,
        ls="-", weight="normal", tc="0.10"):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h,
                           facecolor=fc, edgecolor=ec, lw=lw, ls=ls,
                           zorder=2, joinstyle="round"))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3,
            color=tc, weight=weight, linespacing=1.45)


def arrow(ax, p, q, color=BLUE, lw=1.7, style="-|>", ms=11):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, color=color,
                                 lw=lw, mutation_scale=ms, zorder=1,
                                 shrinkA=0, shrinkB=0))


def main():
    rule("问题四 思维导图")
    setup_plot()
    fig, ax = plt.subplots(figsize=(7.8, 10.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---------------- 标题 ----------------
    box(ax, 0.5, 0.958, 0.94, 0.062,
        "电价实时波动下购电费用最低的两阶段／多阶段滚动线性优化模型",
        fs=11.5, ls=DASH, lw=1.6, weight="bold")

    # ---------------- 电价场景 ----------------
    arrow(ax, (0.5, 0.927), (0.5, 0.908))
    box(ax, 0.5, 0.881, 0.40, 0.052, "电价场景（唯一变量）", fs=10.5,
        weight="bold")

    ybr, ysub = 0.757, 0.828
    arrow(ax, (0.5, 0.855), (0.5, ysub))
    ax.plot([0.24, 0.76], [ysub, ysub], color=BLUE, lw=1.7)
    arrow(ax, (0.24, ysub), (0.24, ybr + 0.035))
    arrow(ax, (0.76, ysub), (0.76, ybr + 0.035))
    box(ax, 0.24, ybr, 0.44, 0.070,
        "附件 4：逐日波动电价\n（每天 144 个时段各不相同）", fs=9.5)
    box(ax, 0.76, ybr, 0.44, 0.070,
        "对照：附件 1 固定分时电价曲线\n（问题二／三 所用口径）", fs=9.5)

    # ---------------- 受控实验 ＋ 右侧报童裕量 ----------------
    yc = 0.642
    arrow(ax, (0.24, ybr - 0.035), (0.24, 0.700))
    arrow(ax, (0.76, ybr - 0.035), (0.76, 0.700))
    ax.plot([0.24, 0.76], [0.700, 0.700], color=BLUE, lw=1.7)
    arrow(ax, (0.40, 0.700), (0.40, yc + 0.047))
    box(ax, 0.36, yc, 0.58, 0.094,
        "受控实验：只替换电价矩阵\n"
        "负载 wday:7、光伏 ma:3／附件 3 混合 $m$=0.4\n"
        "报童分位裕量、逐槽执行规则全部不变", fs=9.0, fc="#f2f7fc")
    arrow(ax, (0.65, yc), (0.685, yc))
    box(ax, 0.83, yc, 0.29, 0.058, "报童型安全裕量 $X$", fs=9.5)

    # ---------------- 三级竖直链 ----------------
    y1, y2, y3 = 0.512, 0.388, 0.262
    arrow(ax, (0.40, yc - 0.047), (0.40, y1 + 0.051))
    box(ax, 0.5, y1, 0.90, 0.102,
        "情形一（对应问题二）：0:00 计划 LP ＋ 逐槽被动执行\n"
        "缺口按 $5\\pi_t$ 紧急购电　→　全年 14,634,080.5 元（$+4.91\\%$）",
        fs=9.8)

    arrow(ax, (0.5, y1 - 0.051), (0.5, y2 + 0.056))
    box(ax, 0.5, y2, 0.90, 0.112,
        "情形二（对应问题三）：0:00 计划 ＋ 6:00／12:00／18:00 滚动调整\n"
        "偏差按 $0.5\\pi_t$／$1.5\\pi_t$ 双向结算　→　全年 14,210,954.8 元（$+4.57\\%$）",
        fs=9.8)

    arrow(ax, (0.5, y2 - 0.056), (0.5, y3 + 0.047))
    box(ax, 0.5, y3, 0.90, 0.094,
        "目标：全年购电费用最少（$\\min Z$）\n"
        "涨价归因（受控分解）：逐时段均价项 2.8\\%　｜　逐日漂移项 97.2\\%",
        fs=9.8)

    # ---------------- 左侧方括号 + 底部红色方法名 ----------------
    bx, top, bot = 0.028, y1 + 0.054, y3 - 0.050
    ax.plot([bx + 0.026, bx, bx, bx + 0.026], [top, top, bot, bot],
            color="0.20", lw=2.4, solid_capstyle="round", zorder=1)
    ax.text(0.47, 0.168, "波动电价下的受控对比实验",
            ha="center", va="center", fontsize=10.5, color=RED,
            weight="bold")
    ax.annotate("", xy=(0.655, 0.168), xytext=(0.775, 0.168),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.7,
                                mutation_scale=11))
    ax.text(0.795, 0.168, "第四问建模结果", ha="left", va="center",
            fontsize=10.5, color="0.10", weight="bold")

    rule("内容", width=60)
    for s in ("唯一变量：电价矩阵（附件1 固定 → 附件4 逐日波动）",
              "情形一（对应问题二）14,634,080.5 元（+4.91%）",
              "情形二（对应问题三）14,210,954.8 元（+4.57%）",
              "涨价 97.2% 来自逐日漂移，日购电量与当日均价 r=+0.9651"):
        log("    " + s)
    save_fig(fig, "fig_第四问思维导图.png", FIGDIR)
    write_report("q4_思维导图说明.txt")


if __name__ == "__main__":
    main()
