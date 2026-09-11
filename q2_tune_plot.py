# -*- coding: utf-8 -*-
"""问题二安全裕量：不同取值下的最终结果（绘图）
=====================================================================
数据来源：`q2_fine_tune.py` 的实测输出（2026-09-11，全年 334 天，`q2_裕量精细寻优.txt`）
  · 扫描 A：固定 win=60，扫分位 q              （7 个点）
  · 扫描 B：固定 q=0.76，扫回看窗口 win        （6 个点，含 win=60）
  · 扫描 C：固定 win=30，重扫分位 q            （6 个点，含 q=0.76）

本脚本**只绘图、不重算 LP**（重算请运行 `python q2_fine_tune.py`，约 8 分钟）。

四张子图：
  (a) 分位水平 q → 总购电费（win=30 与 win=60 两条线对比）
  (b) 回看窗口 win → 总购电费（q=0.76）
  (c) 代表方案的费用构成（计划购电费 + 紧急购电费）
  (d) 「日均裕量 vs 总购电费」：裕量不是越大越好，存在最优区间

产出：figures/q2/fig_裕量寻优.png
"""
import numpy as np

from config import *

FIGDIR = os.path.join(DIR_FIG, "q2")
OKABE = "#0072B2"

# ---------------------------------------------------------------- 实测数据
# 每条记录：q, win, 日均裕量(kW), 计划购电费, 紧急购电费
# —— 扫描 A（win=60）与扫描 C（win=30）：q 扫描
SCAN_Q_60 = [
    (0.70, 60, 164, 12_891_345.1, 1_157_436.8),
    (0.73, 60, 194, 13_014_693.8,   987_337.7),
    (0.74, 60, 204, 13_057_653.5,   933_370.7),
    (0.75, 60, 215, 13_101_901.1,   881_137.4),
    (0.76, 60, 226, 13_147_498.1,   832_482.4),
    (0.77, 60, 237, 13_194_574.7,   785_875.1),
    (0.80, 60, 273, 13_347_034.6,   654_941.1),
]
SCAN_Q_30 = [
    (0.72, 30, 185, 12_980_725.2, 1_011_052.0),
    (0.74, 30, 206, 13_066_701.0,   895_523.6),
    (0.75, 30, 216, 13_110_585.1,   842_016.1),
    (0.76, 30, 227, 13_155_069.0,   794_039.5),
    (0.78, 30, 250, 13_252_626.2,   698_245.1),
    (0.80, 30, 275, 13_355_155.4,   610_673.5),
]
# —— 扫描 B（q=0.76）：win 扫描
SCAN_W = [
    (0.76,  20, 228, 13_157_868.2, 853_960.6),
    (0.76,  30, 227, 13_155_069.0, 794_039.5),
    (0.76,  45, 227, 13_153_842.1, 802_326.7),
    (0.76,  60, 226, 13_147_498.1, 832_482.4),
    (0.76,  90, 222, 13_132_727.4, 877_529.6),
    (0.76, 120, 219, 13_115_665.5, 919_656.3),
]
# —— 代表方案（问题二全年结果，见 README「结果汇总」）
PLANS = [
    ("无裕量\n(hedge=0)",       12_195_725.6, 3_032_588.7),
    ("固定 300 kW\n均匀裕量",    13_488_522.9,   633_131.1),
    ("报童解析值\nq=0.80, win=60", 13_347_034.6,   654_941.1),
    ("上一版主口径\nq=0.75, win=60", 13_101_901.1,   881_137.4),
    ("最优\nq=0.76, win=30",    13_155_069.0,   794_039.5),
]
Q_THEORY = 0.80          # 报童解析值 1 − 1/k，k=5


def tot(rec):
    return rec[3] + rec[4]


def main():
    setup_plot()
    # 拆成两张图，每张 1 行 2 列（便于在论文里单独插入与排版）；不再用总标题
    figA, axA = plt.subplots(1, 2, figsize=(13.8, 4.8))
    figB, axB = plt.subplots(1, 2, figsize=(13.8, 4.8))

    # ------------------------------------------------ (a) 分位水平 q
    ax = axA[0]
    for data, win, col, ls in ((SCAN_Q_30, 30, C_PRICE, "-"),
                               (SCAN_Q_60, 60, OKABE, "--")):
        qs = [r[0] for r in data]
        ys = [tot(r) / 1e4 for r in data]
        k = int(np.argmin(ys))
        ax.plot(qs, ys, ls, marker="o", ms=6, lw=LW, color=col,
                label=f"回看窗口 win={win} 天")
        ax.plot([qs[k]], [ys[k]], "*", ms=16, color=col, zorder=5)
        ax.annotate(f"{ys[k]:,.1f}", (qs[k], ys[k]), textcoords="offset points",
                    xytext=(6, -14), fontsize=9, color=col)
    ax.axvline(Q_THEORY, color=C_NET, ls=":", lw=LW)
    ax.annotate(f"报童解析值\nq* = 1-1/k = {Q_THEORY:.2f}", (Q_THEORY, ax.get_ylim()[0]),
                xytext=(Q_THEORY - 0.085, ax.get_ylim()[0] + 3), fontsize=8.5,
                color=C_NET)
    ax.set_xlabel("安全裕量分位数 $q$")
    ax.set_ylabel("全年总购电费（万元）")
    ax.set_title("(a) 分位水平 $q$ 的影响（星号 = 各窗口下的最优点）")
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)

    # ------------------------------------------------ (b) 回看窗口 win
    ax = axA[1]
    ws = [r[1] for r in SCAN_W]
    ys = [tot(r) / 1e4 for r in SCAN_W]
    k = int(np.argmin(ys))
    ax.plot(ws, ys, "o-", ms=6, lw=LW, color=C_PRICE)
    ax.plot([ws[k]], [ys[k]], "*", ms=16, color=C_PRICE, zorder=5)
    ax.annotate(f"最优 {ws[k]} 天\n{ys[k]:,.1f} 万元", (ws[k], ys[k]),
                textcoords="offset points", xytext=(10, 8), fontsize=9, color=C_PRICE)
    w60 = [r for r in SCAN_W if r[1] == 60][0]
    ax.plot([60], [tot(w60) / 1e4], "s", ms=8, color=OKABE, zorder=5)
    ax.annotate(f"原先采用的 60 天\n{tot(w60) / 1e4:,.1f} 万元（贵 "
                f"{tot(w60) - tot(SCAN_W[k]):,.0f} 元）",
                (60, tot(w60) / 1e4), textcoords="offset points",
                xytext=(-104, 26), fontsize=8.5, color=OKABE)
    ax.set_xlabel("回看窗口长度 win（天）")
    ax.set_ylabel("全年总购电费（万元）")
    ax.set_title("(b) 回看窗口长度的影响（q=0.76）")
    ax.set_xticks(ws)
    ax.grid(alpha=.3)
    ax.legend(["分位数规则", "最优点", "旧口径"], fontsize=9)

    # ------------------------------------------------ (c) 代表方案构成
    ax = axB[0]
    names = [p[0] for p in PLANS]
    plan = np.array([p[1] for p in PLANS]) / 1e4
    emg = np.array([p[2] for p in PLANS]) / 1e4
    ypos = np.arange(len(PLANS))[::-1]
    ax.barh(ypos, plan, height=.6, color=C_LOAD, label="计划购电费")
    ax.barh(ypos, emg, height=.6, left=plan, color=C_PV, label="紧急购电费")
    for y, p, e in zip(ypos, plan, emg):
        ax.text(p + e + 2, y, f"{p + e:,.1f}", va="center", fontsize=9)
        ax.text(p / 2, y, f"{p:,.0f}", va="center", ha="center", fontsize=8, color="white")
        ax.text(p + e / 2, y, f"{e:,.0f}", va="center", ha="center", fontsize=8,
                color="0.15")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("费用（万元）")
    ax.set_xlim(0, 1750)
    ax.set_title("(c) 代表方案的费用构成")
    ax.grid(alpha=.3, axis="x")
    ax.legend(fontsize=9, loc="upper left")

    # ------------------------------------------------ (d) 日均裕量 vs 总费用
    ax = axB[1]
    for data, nm, col, mk in ((SCAN_Q_60, "扫分位 q（win=60）", OKABE, "o"),
                              (SCAN_Q_30, "扫分位 q（win=30）", C_PRICE, "s"),
                              (SCAN_W, "扫回看窗口 win（q=0.76）", C_NET, "^")):
        xs = [r[2] for r in data]
        ys = [tot(r) / 1e4 for r in data]
        ax.plot(xs, ys, mk + "-", ms=6, lw=LW, color=col, alpha=.85, label=nm)
    ax.plot([300], [14121654.0 / 1e4], "P", ms=11, color=C_LOAD, zorder=5,
            label="固定 300 kW 均匀裕量")
    best = min(SCAN_Q_30 + SCAN_Q_60 + SCAN_W, key=tot)
    ax.plot([best[2]], [tot(best) / 1e4], "*", ms=18, color="k", zorder=6,
            label=f"联立最优 ({best[0]:.2f}, win={best[1]})")
    ax.set_xlabel("日均安全裕量（kW）")
    ax.set_ylabel("全年总购电费（万元）")
    ax.set_title("(d) 裕量不是越大越好：存在最优区间")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)

    figA.tight_layout()
    figB.tight_layout()
    save_fig(figA, "fig_裕量寻优_参数扫描.png", FIGDIR)
    save_fig(figB, "fig_裕量寻优_结果对比.png", FIGDIR)

    # ------------------------------------------------ 文字小结
    rule("问题二裕量参数寻优小结（图中数据）")
    log(f"{'方案':<26s}{'日均裕量':>10s}{'总购电费':>14s}{'相对最优':>12s}")
    log("-" * 64)
    for nm, p, e in PLANS:
        c = p + e
        log(f"{nm.replace(chr(10), ' '):<26s}{'':>10s}{c:>14,.1f}"
            f"{c - 13_949_108.5:>+12,.1f}")
    log(f"{'(a) q 最优 (0.76, win=60)':<26s}{226:>10d}{13_979_980.5:>14,.1f}"
        f"{13_979_980.5 - 13_949_108.5:>+12,.1f}")
    log(f"{'(b) win 最优 (0.76, win=30)':<26s}{227:>10d}{13_949_108.5:>14,.1f}{0:>+12,.1f}")
    log("")
    log("结论：")
    log("  1. 分位水平 q 在 0.75~0.78 之间极平坦（差 <0.03%），与报童解析值 0.80 也仅差 0.12%")
    log("     ⇒ 该参数不敏感，论文里用解析值即可；")
    log("  2. 回看窗口 win 才是真正的杠杆：30 天最优，比 60 天省 30,872 元（0.22%），")
    log("     且 20 天更差、90~120 天明显更差 —— 是内部极小点而非边界效应；")
    log("  3. 无论怎么选，分位数规则都优于固定 300 kW 均匀裕量（省 10~17 万元）。")
    write_report(resolve("q2_裕量寻优.txt"))


if __name__ == "__main__":
    main()
