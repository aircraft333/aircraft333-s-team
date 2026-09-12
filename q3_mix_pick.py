# -*- coding: utf-8 -*-
"""问题三：混合权重 m 的取值依据（一张图说清"三个判据、一条平谷"）

三张判据给出三个不同的"最优点"，但彼此差距都在噪声级：
    · 逐槽 MAE     → m = 0.35（本脚本按 0.05 步长重算）
    · 全年购电费用 → m = 0.30（读 q3_光伏权重复检.txt）
    · 解析（方差反比）→ m* = 0.381（忽略弱相关时 0.400）
本文取 m = 0.40：它是解析值 0.38~0.40 的整数上界，且落在共同的平坦谷底内。

图：fig_问题三_混合权重取值.png
    (a) 全区间 [0,1]：左轴 MAE（两个统计窗口），右轴 全年购电费用
    (b) 局部放大 [0.20, 0.55]：两条曲线各自减去自身最小值 → 相对劣化 (%)
报告：q3_混合权重取值.txt

运行：python q3_mix_pick.py     （只做加权与 MAE 统计，秒级）
"""
import os
import re

import numpy as np

from config import *          # noqa: F401,F403

FIGDIR = os.path.join(DIR_FIG, "q3")
TXT_COST = "q3_光伏权重复检.txt"
OUT_START = "2025-02-01"
HIST_N = 3                    # 历史外推窗口（前 3 天滑动平均）
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
U_SLOT = (np.arange(N_SLOT) + 1) / 6.0        # 时段右端点（小时）


def ma_scan():
    """按论文口径重算 m 的 MAE 扫描：0:00 发布版，逐槽（全 144 槽）

    预报 = m · 附件3(0:00 版, 整点点值插值到时段) + (1−m) · 前 3 天滑动平均
    """
    df3 = load_att3()
    dates, L, G = load_att2()
    hist = fc_ma(G, HIST_N, G[0])                 # (D, 144) 历史外推
    D = len(dates)
    f3 = np.full((D, N_SLOT), np.nan)
    for i, d in enumerate(dates):
        v = att3_forecast(df3, d, 0)
        if v is None:
            continue
        fh = {0: 0.0, 24: 0.0}
        for k in range(1, 25):
            fh[k] = float(v[k - 1])
        xs = np.array(sorted(fh))
        f3[i] = np.interp(U_SLOT, xs, np.array([fh[x] for x in xs]))

    out = {}
    for tag, msk in (("334", dates >= np.datetime64("2025-02-01")),
                     ("365", np.ones(D, bool))):
        ok = msk & ~np.isnan(f3).any(axis=1)
        g = G[ok]
        h = hist[ok]
        f3o = f3[ok]
        out[tag] = np.array([np.abs(m * f3o + (1 - m) * h - g).mean()
                             for m in GRID])
    return out


def cost_scan():
    """从 q3_光伏权重复检.txt 解析"最终口径"下的 mix → 总费用"""
    path = resolve(TXT_COST)
    if not os.path.exists(path):
        return None
    pat = re.compile(r"mix\s*=\s*([\d\.]+)\s+总费用\s+([\d,\.]+)\s*元")
    d = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                d[float(m.group(1))] = float(m.group(2).replace(",", ""))
    return d or None


def main():
    arm_report("q3_混合权重取值.txt")
    rule("问题三：混合权重 m 的取值依据")
    setup_plot()

    ms = ma_scan()
    cs = cost_scan()
    m_mae = {t: GRID[int(np.argmin(v))] for t, v in ms.items()}
    log(f"    MAE 最优（0.05 步长）：334 天 → m={m_mae['334']:.2f}"
        f"（{ms['334'].min():.1f} kW）；365 天 → m={m_mae['365']:.2f}"
        f"（{ms['365'].min():.1f} kW）")
    log(f"    自检：m=0 应为 154.5 / 148.7 kW，实得 {ms['334'][0]:.1f} / {ms['365'][0]:.1f}；"
        f"m=1 应为 197.5 / 191.7 kW，实得 {ms['334'][-1]:.1f} / {ms['365'][-1]:.1f}")
    if cs:
        mk = min(cs, key=lambda k: cs[k])
        log(f"    费用最优：m={mk:.2f} → {cs[mk]:,.1f} 元"
            f"（比 m=0.40 省 {cs[0.40] - cs[mk]:,.1f} 元 = "
            f"{(cs[0.40] - cs[mk]) / cs[0.40]:.3%}）")
        sub = {k: v for k, v in cs.items() if 0.2 - 1e-9 <= k <= 0.6 + 1e-9}
        log(f"    m∈[0.2,0.6] 费用极差 {max(sub.values()) - min(sub.values()):,.1f} 元"
            f" = {((max(sub.values()) - min(sub.values())) / min(sub.values())):.3%}")
    log(f"    解析最优：m* = 0.381（含弱相关 ρ=0.165）／0.400（ρ=0）")

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4))

    # ---------------- (a) 全区间 ----------------
    ax = axes[0]
    for tag, c, mk in (("334", C_LOAD, "s"), ("365", "0.55", "o")):
        ax.plot(GRID, ms[tag], mk + "-", color=c, ms=5, lw=1.8,
                label=f"MAE，{tag} 天窗口")
    j35 = int(np.argmin(ms["334"]))
    ax.plot([GRID[j35]], [ms["334"][j35]], "*", ms=17, color=C_LOAD, zorder=6)
    ax.annotate(f"MAE 最优 $m$={GRID[j35]:.2f}\n{ms['334'][j35]:.1f} kW",
                (GRID[j35], ms["334"][j35]), textcoords="offset points",
                xytext=(10, 14), fontsize=9, color=C_LOAD)
    ax.set_xlabel("混合权重 $m$（附件 3 的占比）")
    ax.set_ylabel("逐槽 MAE（kW）")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(alpha=.3)

    ax2 = ax.twinx()
    if cs:
        xs = sorted(cs)
        ax2.plot(xs, [cs[k] / 1e4 for k in xs], "o-", color=C_PRICE, ms=7, lw=2,
                 label="全年购电费用")
        mk = min(cs, key=lambda k: cs[k])
        ax2.plot([mk], [cs[mk] / 1e4], "*", ms=17, color=C_PRICE, zorder=6)
        ax2.annotate(f"费用最优 $m$={mk:.2f}\n{cs[mk]/1e4:,.1f} 万元",
                     (mk, cs[mk] / 1e4), textcoords="offset points",
                     xytext=(-8, -30), ha="center", fontsize=9, color=C_PRICE)
        ax2.set_ylabel("全年总购电费（万元）", color=C_PRICE)
        ax2.tick_params(axis="y", colors=C_PRICE)
    for x, lb, c, dy in ((0.381, "解析 $m^{*}$=0.381", "0.25", 0.02),
                         (0.40, "本文取 0.40", C_NET, 0.02)):
        ax.axvline(x, color=c, ls="--" if c != C_NET else "-.", lw=1.8)
        ax.annotate(lb, (x, ax.get_ylim()[1]), xytext=(3, -14 * (1 if dy else 1)),
                    textcoords="offset points", fontsize=8.5, color=c,
                    rotation=90, va="top")
    ax.set_title("(a) 三个判据给出的最优点并不相同（全区间）", fontsize=11)
    h1, l1 = ax.get_legend_handles_labels()
    if cs:
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper center")
    else:
        ax.legend(fontsize=8.5, loc="upper center")

    # ---------------- (b) 局部放大 ----------------
    ax = axes[1]
    lo, hi = 0.20, 0.60
    sel = (GRID >= lo - 1e-9) & (GRID <= hi + 1e-9)
    v = ms["334"][sel]
    ax.plot(GRID[sel], (v - v.min()) / v.min() * 100, "s-", color=C_LOAD,
            ms=6, lw=2, label="逐槽 MAE（334 天，相对自身最小值）")
    if cs:
        xs = np.array(sorted(k for k in cs if lo - 1e-9 <= k <= hi + 1e-9))
        ys = np.array([cs[k] for k in xs])
        ax.plot(xs, (ys - ys.min()) / ys.min() * 100, "o-", color=C_PRICE,
                ms=7, lw=2, label="全年购电费用（相对自身最小值）")
    ax.axvline(0.40, color=C_NET, ls="-.", lw=2)
    ax.annotate("本文取 $m$=0.40", (0.40, ax.get_ylim()[1]),
                textcoords="offset points", xytext=(-6, -6), ha="right",
                fontsize=9, color=C_NET, va="top")
    for x, c in ((0.30, C_PRICE), (0.35, C_LOAD)):
        ax.axvline(x, color=c, ls=":", lw=1.6)
    ax.set_xlabel("混合权重 $m$（附件 3 的占比）")
    ax.set_ylabel("相对自身最小值的劣化（%）")
    ax.set_xlim(lo - 0.01, hi + 0.01)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8.5, loc="upper center")
    ax.set_title("(b) 放大看：谷底平坦，0.30/0.35/0.40 相差不到 0.3%", fontsize=11)

    fig.tight_layout()
    save_fig(fig, "fig_问题三_混合权重取值.png", FIGDIR)
    write_report("q3_混合权重取值.txt")


if __name__ == "__main__":
    main()
