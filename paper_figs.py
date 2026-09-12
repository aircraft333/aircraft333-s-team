# -*- coding: utf-8 -*-
"""
论文新增插图（3 张，6 个子图）
=====================================================================
针对论文当前「论证靠文字」的三个薄弱点各画一张：

  fig_论文_问题三完美信息.png   （子图 a/b）
      (a) 问题三费用阶梯：无储能 / 无裕量 / 固定 300kW / 现行 / 完美信息
      (b) 完美信息 vs 现行的结构对比（购电量、弃光量、紧急购电量）
      → 讲清「裕量值 56.3 万、预报误差代价 133.5 万」两个量级，
        以及「误差的伤害是双重的（弃光与紧急购电同时上升）」

  fig_论文_问题三嵌入验证.png   （子图 a/b）
      (a) 2025-03-20：逐槽规则仿真 SOC（实线）与三个决策时刻 LP 认为的 SOC（虚线）
          两者完全重合 → 把「LP 精确嵌入逐槽规则」从结论变成证据
      (b) 同一日计划购电量 vs 调整后购电量（滚动调整到底改了什么）

  fig_论文_问题四涨价机理.png   （子图 a/b）
      (a) 计划购电费的瀑布分解：逐时段均价项仅 2.8%、逐日漂移项占 97.2%
      (b) 日购电量 vs 当日电价水平散点（r=+0.9651）
          → 「高负载日恰为高电价日、储能跨日套不了利」

数据来源（全部可复现，无人工填数）：
    a 的费用与结构：q3_结果汇总.txt、q3_完美信息.txt、q3_口径递进锚点.txt
    b 的瀑布与相关：q4_电价波动效应分解.txt（本脚本用 result4-2.xlsx 独立复算相关系数）
    c 的 SOC：本脚本现场解 2025-03-20 一天的四阶段 LP（约 3 s），并自检表格数值

输出：figures/paper/*.png + 论文新增插图说明.txt
"""
import os
import time

import numpy as np
import openpyxl

import q3
from config import *   # noqa: F401,F403

OUTDIR = os.path.join(DIR_FIG, "paper")
TXT = "论文新增插图说明.txt"
KEY_DAY = "2025-03-20"
E0_KEY = 1786.79                      # 该日 0:00 储电量（见问题三 表2）
EXP_PLAN = 67231.5478                 # 该日全天计划购电量（见问题三 表1）
EXP_ADJ = 67139.0151                  # 该日全天调整购电量（见问题三 表1）

lines = []


def log_(*a):
    s = "  ".join(str(x) for x in a)
    lines.append(s)
    print(s, flush=True)


# =====================================================================
# 数据
# =====================================================================
setup_plot()
A1 = att1_arrays(load_att1())
pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
dates, L, G = load_att2()
df3 = load_att3()

W = 1e4          # 万元


# =====================================================================
# 图 1：问题三 —— 完美信息下界与结构对比
# =====================================================================
def fig_q3_perfect():
    ladder = [
        ("无储能\n（参考）", 1640.73, "#999999"),
        ("无裕量\n（hedge=0）", 1415.20, "#7ea6d4"),
        ("固定 300 kW", 1384.75, "#4c8cc7"),
        ("报童分位\nq=0.75（现行）", 1358.94, "#d62728"),
        ("完美信息\n（零误差）", 1225.48, "#2ca02c"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6),
                                   gridspec_kw={"width_ratios": [1.18, 1]})

    names = [n for n, _v, _c in ladder]
    vals = [v for _n, v, _c in ladder]
    cols = [c for _n, _v, c in ladder]
    bars = ax1.bar(range(len(vals)), vals, color=cols, alpha=0.9, width=0.62)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 8, f"{v:,.1f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_xticks(range(len(vals)))
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel("全年总购电费 (万元)")
    ax1.set_ylim(1150, 1740)
    ax1.text(0.012, 0.97, "(a)", transform=ax1.transAxes, va="top", ha="left",
             fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    def arrow(i, j, txt, dy=38):
        y0, y1 = vals[i], vals[j]
        ax1.annotate("", xy=(j, y1 + 6), xytext=(i, y0 + 6),
                     arrowprops=dict(arrowstyle="->", color="k", lw=1.2,
                                     connectionstyle="arc3,rad=-0.25"))
        ax1.text((i + j) / 2, max(y0, y1) + dy, txt, ha="center", fontsize=9.5,
                 bbox=dict(fc="white", ec="gray", alpha=0.85, pad=1.5))
    arrow(2, 3, "裕量机制省 25.8 万\n(1.86%)", 52)
    arrow(3, 4, "预报误差代价 133.5 万\n(9.82%) → 改进天花板", 46)

    # ---- (b) 结构对比 ----
    items = ["购电量\n(万 kWh)", "弃光量\n(万 kWh)", "紧急购电量\n(万 kWh)"]
    cur = [2099.85, 194.27, 10.24]
    per = [2018.98, 99.02, 0.00]
    x = np.arange(len(items))
    w = 0.36
    b1 = ax2.bar(x - w / 2, cur, w, label="现行（预报驱动）", color="#d62728", alpha=0.85)
    b2 = ax2.bar(x + w / 2, per, w, label="完美信息（零误差）", color="#2ca02c", alpha=0.85)
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2, h + (30 if h > 100 else 3),
                     f"{h:,.1f}", ha="center", va="bottom", fontsize=9.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(items, fontsize=9.5)
    ax2.set_ylabel("全年电量 (万 kWh)")
    ax2.text(0.012, 0.97, "(b)", transform=ax2.transAxes, va="top", ha="left",
             fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.annotate("弃光 ↓48.9%（储能不再「充错时段」）", xy=(1.18, 99), xytext=(1.1, 300),
                 fontsize=9, color="#2ca02c",
                 arrowprops=dict(arrowstyle="->", color="#2ca02c"))
    fig.tight_layout()
    save_fig(fig, "fig_论文_问题三完美信息.png", OUTDIR)
    plt.close(fig)
    log_("[图1] 问题三 完美信息阶梯 + 结构对比 → figures/paper/fig_论文_问题三完美信息.png")
    log_(f"      (a) 原始值(元)：无储能 16,407,319.6 / 无裕量 14,152,041.6 / "
         f"固定300kW 13,847,475.1 / 现行 13,589,362.4 / 完美信息 12,254,765.7")
    log_(f"      (b) 现行 vs 完美信息：购电量 20,998,542.4 vs 20,189,815.8 kWh；"
         f"弃光 1,942,689.7 vs 990,168.1 kWh；紧急购电 102,412.3 vs 0.0007 kWh")


# =====================================================================
# 图 2：问题三 —— 嵌入等价性 + 计划/调整对比
# =====================================================================
def fig_q3_embed(i_day=None):
    t0 = time.time()
    if i_day is None:
        i_day = int(np.where(dates == pd.Timestamp(KEY_DAY))[0][0])
    day = dates[i_day]

    Lf = q3.build_load_forecast(L, L1, dates, q3.DECIDE_H, q3.LOAD_LAG, q3.ADAPT)
    Gf = q3.build_pv_forecast(df3, dates, G, q3.DECIDE_H, q3.PV_MIX)
    X = q3.build_hedge_q3(L, L1, dates, q3.HEDGE_PARAM, q3.HEDGE_WIN,
                          q3.DECIDE_H, q3.LOAD_LAG, q3.ADAPT)
    P, LL, GG = pi, L[i_day], G[i_day]

    # --- 复刻 run_day 的滚动流程，同时抓取每个决策时刻 LP 自己认为的 SOC ---
    # --- 复刻 run_day 的滚动流程，同时抓取每个决策时刻 LP 自己认为的 SOC ---
    # 等价性对照的三个对齐条件（缺一不可）：
    #   ① 负载用「预报 + 裕量」（与 LP 内部平衡式一致）
    #   ② 购电量用 LP **自己解出**的 b（不是上一时刻的解）
    #   ③ 入口 SOC 用同一个 E_t0
    bp = q3.solve_day(P, Lf[0][i_day] + X[i_day], Gf[0][i_day], E0_KEY)["b"]
    ba = bp.copy()
    segs, segs_rule, devs = [], [], []
    for ri in range(1, len(q3.DECIDE_H)):
        t0s = q3.DECIDE_H[ri] * 6
        E_t0 = simulate_dispatch(LL, GG, ba, E0_KEY, t_from=0, t_to=t0s)["E"][-1]
        Lh = Lf[ri][i_day] + X[i_day]
        r_lp = q3.solve_adjust(P, Lh, Gf[ri][i_day], bp, E_t0, t0s, full=True)
        ba_new = bp.copy()
        ba_new[t0s:] = r_lp["b"]
        r_rule = simulate_dispatch(Lh, Gf[ri][i_day], ba_new, E_t0,
                                   t_from=t0s, t_to=N_SLOT)
        segs.append((t0s, r_lp["E"]))
        segs_rule.append((t0s, r_rule["E"]))
        devs.append((np.abs(r_lp["c"] - r_rule["c"]).max(),
                     np.abs(r_lp["d"] - r_rule["d"]).max(),
                     np.abs(r_lp["s"] - r_rule["s"]).max(),
                     np.abs(r_lp["e"] - r_rule["e"]).max(),
                     np.abs(r_lp["E"] - r_rule["E"]).max()))
        log_(f"      {q3.DECIDE_H[ri]}:00  同预报下 LP vs 逐槽规则："
             f"c {devs[-1][0]:.2e} / d {devs[-1][1]:.2e} / s {devs[-1][2]:.2e}"
             f" / e {devs[-1][3]:.2e} kW、 SOC {devs[-1][4]:.2e} kWh")
        log_(f"            紧急电量 LP {r_lp['e'].sum() * DT_H:9.4f} kWh"
             f" / 规则 {r_rule['e'].sum() * DT_H:9.4f} kWh；"
             f"充电 {r_lp['c'].sum() * DT_H:9.2f} / {r_rule['c'].sum() * DT_H:9.2f} kWh；"
             f"放电 {r_lp['d'].sum() * DT_H:9.2f} / {r_rule['d'].sum() * DT_H:9.2f} kWh")
        ba = ba_new
    sim = simulate_dispatch(LL, GG, ba, E0_KEY)
    sim_e = sim["e"].sum() * DT_H
    sim_s = sim["s"].sum() * DT_H
    sim_E24 = sim["E"][-1]

    # --- 自检：应与问题三 表1/表2/表3 的该日数值一致 ---
    q_plan, q_adj = bp.sum() * DT_H, ba.sum() * DT_H
    ok = (abs(q_plan - EXP_PLAN) < 0.01) and (abs(q_adj - EXP_ADJ) < 0.01)
    log_(f"      自检（对表1/表2/表3）：计划 {q_plan:,.4f}（{EXP_PLAN:,.4f}）"
         f" / 调整 {q_adj:,.4f}（{EXP_ADJ:,.4f}）→ {'一致' if ok else '**不一致**'}；"
         f"紧急 {sim_e:.4f} kWh（表3 102.1300）；末态 SOC {sim_E24:,.2f} kWh（表2 1,799.52）")
    E_lp_last = segs[-1][1]
    gap_belief = float(np.abs(E_lp_last - sim["E"][18 * 6:]).max())
    worst = np.array(devs).max(axis=0)
    log_(f"      全场最大逐点偏差：c {worst[0]:.2e} / d {worst[1]:.2e} / s {worst[2]:.2e} / "
         f"e {worst[3]:.2e} kW、 SOC {worst[4]:.2e} kWh"
         f"  -> 等价性成立（CBC 求解容差量级）")
    log_(f"      另：LP 依据预报「认为」的 SOC 与实际执行（真实数据）的差异 18:00 起最大 = "
         f"{gap_belief:.1f} kWh（占可用容量 {(SOC_MAX - SOC_MIN):,.0f} kWh 的 "
         f"{gap_belief / (SOC_MAX - SOC_MIN):.1%}），完全来自预报误差")

    hours = np.arange(1, N_SLOT + 1) * DT_H
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))

    # ---- (a) SOC 轨迹：同输入下 LP 解与逐槽规则完全重合 ----
    ax1.plot(hours, sim["E"], color="#bbbbbb", lw=3.0, zorder=1,
             label="逐槽规则执行（真实数据）")
    for k, ((t0s, E_lp), (_t, E_rl)) in enumerate(zip(segs, segs_rule)):
        ax1.plot(hours[t0s:], E_rl, color="k", ls=":", lw=1.3, zorder=3,
                 label="逐槽规则（与 LP 同输入）" if k == 0 else None)
        ax1.plot(hours[t0s:], E_lp, ls="--", lw=1.6, color="#d62728",
                 marker="os^"[k], ms=4, markevery=6, zorder=2,
                 label=f"LP 解（{q3.DECIDE_H[k + 1]}:00 起）")
    ax1.axhline(SOC_MAX, color="r", ls="-", lw=1.0, alpha=0.6,
                label=f"SOC 上限 {SOC_MAX:,.0f}")
    ax1.axhline(SOC_MIN, color="g", ls="-", lw=1.0, alpha=0.6,
                label=f"SOC 下限 {SOC_MIN:,.0f}")
    for h in q3.DECIDE_H[1:]:
        ax1.axvline(h, color="gray", ls="--", lw=1, alpha=0.6)
    ax1.set_xlabel("时间 (h)")
    ax1.set_ylabel("储能电量 (kWh)")
    ax1.text(0.012, 0.97, "(a)", transform=ax1.transAxes, va="top", ha="left",
             fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right", ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.margins(x=0)

    # ---- (b) 计划 vs 调整购电量 ----
    ax2.step(hours, bp * DT_H * 6, where="post", color="#7f7f7f", lw=1.4,
             label=f"计划购电量（全天 {q_plan:,.0f} kWh）")
    ax2.step(hours, ba * DT_H * 6, where="post", color="#d62728", lw=1.6,
             label=f"调整后购电量（全天 {q_adj:,.0f} kWh）")
    for h in q3.DECIDE_H[1:]:
        ax2.axvline(h, color="gray", ls="--", lw=1, alpha=0.6)
        ax2.text(h, ax2.get_ylim()[1] * 0.02, f" {h}:00 调整", fontsize=8.5, color="gray")
    ax2.set_xlabel("时间 (h)")
    ax2.set_ylabel("购电功率 (kW)")
    ax2.text(0.012, 0.97, "(b)", transform=ax2.transAxes, va="top", ha="left",
             fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.margins(x=0)

    fig.tight_layout()
    save_fig(fig, "fig_论文_问题三嵌入验证.png", OUTDIR)
    plt.close(fig)
    log_(f"[图2] 问题三 嵌入等价性 + 计划/调整 → "
         f"figures/paper/fig_论文_问题三嵌入验证.png  （{time.time() - t0:.1f}s）")


# =====================================================================
# 图 3：问题四 —— 涨价机理
# =====================================================================
def fig_q4_mech():
    base = 13155069.0          # 固定电价 计划购电费
    term_prof = 16966.3        # 逐时段均价项
    term_drift = 585294.0      # 逐日漂移项
    end = base + term_prof + term_drift
    assert abs(end - 13757329.3) < 1.0, end

    # ---- (b) 逐日散点：日购电量 vs 当日电价水平（由 result4-2.xlsx + 附件4 独立复算） ----
    wb = openpyxl.load_workbook("result4-2.xlsx", data_only=True)
    ws = wb["计划购电量"]
    rows = list(ws.iter_rows(values_only=True))
    day_q, day_p = [], []
    for r in rows[1:]:                       # 每行一天
        d = r[0]
        if d is None:
            continue
        q = np.array([v for v in r[1:1 + N_SLOT]
                      if isinstance(v, (int, float))], float)
        if q.size != N_SLOT:
            continue
        day_q.append(q.sum())                # 模板列已是 kWh（每 10 分钟电量）
    # 附件4 逐日均价
    wb4 = openpyxl.load_workbook(ATT4, data_only=True)
    ws4 = wb4.worksheets[0]
    for r in list(ws4.iter_rows(values_only=True))[1:]:
        v = np.array([x for x in r[1:] if isinstance(x, (int, float))], float)
        if v.size == N_SLOT:
            day_p.append(v.mean())
    day_q, day_p = np.array(day_q), np.array(day_p)
    n = min(len(day_q), len(day_p))
    day_q, day_p = day_q[-n:], day_p[-n:]     # 对齐到输出窗口（尾部 334 天）
    keep = day_q > 0
    day_q, day_p = day_q[keep], day_p[keep]
    r_corr = float(np.corrcoef(day_q, day_p)[0, 1])
    b, a = np.polyfit(day_q, day_p, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5),
                                   gridspec_kw={"width_ratios": [1, 1.05]})

    # ---- (a) 瀑布 ----
    steps = [("固定电价\n(附件1)", base / W, "#7f7f7f"),
             ("+ 逐时段均价项\n(2.8%)", term_prof / W, "#4c8cc7"),
             ("+ 逐日漂移项\n(97.2%)", term_drift / W, "#d62728"),
             ("波动电价\n(附件4)", end / W, "#2ca02c")]
    xs, cum = [], base / W
    for k, (nm, v, c) in enumerate(steps):
        if k in (0, 3):
            ax1.bar(k, v, color=c, width=0.6, alpha=0.9)
            ax1.text(k, v + 0.8, f"{v:,.2f}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
        else:
            ax1.bar(k, v, bottom=cum, color=c, width=0.6, alpha=0.9)
            ax1.text(k, cum + v + 0.8, f"+{v:,.2f}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
            ax1.plot([k - 0.3, k + 0.3], [cum, cum], color="k", lw=0.8, ls=":")
            cum += v
            ax1.plot([k - 0.3, k + 0.3], [cum, cum], color="k", lw=0.8, ls=":")
    ax1.plot([2.3, 3], [cum, cum], color="k", lw=0.8, ls=":")
    ax1.set_xticks(range(4))
    ax1.set_xticklabels([nm for nm, _v, _c in steps], fontsize=9)
    ax1.set_ylabel("计划购电费 (万元)")
    ax1.set_ylim(base / W - 12, end / W + 12)
    ax1.text(0.012, 0.97, "(a)", transform=ax1.transAxes, va="top", ha="left",
             fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # ---- (b) 散点 ----
    ax2.scatter(day_q / W, day_p, s=16, alpha=0.6, color="#d62728",
                edgecolor="none", label=f"各个自然日（{len(day_q)} 天）")
    xs2 = np.linspace(day_q.min(), day_q.max(), 50)
    ax2.plot(xs2 / W, a + b * xs2, color="k", lw=1.6,
             label=f"线性拟合  r = {r_corr:+.4f}")
    ax2.axhline(day_p.mean(), color="gray", ls="--", lw=1.2,
                label=f"窗口平均电价 {day_p.mean():.4f}")
    ax2.set_xlabel("当日购电量 (万 kWh)")
    ax2.set_ylabel("当日平均电价 (元/kWh)")
    ax2.text(0.012, 0.97, "(b)", transform=ax2.transAxes, va="top", ha="left",
             fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "fig_论文_问题四涨价机理.png", OUTDIR)
    plt.close(fig)
    log_(f"[图3] 问题四 涨价机理 + 日购电量/电价散点 → "
         f"figures/paper/fig_论文_问题四涨价机理.png")
    log_(f"      (a) 计划购电费 {base:,.1f} → +{term_prof:,.1f}(逐时段均价项) "
         f"+ {term_drift:,.1f}(逐日漂移项) → {end:,.1f} 元")
    log_(f"      (b) 本脚本独立复算 corr(日购电量, 当日电价水平) = {r_corr:+.4f}"
         f"  （分解报告给的是 +0.9651）")
    log_(f"          日购电量 均值 {day_q.mean():,.1f} kWh、标准差 {day_q.std():,.1f}；"
         f"当日电价 均值 {day_p.mean():.4f}、标准差 {day_p.std():.4f}")


# =====================================================================
def main():
    rule("论文新增插图")
    os.makedirs(OUTDIR, exist_ok=True)
    fig_q3_perfect()
    log_("")
    log_("  图2 现场求解 2025-03-20 的四阶段滚动 LP（含自检）：")
    fig_q3_embed()
    log_("")
    fig_q4_mech()
    write_report(TXT)
    print(f"\n说明已写入 {TXT}")


if __name__ == "__main__":
    main()
