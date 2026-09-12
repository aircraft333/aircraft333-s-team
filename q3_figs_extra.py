# -*- coding: utf-8 -*-
"""问题三补充插图（3 张，均输出 PNG + 矢量 PDF）

F1 fig_问题三_决策时刻边际价值.png
     数据来源：q3.py 的 ablation() 输出 q3_预报时刻边际价值.txt
     (a) 各决策时刻组合的全年费用（相对“仅 0:00”的节省）
     (b) 边际价值瀑布：不调整 → 加 18:00 → 加 6:00/12:00 → 四时刻 → 加密
     (c) 对应的紧急购电量

F2 fig_问题三_预报版本修订.png
     数据来源：附件3（预报）+ 附件2（光伏实际），无需求解 LP
     (a) 四个发布版本的逐小时误差箱线 vs 相邻版本间的修订幅度箱线
     (b) 修订幅度 |ΔF| 与误差变化 |e_new|−|e_old| 的散点
         —— 若点云无斜率（相关≈0），说明“版本更新”主要是噪声，按它调整要付无谓的摩擦费

F3 fig_问题三_调整结构.png
     数据来源：result3.xlsx（计划购电量 / 调整购电量 / 紧急购电量）
     (a) 按时段分组的 δ⁺ / δ⁻ 总量（δ⁺=上调、δ⁻=下调）
     (b) 全年逐日“调整净增费用”的分布
     (c) 日净增调整量 vs 当日紧急购电量（调整是否成功削掉了紧急购电）

运行：python q3_figs_extra.py      （F2/F3 秒级；F1 只需读报告，不重解 LP）
"""
import os
import re

import numpy as np
import openpyxl

from config import *          # noqa: F401,F403

FIGDIR = os.path.join(DIR_FIG, "q3")
TXT_ABL = "q3_预报时刻边际价值.txt"
OUT_START = "2025-02-01"
REL_H = (0, 6, 12, 18)        # 附件3 的发布时刻
BLK4H = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]
LAB4H = ["0:00-4:00", "4:00-8:00", "8:00-12:00",
         "12:00-16:00", "16:00-20:00", "20:00-24:00"]


# =====================================================================
# F1：决策时刻的边际价值
# =====================================================================
def parse_ablation():
    """从 q3_预报时刻边际价值.txt 解析各档结果"""
    path = resolve(TXT_ABL)
    if not os.path.exists(path):
        return []
    pat = re.compile(r"^\s*(.+?)\s{2,}总费用\s+([\d,\.]+)\s*元\s+"
                     r"较不调整节省\s+(-?[\d,\.]+)\s*元\s+"
                     r"紧急购电\s+([\d,\.]+)\s*kWh")
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.match(line)
            if m:
                g = lambda s: float(s.replace(",", ""))       # noqa: E731
                rows.append((m.group(1).strip(), g(m.group(2)),
                             g(m.group(3)), g(m.group(4))))
    return rows


def fig_ablation(rows):
    if not rows:
        log("    [跳过] 未找到 %s（先跑 python q3.py ablate）" % TXT_ABL)
        return
    names = [r[0] for r in rows]
    cost = np.array([r[1] for r in rows])
    save = np.array([r[2] for r in rows])
    emg = np.array([r[3] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))

    # (a) 费用条形（以“仅 0:00”为基准画节省量）
    ax = axes[0]
    y = np.arange(len(names))[::-1]
    col = [C_NET if s > 0 else C_PRICE for s in save]
    ax.barh(y, save / 1e4, color=col, height=.62)
    for yi, s, c in zip(y, save, cost):
        off = 1.1 if s > 0 else -1.1
        ax.text(s / 1e4 + off, yi, f"{s:,.0f} 元\n({c/1e4:,.0f} 万元)",
                va="center", ha="left" if s > 0 else "right", fontsize=8.5)
    ax.axvline(0, color="0.3", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("相对“仅 0:00（不调整）”的节省（万元）")
    ax.set_xlim(-8, 58)
    ax.grid(alpha=.3, axis="x")
    ax.set_title("(a) 各决策时刻组合的全年费用", fontsize=11)

    # (b) 边际价值阶梯：按“信息量递增”排序的累计节省
    ax = axes[1]
    order = ["仅 0:00（不调整）", "0:00 + 6:00", "0:00 + 12:00", "0:00 + 18:00",
             "0:00 + 6:00 + 12:00", "0:00 + 6:00 + 12:00 + 18:00（题目给定时刻）"]
    idx = [names.index(n) for n in order if n in names]
    xs = np.arange(len(idx))
    ys = cost[idx] / 1e4
    ax.plot(xs, ys, "o-", color=C_PRICE, lw=2, ms=7)
    for xi, yi, nm in zip(xs, ys, [names[i] for i in idx]):
        ax.annotate(f"{yi:,.1f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5)
    k3 = next((j for j, n in enumerate(names) if "每 3 小时" in n), None)
    k2 = next((j for j, n in enumerate(names) if "每 2 小时" in n), None)
    k1 = next((j for j, n in enumerate(names) if "每小时" in n), None)
    xs2, ys2, lb2 = [], [], []
    for j, lb in ((k3, "每 3 h\n(8 次)"), (k2, "每 2 h\n(12 次)"), (k1, "每小时\n(24 次)")):
        if j is not None:
            xs2.append(len(idx) + len(xs2))
            ys2.append(cost[j] / 1e4)
            lb2.append(lb)
    if xs2:
        ax.plot(xs2, ys2, "s--", ms=9, color=C_LOAD, lw=1.6, label="加密决策时刻")
        for xi, yi, lb in zip(xs2, ys2, lb2):
            ax.annotate(lb, (xi, yi), textcoords="offset points", xytext=(0, -30),
                        ha="center", fontsize=8, color=C_LOAD)
        base0 = cost[idx[0]] / 1e4
        ax.axhline(base0, color="0.45", ls=":", lw=1.4)
        ax.annotate("完全不调整的水位", (len(idx) + len(xs2) - 0.1, base0),
                    textcoords="offset points", xytext=(-4, 7), ha="right",
                    fontsize=8.5, color="0.35")
        ax.set_xticks(np.append(xs, xs2))
        ax.set_xticklabels(["仅\n0:00", "+6:00", "+12:00", "+18:00", "+6+12",
                            "+6+12+18\n(题目)"] + lb2, fontsize=7.8)
    else:
        ax.set_xticks(xs)
        ax.set_xticklabels(["仅\n0:00", "+6:00", "+12:00", "+18:00",
                            "+6+12", "+6+12+18\n(题目)"], fontsize=8.5)
    ax.set_ylabel("全年总购电费（万元）")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8.5, loc="upper center")
    ax.set_title("(b) 决策时刻增加 ≠ 费用单调下降", fontsize=11)

    # (c) 紧急购电量
    ax = axes[2]
    ys2 = emg / 1e4
    axes[2].barh(y, ys2, color=C_PV, height=.62)
    for yi, v in zip(y, ys2):
        ax.text(v + .15, yi, f"{v:,.2f}", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("紧急购电量（万 kWh）")
    ax.set_xlim(0, max(ys2) * 1.28)
    ax.grid(alpha=.3, axis="x")
    ax.set_title("(c) 对应的紧急购电量", fontsize=11)

    fig.tight_layout()
    save_fig(fig, "fig_问题三_决策时刻边际价值.png", FIGDIR)


# =====================================================================
# F2：预报版本修订 —— 版本更新携带多少真信息？
# =====================================================================
def fig_revision():
    df3 = load_att3()
    dates, L, G = load_att2()
    msk = dates >= pd.Timestamp(OUT_START)

    # 实际光伏的“整点点值”：与附件3 同口径（附件3 的“预报k小时”是发布后第 k 个
    # 整点的瞬时功率点值，不是小时均值 —— 见 q3.build_pv_forecast 的口径说明）。
    # 附件2 的 G 是 10 min 均值，故整点 H:00 的点值取“以 H:00 为右端点”的那个时段，
    # 即当日第 6H-1 个时段（0-based）。
    Gpt = {}
    for i, d in enumerate(dates):
        base = pd.Timestamp(d)
        for H in range(1, 25):
            Gpt[base + pd.Timedelta(hours=H)] = float(G[i, 6 * H - 1])

    recs = []      # (target_ts, release, fc, actual)
    for d in dates[msk]:
        for r in REL_H:
            f = att3_forecast(df3, d, r)
            if f is None:
                continue
            base = pd.Timestamp(d) + pd.Timedelta(hours=r)
            for i in range(24):                  # 发布后第 i+1 个整点
                tgt = base + pd.Timedelta(hours=i + 1)
                if tgt not in Gpt:
                    continue
                recs.append((tgt, r, i + 1, float(f[i]), Gpt[tgt]))
    R = pd.DataFrame(recs, columns=["tgt", "rel", "lead", "fc", "act"])
    R["err"] = R["fc"] - R["act"]
    log(f"    版本-时点配对数 {len(R):,}；覆盖 {R['tgt'].min()} ~ {R['tgt'].max()}")

    piv = R.pivot_table(index="tgt", columns="rel", values="fc")
    err_piv = R.pivot_table(index="tgt", columns="rel", values="err")
    lead_piv = R.pivot_table(index="tgt", columns="rel", values="lead")
    # 注意：附件3 的四个版本是**滚动发布**的，对同一个目标时点，谁“更新”取决于目标落在
    # 周期中的位置。必须只保留“新版确实是后发布（提前量更短）”的时点，否则会拿不同轮次的
    # 预报做比较。等价地：保留 lead_old − lead_new > 0 的时点。
    pairs = [(6, 0), (12, 6), (18, 12)]          # (新版, 旧版)，相隔 6 h
    pair_lab, pair_stat, rev_val, dve = [], [], [], []
    for new, old in pairs:
        sub = R[R["rel"].isin([new, old])]
        w = sub.pivot_table(index="tgt", columns="rel", values=["fc", "err"]).dropna()
        lw = sub.pivot_table(index="tgt", columns="rel", values="lead").reindex(w.index)
        w, lw = w[(lw[old] - lw[new]) > 0], lw[(lw[old] - lw[new]) > 0]
        if not len(w):
            continue
        dw = (w[("err", old)].abs() - w[("err", new)].abs())     # 正 = 新版更准
        lab = f"{old}:00→{new}:00"
        pair_lab.append(lab)
        pair_stat.append((w[("err", old)].abs().mean(),
                          w[("err", new)].abs().mean(), dw.mean()))
        rev_val.append((w[("fc", new)] - w[("fc", old)]).abs().to_numpy())
        dve.append(dw.to_numpy())
        log(f"    {lab}  N={len(w):,}   旧版 MAE {pair_stat[-1][0]:7.2f} kW"
            f" → 新版 {pair_stat[-1][1]:7.2f} kW"
            f"（平均降幅 {pair_stat[-1][2]:+7.2f} kW，新版更好的比例 "
            f"{(dw > 0).mean():.1%}）"
            f"   corr(|ΔF|, Δ|e|) = {np.corrcoef(rev_val[-1], dve[-1])[0, 1]:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # (a) 同一个目标时点，用更新版本的误差对比
    ax = axes[0]
    x = np.arange(len(pair_lab))
    oldv = [p[0] for p in pair_stat]
    newv = [p[1] for p in pair_stat]
    ax.bar(x - .19, oldv, .36, color=C_LOAD, label="旧版（提前量多 6 h）")
    ax.bar(x + .19, newv, .36, color=C_NET, label="新版（提前量少 6 h）")
    for xi, (o, n, d) in zip(x, pair_stat):
        ax.annotate(f"{o:,.1f}", (xi - .19, o), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8.5)
        ax.annotate(f"{n:,.1f}", (xi + .19, n), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8.5)
        ax.annotate(f"平均降 {d:,.0f} kW", (xi, max(o, n) * 1.10),
                    ha="center", fontsize=9, color=C_PRICE)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_lab)
    ax.set_ylabel("同目标时点的 MAE（kW）")
    ax.set_ylim(0, max(oldv + newv) * 1.28)
    ax.legend(fontsize=9)
    ax.grid(alpha=.3, axis="y")
    ax.set_title("(a) 更新版本确实更准（同目标时点配对比较）", fontsize=11)

    # (b) 修订幅度 vs 误差下降
    ax = axes[1]
    for v, de, lb, c in zip(rev_val, dve, pair_lab, (C_NET, C_LOAD, C_PV)):
        ax.scatter(v, de, s=5, alpha=.22, color=c, label=lb)
        if len(v) > 30 and np.std(v) > 0 and np.std(de) > 0:
            k = np.polyfit(v, de, 1)
            xs = np.linspace(0, np.quantile(v, .995), 50)
            ax.plot(xs, np.polyval(k, xs), color=c, lw=2)
            rr = np.corrcoef(v, de)[0, 1]
            ax.annotate(f"{lb}  r={rr:+.3f}", (xs[-1], np.polyval(k, xs[-1])),
                        fontsize=8.5, color=c, ha="right",
                        va="top" if rr < 0 else "bottom")
    ax.axhline(0, color="0.3", lw=1, ls="--")
    ax.set_xlim(0, np.quantile(np.concatenate(rev_val), .995))
    lim = np.quantile(np.concatenate(dve), .99)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("相邻版本对同一时点的预报修订幅度 |ΔF|（kW）")
    ax.set_ylabel("误差下降 |e_old| − |e_new|（kW，正值=新版更准）")
    ax.legend(fontsize=8.5, markerscale=2.5, loc="upper left")
    ax.grid(alpha=.3)
    ax.set_title("(b) 修订幅度越大，改进越多（修订携带真信息）", fontsize=11)

    fig.tight_layout()
    save_fig(fig, "fig_问题三_预报版本修订.png", FIGDIR)
    return R


# =====================================================================
# F3：调整结构 —— 调在哪里、调了多少、有没有削掉紧急购电
# =====================================================================
def fig_adjust():
    pi = att1_arrays(load_att1())["price"]
    wb = openpyxl.load_workbook(resolve(OUT3), data_only=True)
    wp, wa, we = wb["计划购电量"], wb["调整购电量"], wb["紧急购电量"]

    dates, L, G = load_att2()
    d0 = int(np.argmax(dates >= pd.Timestamp(OUT_START)))
    nd = int((dates >= pd.Timestamp(OUT_START)).sum())

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    bp = np.array([[num(wp.cell(row=2 + k, column=2 + t).value) or 0.0
                    for t in range(N_SLOT)] for k in range(nd)])
    ba = np.array([[num(wa.cell(row=2 + k, column=2 + t).value) or 0.0
                    for t in range(N_SLOT)] for k in range(nd)])
    up = np.maximum(ba - bp, 0.0)          # δ⁺ 上调（kWh）
    dn = np.maximum(bp - ba, 0.0)          # δ⁻ 下调（kWh）
    fee_up = np.array([float((1.5 * pi * u).sum()) for u in up])     # 元
    fee_dn = np.array([float((0.5 * pi * d).sum()) for d in dn])
    net_fee = fee_up - fee_dn              # 相对计划口径的净增费用

    # 逐日紧急购电量
    emg_day = np.zeros(nd)
    cur = None
    for r in range(2, we.max_row + 1):
        dv = we.cell(row=r, column=1).value
        if dv is not None:
            cur = pd.Timestamp(dv).normalize()
        q = num(we.cell(row=r, column=3).value)
        if cur is None or q is None:
            continue
        k = (cur - pd.Timestamp(OUT_START)).days
        if 0 <= k < nd:
            emg_day[k] += q

    net_q = ba.sum(axis=1) - bp.sum(axis=1)      # 日净增购电量 kWh
    log(f"    δ⁺ 合计 {up.sum():,.1f} kWh；δ⁻ 合计 {dn.sum():,.1f} kWh")
    log(f"    调整相关费用：上调 1.5π {fee_up.sum():,.1f} 元，"
        f"下调退 0.5π {-fee_dn.sum():,.1f} 元，净增 {net_fee.sum():,.1f} 元")
    log(f"    逐日净增费用：均值 {net_fee.mean():,.1f} 元、中位 {np.median(net_fee):,.1f} 元、"
        f"为正的天数 {int((net_fee > 0).sum())}/{nd}")
    log(f"    corr(日净增购电量, 当日紧急购电量) = "
        f"{np.corrcoef(net_q, emg_day)[0, 1]:+.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # (a) 按时段分组的 δ⁺/δ⁻
    ax = axes[0]
    u = [up[:, a:b].sum() / 1e4 for a, b in BLK4H]
    d_ = [dn[:, a:b].sum() / 1e4 for a, b in BLK4H]
    x = np.arange(6)
    ax.bar(x, u, .6, color=C_PV, label=r"$\delta^{+}$ 上调（多买，1.5$\pi$）")
    ax.bar(x, [-v for v in d_], .6, color=C_LOAD, label=r"$\delta^{-}$ 下调（少买，退 0.5$\pi$）")
    for xi, (a, b) in zip(x, zip(u, d_)):
        ax.text(xi, a + .5, f"{a:,.1f}", ha="center", fontsize=8.5)
        ax.text(xi, -b - .9, f"{b:,.1f}", ha="center", fontsize=8.5)
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(LAB4H, fontsize=8.5, rotation=20)
    ax.set_ylabel("全年累计调整量（万 kWh）")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=.3, axis="y")
    ax.set_title("(a) 调整发生在哪些时段", fontsize=11)

    # (b) 逐日净增费用分布
    ax = axes[1]
    ax.hist(net_fee, bins=44, color=C_NET, alpha=.75, edgecolor="white", lw=.5)
    for v, c, lb in ((net_fee.mean(), C_PRICE, "均值"),
                     (float(np.median(net_fee)), C_LOAD, "中位数")):
        ax.axvline(v, color=c, ls="--", lw=1.8, label=f"{lb} {v:,.0f} 元")
    ax.axvline(0, color="0.3", lw=1, ls=":")
    ax.set_xlabel("逐日“调整净增费用”（元）")
    ax.set_ylabel("天数")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=.3)
    ax.set_title("(b) 调整并非每天赚钱", fontsize=11)

    # (c) 净增购电量 vs 紧急购电量
    ax = axes[2]
    ax.scatter(net_q / 1e3, emg_day / 1e3, s=16, alpha=.55, color=C_PRICE)
    r = np.corrcoef(net_q, emg_day)[0, 1]
    if np.std(net_q) > 0:
        k = np.polyfit(net_q, emg_day, 1)
        xs = np.linspace(net_q.min(), net_q.max(), 50)
        ax.plot(xs / 1e3, np.polyval(k, xs) / 1e3, color="k", lw=1.8)
    ax.annotate(f"$r$ = {r:+.3f}", (0.04, 0.93), xycoords="axes fraction",
                fontsize=10)
    ax.set_xlabel("日净增购电量（千 kWh）")
    ax.set_ylabel("当日紧急购电量（千 kWh）")
    ax.grid(alpha=.3)
    ax.set_title("(c) 多买能否少触发紧急购电", fontsize=11)

    fig.tight_layout()
    save_fig(fig, "fig_问题三_调整结构.png", FIGDIR)


def main():
    arm_report("q3_补充插图.txt")
    rule("问题三补充插图")
    setup_plot()

    rule("F1 决策时刻的边际价值", width=60)
    fig_ablation(parse_ablation())

    rule("F2 预报版本修订", width=60)
    fig_revision()

    rule("F3 调整结构", width=60)
    fig_adjust()

    write_report("q3_补充插图.txt")


if __name__ == "__main__":
    main()
