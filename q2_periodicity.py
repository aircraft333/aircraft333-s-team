# -*- coding: utf-8 -*-
"""第二问依据图：小区负载的周周期性
=====================================================================
目的
    证明「用上周同一天（lag = 7）预测小区负载」是合理的 —— 即负载曲线存在
    以 7 天为周期的强重复性，且 lag = 7 优于持续性预测(lag=1)与其他滞后期。

四联图
    (a) 一周内七天的平均曲线：工作日彼此接近，周末与工作日明显分离
    (b) 全年所有周一 / 所有周日曲线叠加：同星期的离散带很窄 ⇒ 周期性强
    (c) 逐日曲线的自相关系数 vs 滞后天数：在 lag = 7、14 处出现尖峰
    (d) 各类日前预测方法的逐槽 MAE 对比：lag = 7 明显最低

运行：python q2_periodicity.py
产出：figures/q2/fig_负载周周期性.png + q2_负载周周期性.txt
"""
import numpy as np

from config import *

FIGDIR = os.path.join(DIR_FIG, "q2")
TXT = "q2_负载周周期性.txt"

A1 = att1_arrays(load_att1())
pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
dates, L, G = load_att2()
D, T = L.shape
HOURS = np.arange(T) / 6.0 + 1 / 12.0          # 各时段中点（小时）
DOW = np.array([d.dayofweek for d in dates])   # 0=周一 … 6=周日
CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# =====================================================================
# 一、统计量
# =====================================================================
rule("【1】一周内各天的平均曲线（逐槽，kW）")
log(f"{'星期':<6s} {'日均值':>9s} {'日最小':>9s} {'日最大':>9s} {'峰谷差':>9s} {'与周一相关性':>12s}")
prof = np.vstack([L[DOW == k].mean(axis=0) for k in range(7)])
for k in range(7):
    r = float(np.corrcoef(L[DOW == k].ravel(), np.tile(prof[0], (DOW == k).sum()))[0, 1])
    log(f"{CN[k]:<6s} {prof[k].mean():>9.1f} {prof[k].min():>9.1f} {prof[k].max():>9.1f}"
        f" {np.ptp(prof[k]):>9.1f} {r:>12.4f}")

rule("【2】曲线自相关（逐槽 Pearson 相关，按天平均）")
lo, hi = 21, D
cor = {}
for k in range(1, 15):
    a, b = L[lo - k:D - k], L[lo:D]
    cor[k] = float(np.mean([np.corrcoef(a[:, t], b[:, t])[0, 1] for t in range(T)]))
    log(f"  滞后 {k:2d} 天：  r = {cor[k]:.4f}")
best = max(cor, key=cor.get)
log(f"  最优滞后 = {best} 天（r = {cor[best]:.4f}）；lag1 = {cor[1]:.4f}，"
    f"lag7 比 lag1 高 {cor[7] - cor[1]:+.4f}")

rule("【3】同星期离散度 vs 相邻日离散度")
band = {k: float(L[DOW == k].std(axis=0).mean()) for k in range(7)}
for lab, lag in (("相邻日 (lag=1)", 1), ("上周同一天 (lag=7)", 7), ("上上周 (lag=14)", 14)):
    a, b = L[lo - lag:D - lag], L[lo:D]
    d = np.abs(a - b)
    log(f"  {lab:<22s} 平均绝对偏差 {d.mean():>8.1f} kW   最大 {d.max():>8.1f} kW")
log("  同星期内部的逐槽标准差（均值）："
    + "  ".join(f"{CN[k]} {band[k]:.0f}" for k in range(7)))
low = [k for k in range(7) if prof[k].mean() < prof.mean()]
log(f"  周内模式：低负载日为 " + "、".join(CN[k] for k in low)
    + f"（日均 {np.mean([prof[k].mean() for k in low]):.0f} kW），"
    + "其余 " + "、".join(CN[k] for k in range(7) if k not in low)
    + f"（日均 {np.mean([prof[k].mean() for k in range(7) if k not in low]):.0f} kW）")

rule("【4】各类日前预测方法的精度（评估区间：第 21 天之后）")
methods = [
    ("持续性 lag1", lambda: fc_ma(L, 1, L1)),
    ("上周同一天 lag7", lambda: fc_wday(L, 7, L1)),
    ("lag14", lambda: fc_wday(L, 14, L1)),
    ("前3天均值 ma3", lambda: fc_ma(L, 3, L1)),
    ("前7天均值 ma7", lambda: fc_ma(L, 7, L1)),
    ("前7天均值 lag7 混合", lambda: 0.5 * fc_ma(L, 7, L1) + 0.5 * fc_wday(L, 7, L1)),
    ("附件1 基准日", lambda: np.tile(L1, (D, 1))),
]
res = {}
for nm, fn in methods:
    F = fn()
    res[nm] = float(np.abs(F[lo:] - L[lo:]).mean())
log(f"  {'方法':<22s} {'MAE (kW)':>10s} {'相对持久性':>12s}")
ref = res["持续性 lag1"]
for nm, _ in methods:
    log(f"  {nm:<22s} {res[nm]:>10.2f} {res[nm] / ref - 1:>+11.1%}")
log(f"  -> lag7 相对 lag1 降低 {1 - res['上周同一天 lag7'] / ref:.1%}，"
    f"相对附件1 基准日降低 {1 - res['上周同一天 lag7'] / res['附件1 基准日']:.1%}")

# =====================================================================
# 二、绘图（四张独立单图 + 一张 2×2 组合图）
# =====================================================================
setup_plot()

dmean = prof.mean(axis=1)
LOW = np.where(dmean < dmean.mean())[0]          # 低负载日（实测为周五、周六）
HIGH = np.where(dmean >= dmean.mean())[0]
ks = np.arange(1, 22)
rs = np.array([cor[k] if k in cor else np.mean(
    [np.corrcoef(L[lo - k:D - k, t], L[lo:D, t])[0, 1] for t in range(T)]) for k in ks])
names = [m[0] for m in methods]
vals = [res[n] for n in names]


def panel_a(ax):
    """(a) 一周内各天的平均负载曲线"""
    for k in HIGH:
        ax.plot(HOURS, prof[k], lw=1.1, color="0.62", alpha=.8)
    for k in LOW:
        ax.plot(HOURS, prof[k], lw=1.6, color=C_PV, alpha=.95, label=CN[k] + "（低负载日）")
    ax.plot(HOURS, prof[HIGH].mean(axis=0), lw=3.0, color=C_LOAD,
            label="高负载日均值（" + "、".join(CN[k] for k in HIGH) + "）")
    ax.plot(HOURS, prof[LOW].mean(axis=0), lw=3.0, color=C_PRICE,
            label="低负载日均值（" + "、".join(CN[k] for k in LOW) + "）")
    ax.set_title(f"(a) 一周内各天的平均曲线：{len(HIGH)} 天高负载 + {len(LOW)} 天低负载，"
                 f"低负载日低 {abs(prof[LOW].mean() - prof[HIGH].mean()) / prof[HIGH].mean():.0%}",
                 fontsize=11)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("负载 (kW)")
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="upper left", ncol=1, framealpha=.8)


def panel_b(ax):
    """(b) 全年同星期曲线叠加"""
    info = []
    for k, c in ((HIGH[0], C_LOAD), (LOW[0], C_PV)):
        idx = np.where(DOW == k)[0]
        for j in idx:
            ax.plot(HOURS, L[j], lw=.6, color=c, alpha=.13)
        ax.plot(HOURS, L[idx].mean(axis=0), lw=2.6, color=c,
                label=f"{CN[k]}均值（{len(idx)} 天）")
        info.append(float(L[idx].std(axis=0).mean()))
    ax.set_title("(b) 全年同星期曲线叠加：同星期的离散带很窄（±"
                 + f"{np.mean(info):.0f} kW）", fontsize=11)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("负载 (kW)")
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)


def panel_c(ax):
    """(c) 自相关系数"""
    ax.bar(ks, rs, color=[C_PRICE if k % 7 == 0 else "0.72" for k in ks], width=.65)
    for k in (7, 14, 21):
        ax.axvline(k, color=C_PRICE, ls="--", lw=.8, alpha=.5)
    for k in (7, 14):
        ax.annotate(f"lag={k},  r={rs[k - 1]:.3f}", xy=(k, rs[k - 1]),
                    xytext=(k + .8, rs[k - 1] + .06), color=C_PRICE, fontsize=8,
                    arrowprops=dict(arrowstyle="-", color=C_PRICE, lw=.8))
    ax.set_title("(c) 曲线自相关系数：在 7 天倍数处出现尖峰", fontsize=11)
    ax.set_xlabel("滞后天数（天）")
    ax.set_ylabel("逐槽相关系数 r")
    ax.set_xticks(ks[::2])
    ax.set_ylim(min(-.3, rs.min() - .05), 1.08)
    ax.grid(alpha=.3, axis="y")


def panel_d(ax):
    """(d) 各类预测方法的精度对比"""
    cols = [C_NET if n == "上周同一天 lag7" else ("0.72" if "lag1" not in n else C_PRICE)
            for n in names]
    bars = ax.barh(names[::-1], vals[::-1], color=cols[::-1])
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 6, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=8)
    ax.set_title("(d) 各类日前预测方法的逐槽 MAE：lag=7 最优", fontsize=11)
    ax.set_xlabel("MAE (kW)")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(alpha=.3, axis="x")


PANELS = [("figA_周内各天平均曲线", panel_a), ("figB_同星期曲线叠加", panel_b),
          ("figC_自相关系数", panel_c), ("figD_预测方法精度对比", panel_d)]

# ---------- 四张独立单图（论文按需单张插入）----------
for nm, fn in PANELS:
    f, a = plt.subplots(figsize=(6.8, 4.4))
    fn(a)
    f.tight_layout()
    save_fig(f, nm + ".png", FIGDIR)
    plt.close(f)

# ---------- 一张 2×2 组合图（便于一眼看全）----------
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
for a, (_nm, fn) in zip(axes.ravel(), PANELS):
    fn(a)
fig.suptitle("小区负载的周周期性 —— 支撑第二问「用上周同一天预测负载」的合理性", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .965))
save_fig(fig, "fig_负载周周期性.png", FIGDIR)
plt.close(fig)

write_report(TXT)
