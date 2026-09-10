# -*- coding: utf-8 -*-
"""
2026 高教社杯 C 题 · 问题一：附件 1 数据探索与可视化
=====================================================================
数据：附件1.xlsx = 某天的电价(元/kWh)、小区负载(kW)、光伏发电预测功率(kW)，10 分钟粒度

时间口径（重要，务必与 result1.xlsx 模板保持一致）：
    附件1 的时间戳按「时段右端点」解释，第 k 行时间 T_k 对应区间 [T_k-10min, T_k]：
        第 1   行 00:10      -> [00:00, 00:10]
        最后 1 行 0:00+1     -> [23:50, 24:00]   （0:00+1 表示当天 24:00）
    这样 144 段恰好铺满 [00:00, 24:00]，与题目「储能 0:00 与 24:00 储电量相同」对得上。

输出：figures/q1/*.png（论文插图）+ q1_数据分析结果.txt（数据事实汇总）
所有常量与工具函数集中在 config.py，本脚本只负责分析与作图。
"""
from config import *

setup_plot()

# ==================== 1. 读取与解析 ====================
df = load_att1()
A = att1_arrays(df)
minutes, hours = A["minutes"], A["hours"]
price, load, power, net = A["price"], A["load"], A["pv"], A["net"]

rule("【1】数据概况")
log(f"时段数 = {len(df)}（每段 10 min，合计 {len(df) * DT_H:.1f} h）；"
    f"缺失值合计 = {int(df[['电价', '小区负载', '光伏发电预测功率']].isna().sum().sum())}")

stat_line("电价", price, "元/kWh")
stat_line("小区负载", load, "kW")
stat_line("光伏预测", power, "kW")
stat_line("净负荷", net, "kW")

E_load = load.sum() * DT_H                           # 全天负载电量 kWh
E_pv = power.sum() * DT_H                            # 全天光伏电量 kWh
E_self = np.minimum(load, power).sum() * DT_H         # 光伏被本地消纳部分
E_surplus = np.maximum(power - load, 0).sum() * DT_H  # 光伏富余（负载吃不下）
E_gap = np.maximum(load - power, 0).sum() * DT_H      # 净负荷缺口（需外网购电）
log(f"\n全天负载电量 = {E_load:,.1f} kWh   全天光伏电量 = {E_pv:,.1f} kWh   "
    f"光伏渗透率 = {E_pv / E_load:.1%}")
log(f"光伏自消纳电量 = {E_self:,.1f} kWh（自消纳率 {E_self / E_pv:.1%}）；"
    f"富余电量 = {E_surplus:,.1f} kWh；净负荷缺口 = {E_gap:,.1f} kWh")
log(f"净负荷 > 0 的时段 = {int((net > 0).sum())}/{len(df)}；"
    f"光伏富余时段 = {int((net < 0).sum())}/{len(df)}")
log(f"若不做储能调度：全天购电量 ≈ {E_gap:,.1f} kWh，"
    f"按均价 {price.mean():.4f} 元/kWh 估算购电费 ≈ {E_gap * price.mean():,.0f} 元")

# ==================== 2. 峰谷平时段划分 ====================
tier, q_lo, q_hi = classify_tiers(price)
segs = label_segments(minutes, tier)
log("")
rule("【2】峰谷平时段划分（按电价 33% / 67% 分位数，数据驱动）")
log(f"谷价阈值 ≤ {q_lo:.4f} 元/kWh；峰价阈值 ≥ {q_hi:.4f} 元/kWh；"
    f"峰谷价差 = {price.max() - price.min():.4f} 元/kWh，峰谷比 = {price.max() / price.min():.2f}")
for s, e, lab in segs:
    if lab == TIER_F:
        continue
    mask = (minutes > s) & (minutes <= e)
    log(f"  [{lab}] {fmt(s)}-{fmt(e)}  时长 {mask.sum() * DT_H:5.1f} h  "
        f"均价 {price[mask].mean():.4f}  平均负载 {load[mask].mean():7.1f} kW  "
        f"平均光伏 {power[mask].mean():7.1f} kW")
for lab in (TIER_V, TIER_F, TIER_P):
    mask = tier == lab
    log(f"  {lab}档合计: {mask.sum() * DT_H:5.1f} h，均价 {price[mask].mean():.4f}，"
        f"负载均值 {load[mask].mean():7.1f} kW，光伏均值 {power[mask].mean():7.1f} kW")

# ---------------- 3. 逐小时统计 ----------------
df["小时序号"] = (df["分钟"] - 1) // 60                 # 10:00 -> 0 号时段组(0:00-1:00)
hourly = df.groupby("小时序号").agg(
    电价=("电价", "mean"), 负载=("小区负载", "mean"), 光伏=("光伏发电预测功率", "mean")
)
hourly["净负荷"] = hourly["负载"] - hourly["光伏"]
log("")
rule("【3】逐小时均值（0 表示 00:00-01:00，23 表示 23:00-24:00）")
for h, row in hourly.iterrows():
    log(f"  {h:02d}:00-{h + 1:02d}:00   电价 {row['电价']:.4f}  负载 {row['负载']:7.1f} kW  "
        f"光伏 {row['光伏']:7.1f} kW  净负荷 {row['净负荷']:7.1f} kW")
peak_h = hourly["电价"].idxmax()
valley_h = hourly["电价"].idxmin()
log(f"  → 电价最高的小时：{peak_h:02d}:00-{peak_h + 1:02d}:00（{hourly['电价'].max():.4f} 元/kWh）")
log(f"  → 电价最低的小时：{valley_h:02d}:00-{valley_h + 1:02d}:00（{hourly['电价'].min():.4f} 元/kWh）")

# ---------------- 4. 储能套利空间（用于问题一建模前的可行性判断） ----------------
# 往返效率 eta^2：峰段多放 r kWh，谷段需多买 r/eta^2 kWh
eta2 = ETA ** 2
arb = price.max() - price.min() / eta2
log("")
rule("【4】储能套利空间（附录1：容量 12000 kWh，SOC 1200~10800，功率 5000 kW，效率 90%）")
usable = SOC_MAX - SOC_MIN                                    # 一天内可吞吐的电量上界
seg_max = SLOT_MAX                                            # 单时段最大充/放电量 kWh
v_h = (tier == TIER_V).sum() * DT_H
p_h = (tier == TIER_P).sum() * DT_H
move = min(usable, v_h / DT_H * seg_max * ETA, p_h / DT_H * seg_max)
log(f"单时段最大充/放电量 = {seg_max:.1f} kWh；谷段总时长 = {v_h:.1f} h；峰段总时长 = {p_h:.1f} h")
log(f"受容量约束，全天可搬移电量上界 ≈ {usable:,.0f} kWh；受功率/时长约束后 ≈ {move:,.0f} kWh")
log(f"套利条件：峰谷价差 > 谷价×(1/η²-1)，即 峰价 > 谷价/η² = {price.min() / eta2:.4f}；"
    f"实际峰价 {price.max():.4f} → 套利空间 {arb:.4f} 元/kWh 放电量")
log(f"→ 仅靠储能套利，全天最多可省 ≈ {move * arb:,.0f} 元（未计光伏富余时段的『免费充电』）")

# ---------------- 5. 相关系数 ----------------
log("")
rule("【5】Pearson 相关系数")
cols = {"电价": price, "小区负载": load, "光伏预测": power, "净负荷": net}
names = list(cols)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = cols[names[i]], cols[names[j]]
        r = np.corrcoef(a, b)[0, 1]
        p = sps.pearsonr(a, b)[1]
        star = "*" if p < 0.05 else ""
        log(f"  {names[i]} ~ {names[j]:<8s} r = {r:+.3f}{star}   p = {p:.3g}")

# ==================== 6. 绘图 ====================
# --- 图1：电价 / 负载 / 光伏 日内曲线 ---
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
axes[0].plot(hours, price, color=C_PRICE, lw=1.6)
axes[0].axhline(price.mean(), ls="--", c="gray", lw=1, label=f"均值 {price.mean():.4f}")
axes[0].set_ylabel("电价 (元/kWh)")
axes[0].set_title("图1  附件1 日内曲线：电价 / 小区负载 / 光伏发电预测功率", fontweight="bold")
axes[0].legend()

axes[1].plot(hours, load, color=C_LOAD, lw=1.6)
axes[1].axhline(load.mean(), ls="--", c="gray", lw=1, label=f"均值 {load.mean():.0f} kW")
axes[1].set_ylabel("小区负载 (kW)")
axes[1].legend()

axes[2].plot(hours, power, color=C_PV, lw=1.6)
axes[2].fill_between(hours, 0, power, color=C_PV, alpha=0.15)
axes[2].set_ylabel("光伏功率 (kW)")
axes[2].set_xlabel("时刻")
for ax in axes:
    shade_tiers(ax, segs)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0)
tier_legend(axes[0])
axes[0].legend()
fig.tight_layout()
save_fig(fig, "fig1_日内曲线_电价_负载_光伏.png")

# --- 图2：净负荷曲线 ---
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(hours, net, color="k", lw=1.6, label="净负荷 = 负载 − 光伏")
ax.fill_between(hours, 0, net, where=net > 0, color=C_PRICE, alpha=0.25, label="需外网购电")
ax.fill_between(hours, 0, net, where=net <= 0, color=C_NET, alpha=0.35, label="光伏富余(可充电/弃光)")
ax.axhline(0, c="k", lw=0.8)
shade_tiers(ax, segs)
ax.set_xlabel("时刻")
ax.set_ylabel("净负荷 (kW)")
ax.set_title("图2  净负荷曲线与光伏富余时段（背景色：红=峰价段，绿=谷价段）", fontweight="bold")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
ax.margins(x=0)
fig.tight_layout()
save_fig(fig, "fig2_净负荷曲线.png")

# --- 图3：逐小时均值 ---
fig, ax = plt.subplots(figsize=(12, 4.5))
x = np.arange(24)
ax.bar(x, hourly["电价"], color=C_PRICE, alpha=0.65, label="电价(均值)")
ax.set_xlabel("小时")
ax.set_ylabel("电价 (元/kWh)", color=C_PRICE)
ax.tick_params(axis="y", labelcolor=C_PRICE)
ax.set_xticks(x)
ax.set_xticklabels([f"{h:02d}" for h in x])
ax2 = ax.twinx()
ax2.plot(x, hourly["负载"], "o-", color=C_LOAD, label="负载")
ax2.plot(x, hourly["光伏"], "s-", color=C_PV, label="光伏")
ax2.plot(x, hourly["净负荷"], "^--", color=C_NET, label="净负荷", lw=1)
ax2.set_ylabel("功率 (kW)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left")
ax.set_title("图3  逐小时均值：电价（柱）与负载/光伏/净负荷（线）", fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
save_fig(fig, "fig3_逐小时均值.png")

# --- 图4：相关性热力图 ---
arr = np.column_stack([cols[n] for n in names])
n = len(names)
R = np.corrcoef(arr.T)
P = np.full((n, n), np.nan)
for i in range(n):
    for j in range(n):
        P[i, j] = sps.pearsonr(arr[:, i], arr[:, j])[1]

fig, ax = plt.subplots(figsize=(6.4, 5.4))
im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(n), names)
ax.set_yticks(range(n), names)
for i in range(n):
    for j in range(n):
        star = ""
        if not np.isnan(P[i, j]):
            star = "***" if P[i, j] < 0.001 else ("**" if P[i, j] < 0.01 else ("*" if P[i, j] < 0.05 else ""))
        ax.text(j, i, f"{R[i, j]:.2f}{star}", ha="center", va="center",
                color="white" if abs(R[i, j]) > 0.6 else "black", fontsize=11)
ax.set_title("图4  Pearson 相关性热力图（* p<0.05, ** p<0.01, *** p<0.001）", fontweight="bold")
fig.colorbar(im, ax=ax, shrink=0.85)
fig.tight_layout()
save_fig(fig, "fig4_相关性热力图.png")

# --- 图5：电价持续曲线 ---
fig, ax = plt.subplots(figsize=(12, 4.5))
ps = np.sort(price)[::-1]
ax.plot(np.arange(1, len(ps) + 1) * DT_H, ps, color=C_PRICE, lw=2, label="电价持续曲线")
ax.fill_between(np.arange(1, len(ps) + 1) * DT_H, 0, ps, color=C_PRICE, alpha=0.12)
ax.axhline(price.min() / eta2, ls="--", c="#2ca02c", lw=1.4,
           label=f"套利门槛 谷价/η² = {price.min() / eta2:.4f}")
ax.axhline(price.mean(), ls=":", c="gray", lw=1.2, label=f"均价 {price.mean():.4f}")
ax.annotate(f"最高 {price.max():.4f}", xy=(1, price.max()), xytext=(8, price.max()),
            arrowprops=dict(arrowstyle="->", color="k"), fontsize=9)
ax.annotate(f"最低 {price.min():.4f}", xy=(144, price.min()), xytext=(120, price.min() + 0.12),
            arrowprops=dict(arrowstyle="->", color="k"), fontsize=9)
ax.set_xlabel("累计时长 (h)")
ax.set_ylabel("电价 (元/kWh)")
ax.set_title("图5  电价持续时长曲线（峰谷价差 = 储能套利空间）", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
ax.margins(x=0)
fig.tight_layout()
save_fig(fig, "fig5_电价持续曲线.png")

# --- 图6：光伏消纳结构 ---
fig, ax = plt.subplots(figsize=(12, 4.5))
self_use = np.minimum(load, power)
ax.plot(hours, load, color=C_LOAD, lw=1.8, label="小区负载")
ax.plot(hours, power, color=C_PV, lw=1.8, label="光伏预测功率")
ax.fill_between(hours, 0, self_use, color=C_PV, alpha=0.35, label=f"光伏自用 {E_self:,.0f} kWh")
ax.fill_between(hours, self_use, power, where=power > self_use, color=C_NET, alpha=0.35,
                label=f"光伏富余 {E_surplus:,.0f} kWh")
ax.fill_between(hours, power, load, where=load > power, color=C_PRICE, alpha=0.25,
                label=f"净负荷缺口 {E_gap:,.0f} kWh")
ax.set_xlabel("时刻")
ax.set_ylabel("功率 (kW)")
ax.set_title("图6  光伏消纳结构与净负荷缺口", fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
ax.margins(x=0)
fig.tight_layout()
save_fig(fig, "fig6_光伏消纳结构.png")

# ==================== 7. 写结果文件 ====================
write_report(TXT_Q1)
print(f"图片已保存到 {FIG_Q1}/")

