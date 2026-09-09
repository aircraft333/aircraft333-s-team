# -*- coding: utf-8 -*-
"""2023 C 题 问题一：品类/单品销售量的分布规律与关联关系分析

依据：附件1（6 品类 251 单品信息）+ 附件2（2020-07-01~2023-06-30 销售流水）
问题一要求：合并统计相关数据，分别分析蔬菜各品类、蔬菜单品销售量的
    ① 分布规律（描述统计 + 时间/季节趋势可视化）
    ② 相互关系（品类间日销量 Spearman 相关；单品按销量特征 K-means 分档）
    判断不同品类或不同单品之间是否存在一定的关联关系。

输出：控制台统计 + figures_2023C_q1/*.png + q1_2023C_结果汇总.txt
"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 只存图不弹窗
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
FIG = os.path.join(BASE, "figures_2023C_q1")
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

LINES = []  # 汇总输出


def log(s=""):
    print(s)
    LINES.append(str(s))


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"[已保存] {name}")


def clean_cols(df):
    df.columns = [str(c).replace(" ", "").replace("\u3000", "") for c in df.columns]
    return df


# ---------------------------------------------------------------- 读取与清洗
a1 = clean_cols(pd.read_excel(os.path.join(DATA, "附件1.xlsx")))
a2 = clean_cols(pd.read_excel(os.path.join(DATA, "附件2.xlsx"),
                              parse_dates=["销售日期"]))

# 剔除退货/无效记录：销售类型非"销售"（即 461 条退货，销量<=0）
n_raw = len(a2)
a2 = a2[(a2["销售类型"] == "销售") & (a2["销量(千克)"] > 0)]
# 与附件1 合并得到分类（left join 均可命中）
a2 = a2.merge(a1[["单品编码", "分类名称", "单品名称"]], on="单品编码", how="left")
a2["销售金额"] = a2["销量(千克)"] * a2["销售单价(元/千克)"]
a2["年份"] = a2["销售日期"].dt.year
a2["月份"] = a2["销售日期"].dt.month
a2["星期"] = a2["销售日期"].dt.dayofweek  # 0=周一

log("=" * 70)
log("2023 C 题 问题一 数据分析")
log("=" * 70)
log(f"附件2 原始记录 {n_raw:,} 条，剔除退货后保留 {len(a2):,} 条"
    f"（销量>0 且类型=销售）")
log(f"覆盖时间：{a2['销售日期'].min().date()} ~ {a2['销售日期'].max().date()} "
    f"共 {a2['销售日期'].nunique()} 个销售日")
log(f"单品数：附件1 {len(a1)} 个 / 实际有销售 {a2['单品编码'].nunique()} 个；"
    f"三年零销售单品 {len(a1)-a2['单品编码'].nunique()} 个")
log(f"打折销售占比：{100*(a2['是否打折销售']=='是').mean():.2f}%")

# ------------------------------------------------------------ 品类层汇总
cat_stat = a2.groupby("分类名称").agg(
    销量_kg=("销量(千克)", "sum"),
    销售额_元=("销售金额", "sum"),
    笔数=("销售日期", "size"),
    销售天数=("销售日期", "nunique"),
).sort_values("销量_kg", ascending=False)
cat_stat["销量占比%"] = 100 * cat_stat["销量_kg"] / cat_stat["销量_kg"].sum()
cat_stat["均价_元每kg"] = cat_stat["销售额_元"] / cat_stat["销量_kg"]
log("\n---- 品类三年汇总（按总销量排序）----")
log(cat_stat.round(2).to_string())
# 品种丰富度
rich = a1.groupby("分类名称").size().rename("单品数")
rich = rich.reindex(cat_stat.index)

# ------------------------------------------------------------ 日销量序列
idx_all = pd.date_range("2020-07-01", "2023-06-30", freq="D")
daily = (a2.groupby(["销售日期", "分类名称"])["销量(千克)"].sum()
           .unstack(fill_value=0).reindex(idx_all, fill_value=0))
daily.columns.name = None

# 各品类日销量描述统计
desc = daily.describe().T[["mean", "std", "min", "50%", "max"]]
desc["偏度"] = daily.skew()
desc["峰度"] = daily.kurt()
log("\n---- 品类日销量(kg/天)描述统计（三年 1095 天）----")
log(desc.round(3).to_string())

# ==================================================== A1 品类分布可视化
# 图1：品类总销量与销售额
fig, ax1 = plt.subplots(figsize=(10, 5.2))
x = np.arange(len(cat_stat))
w = 0.38
b1 = ax1.bar(x - w/2, cat_stat["销量_kg"]/1e4, w, color="#4C72B0", label="总销量(万kg)")
ax1.set_xticks(x, cat_stat.index)
ax1.set_ylabel("总销量(万kg)")
ax2 = ax1.twinx()
b2 = ax2.bar(x + w/2, cat_stat["销售额_元"]/1e4, w, color="#DD8452", label="总销售额(万元)")
ax2.set_ylabel("总销售额(万元)")
for i, (v1, v2) in enumerate(zip(cat_stat["销量_kg"]/1e4, cat_stat["销售额_元"]/1e4)):
    ax1.text(i - w/2, v1, f"{v1:.1f}", ha="center", va="bottom", fontsize=9)
    ax2.text(i + w/2, v2, f"{v2:.1f}", ha="center", va="bottom", fontsize=9)
ax1.set_title("六大品类三年总销量与总销售额（2020.07~2023.06）")
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.98), frameon=False)
save(fig, "fig1_品类销量销售额.png")

# 图2：品类销量占比 + 品种丰富度
fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 5))
axa.pie(cat_stat["销量_kg"], labels=cat_stat.index, autopct="%.1f%%",
        startangle=90, counterclock=False,
        colors=plt.cm.tab10(np.arange(len(cat_stat))))
axa.set_title("品类销量占比")
axb.barh(rich.index[::-1], rich.values[::-1], color="#55A868")
axb.set_title("品类品种丰富度（单品数）")
for i, v in enumerate(rich.values[::-1]):
    axb.text(v, i, f" {v}", va="center")
save(fig, "fig2_品类销量占比与丰富度.png")

# 图3：月度季节性（12 月均值）
month_mean = daily.groupby(daily.index.month).mean()
fig, ax = plt.subplots(figsize=(11, 5.2))
for c in daily.columns:
    ax.plot(range(1, 13), month_mean[c], marker="o", ms=4, label=c)
ax.set_xticks(range(1, 13), [f"{m}月" for m in range(1, 13)])
ax.set_xlabel("月份")
ax.set_ylabel("日销量均值(kg)")
ax.set_title("品类日销量的月度季节性（三年平均）")
ax.grid(alpha=0.3)
ax.legend(ncol=3, fontsize=9)
save(fig, "fig3_月度季节性.png")
log("\n月度日销量均值(kg/天，三年平均，含无销售日=0):")
log(month_mean.round(1).to_string())

# 图4：按月总销量堆叠面积（年趋势）
m_total = daily.resample("MS").sum()
fig, ax = plt.subplots(figsize=(12, 5.2))
ax.stackplot(m_total.index, [m_total[c].values for c in daily.columns],
             labels=daily.columns, alpha=0.9)
ax.set_ylabel("月销量(kg)")
ax.set_title("逐月销量堆叠趋势（可见年度周期与品类结构变化）")
ax.legend(ncol=3, fontsize=9, loc="upper left")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate(rotation=30)
save(fig, "fig4_月度堆叠趋势.png")

# 图5：星期规律
wk = a2.groupby("星期").agg(销量=("销量(千克)", "sum"),
                            金额=("销售金额", "sum"))
wk_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar(wk_names, wk["销量"], color="#8172B3")
ax.set_ylabel("总销量(kg)")
ax.set_title("一周内销量的星期规律（三年合计）")
for i, v in enumerate(wk["销量"]):
    ax.text(i, v, f"{v/1e4:.1f}万", ha="center", va="bottom", fontsize=9)
save(fig, "fig5_星期规律.png")
log("\n星期销量合计(kg)与平均单笔金额(元/笔):")
wk2 = a2.groupby("星期").agg(销量kg=("销量(千克)", "sum"),
                             单笔金额元=("销售金额", "mean"))
log(pd.concat([wk, wk2["单笔金额元"]], axis=1).round(2).to_string())
# 图6：各品类 Top3 单品
top3 = (a2.groupby(["分类名称", "单品名称"])["销量(千克)"].sum()
          .reset_index().sort_values(["分类名称", "销量(千克)"], ascending=[True, False])
          .groupby("分类名称").head(3))
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, cat in zip(axes.ravel(), cat_stat.index):
    sub = top3[top3["分类名称"] == cat]
    ax.barh(sub["单品名称"][::-1], sub["销量(千克)"][::-1]/1e3, color="#C44E52")
    ax.set_title(f"{cat} Top3（千kg）", fontsize=10)
    ax.tick_params(labelsize=8)
fig.suptitle("各品类销量最高的 3 个单品（三年）")
fig.tight_layout(rect=[0, 0, 1, 0.96])
save(fig, "fig6_各品类Top3单品.png")

# ==================================================== A2 单品分布可视化
item = (a2.groupby(["单品编码", "单品名称"])["销量(千克)"].sum()
          .reset_index().sort_values("销量(千克)", ascending=False).reset_index(drop=True))
item["累计占比%"] = 100 * item["销量(千克)"].cumsum() / item["销量(千克)"].sum()
log("\n---- 单品销量 Top15 ----")
log(item.head(15).round(1).to_string())
gini_like = item["销量(千克)"][:int(np.ceil(len(item)*0.2))].sum()/item["销量(千克)"].sum()
log(f"\n销量最高的前 20% 单品（约 {int(np.ceil(len(item)*0.2))} 个）贡献了 "
    f"{100*gini_like:.1f}% 的总销量（长尾集中度）")

fig, ax = plt.subplots(figsize=(11, 5.4))
t20 = item.head(20)
ax.bar(t20["单品名称"], t20["销量(千克)"]/1e3, color="#4C72B0")
ax.set_ylabel("总销量(千kg)")
ax.set_xticks(range(len(t20)))
ax.set_xticklabels(t20["单品名称"], rotation=45, ha="right", fontsize=8)
ax.set_title("单品总销量 Top20")
save(fig, "fig7_单品Top20.png")

fig, ax = plt.subplots(figsize=(10, 5.2))
ax.bar(item.index + 1, item["累计占比%"], color="#DD8452", alpha=0.85)
ax.plot(item.index + 1, item["累计占比%"], color="k", lw=1)
ax.axhline(80, ls="--", color="grey")
ax.set_xlabel("按销量降序的单品序号")
ax.set_ylabel("累计销量占比(%)")
ax.set_title("单品销量累计占比曲线（前多少单品贡献 80%）")
n80 = (item["累计占比%"] <= 80).sum()
ax.axvline(n80, ls="--", color="grey")
ax.text(n80 + 3, 30, f"约 {n80} 个单品贡献 80%", fontsize=9)
save(fig, "fig8_单品累计贡献.png")

# ==================================================== B1 品类间相关
corr = daily.corr(method="spearman")
log("\n---- 品类日销量的 Spearman 相关矩阵 ----")
log(corr.round(3).to_string())
pvals = None
try:
    from scipy.stats import spearmanr
    pvals = pd.DataFrame(index=corr.index, columns=corr.columns, dtype=float)
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            r, p = spearmanr(daily[corr.columns[i]], daily[corr.columns[j]])
            pvals.iloc[i, j] = p
    log("\n品类间 Spearman 相关显著性 p 值（上三角）:")
    log(pvals.round(4).to_string())
except Exception as e:
    log(f"(scipy 不可用，跳过 p 值: {e})")

mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns)
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        if not mask[i, j]:
            lab = f"{corr.values[i, j]:.2f}"
            if pvals is not None and pvals.iloc[i, j] < 0.001:
                lab += "***"
            ax.text(j, i, lab, ha="center", va="center", fontsize=10)
ax.set_title("品类日销量 Spearman 相关热力图\n(***: p<0.001)")
fig.colorbar(im, shrink=0.85)
save(fig, "fig9_品类相关热力图.png")

# ==================================================== B2 单品 K-means 分档
d_item = (a2.groupby(["单品编码", "销售日期"])["销量(千克)"].sum()
            .groupby("单品编码").agg(["sum", "max", "count"]))
d_item.columns = ["total", "max_daily", "n_days"]
d_item["avg_daily"] = d_item["total"] / d_item["n_days"]
X = d_item[["total", "max_daily", "avg_daily"]].copy()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
Xs = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=4, random_state=0, n_init=10).fit(Xs)
X["cluster"] = km.labels_
# 按总销量均值给簇排序命名（0 为最高档）
order = X.groupby("cluster")["total"].mean().sort_values(ascending=False).index
names = {c: n for c, n in zip(order, ["热销", "畅销", "平销", "滞销"])}
X["档位"] = X["cluster"].map(names)
prof = X.groupby("cluster").agg(
    单品数=("total", "size"),
    平均总销量kg=("total", "mean"),
    平均单日最大kg=("max_daily", "mean"),
    平均日均销量kg=("avg_daily", "mean"),
).reindex(order)
prof["档位"] = [names[c] for c in prof.index]
log("\n---- 单品销量 K-means 分档（4 类，指标已标准化）----")
log(prof.round(2).to_string())
for c in order:
    toplist = X[X["cluster"] == c].index
    names_map = a2.drop_duplicates("单品编码").set_index("单品编码")["单品名称"]
    ex = [f"{names_map.loc[i]}" for i in toplist[:3]]
    log(f"  {names[c]}({len(toplist)}个) 示例: " + "、".join(ex))

# 图10：聚类散点（总销量 vs 日均销量）
pal = {"热销": "#D62728", "畅销": "#FF7F0E", "平销": "#1F77B4", "滞销": "#7F7F7F"}
fig, ax = plt.subplots(figsize=(9.5, 6))
for k, g in X.groupby("档位"):
    ax.scatter(g["total"], g["avg_daily"], s=40, alpha=0.75,
               color=pal[k], label=f"{k}({len(g)})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("三年总销量(kg, log)"); ax.set_ylabel("有销售日日均销量(kg, log)")
ax.set_title("单品按销量规模 K-means 聚类（总销量×日均销量）")
ax.legend()
save(fig, "fig10_单品聚类散点.png")

# 图11：分档画像（每档均值归一化）
fig, ax = plt.subplots(figsize=(9.5, 5.2))
met = ["total", "max_daily", "avg_daily"]
met_cn = ["总销量", "单日最大销量", "日均销量"]
for m, lab in zip(met, met_cn):
    mm = X.groupby("档位")[m].mean() / X.groupby("档位")[m].mean().max()
    ax.plot([names[c] for c in order], [mm[names[c]] for c in order],
            marker="o", label=lab)
ax.set_ylabel("组均值归一化(0~1)")
ax.set_title("四档单品的销量特征画像")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "fig11_分档画像.png")

# ==================================================== 汇总写盘
summary = "\n".join(LINES)
with open(os.path.join(BASE, "q1_2023C_结果汇总.txt"), "w", encoding="utf-8") as f:
    f.write(summary)
log("\n全部完成，图表已存至 figures_2023C_q1/")
