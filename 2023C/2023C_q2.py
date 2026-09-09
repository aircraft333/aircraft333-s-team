# -*- coding: utf-8 -*-
"""2023 C 题 问题二：品类成本加成定价与销量关系 + 未来一周(7/1~7/7)补货定价优化

数据：附件1(单品/分类) 附件2(销售流水) 附件3(批发价) 附件4(损耗率)
口径（与优秀论文可对照）：
  单品批发价 C；品类平均损耗率 L；单位成本 B = C×(1+L)
  成本加成定价 CPP = 单位成本 × (1+加成率)；加成率口径①隐含λ=加权售价/加权成本-1，
    口径②行业典型 30%
第一小问：各品类日销量与成本加成定价的相关性(Pearson)+线性回归 CPP=β0+β1·销量
第二小问：
  1) 近半年(181天)品类日销量 7 天周期指数平滑 → 预测 2023-07-01~07-07 基准销量 S0
  2) 把第一小问回归视为反需求线并平移过基准点(P0,S0)：D(P)=S0+(P-P0)/β1，β1<0
  3) 逐品类逐日最大化利润 (P-B)·D(P) → 最优定价 P*∈[1.1B,1.5B]（价格越高销量越低）
  4) 补货量 R = D/(1-L)（含损耗补齐），收益 π = D·(P*-B)
输出：控制台 + figures_2023C_q2/*.png + q2_2023C_结果汇总.txt
"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as _st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
FIG = os.path.join(BASE, "figures_2023C_q2")
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

LINES = []


def log(s=""):
    print(s)
    LINES.append(str(s))


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log(f"[已保存] {name}")


def clean_cols(df):
    df.columns = [str(c).replace(" ", "").replace("\u3000", "") for c in df.columns]
    return df


# ---------------------------------------------------------------- 读数据
a1 = clean_cols(pd.read_excel(os.path.join(DATA, "附件1.xlsx")))
a2 = clean_cols(pd.read_excel(os.path.join(DATA, "附件2.xlsx"),
                              parse_dates=["销售日期"]))
a3 = clean_cols(pd.read_excel(os.path.join(DATA, "附件3.xlsx"),
                              parse_dates=["日期"]))
loss_cat_raw = clean_cols(pd.read_excel(os.path.join(DATA, "附件4.xlsx"),
                                        sheet_name=0))
loss_item_raw = clean_cols(pd.read_excel(os.path.join(DATA, "附件4.xlsx"),
                                         sheet_name=1))

loss_cat_col = [c for c in loss_cat_raw.columns if "平均损耗率" in c][0]
lmap = dict(zip(loss_cat_raw["小分类名称"],
                loss_cat_raw[loss_cat_col].astype(float) / 100.0))
loss_item_col = [c for c in loss_item_raw.columns if "损耗率" in c][0]
lmap_item = dict(zip(loss_item_raw["单品编码"],
                     loss_item_raw[loss_item_col].astype(float) / 100.0))
log("品类平均损耗率(%): " + ", ".join(f"{k}{v*100:.1f}" for k, v in lmap.items()))

n_raw = len(a2)
a2 = a2[(a2["销售类型"] == "销售") & (a2["销量(千克)"] > 0)].copy()
a2["销售金额"] = a2["销量(千克)"] * a2["销售单价(元/千克)"]
a2 = a2.merge(a1[["单品编码", "分类名称"]], on="单品编码", how="left")
a2["L"] = a2["分类名称"].map(lmap)

w = a3.rename(columns={"日期": "销售日期"})
s = a2.merge(w, on=["单品编码", "销售日期"], how="left")
log(f"销售 {n_raw:,} -> 有效 {len(s):,}；批发价缺失 {s['批发价格(元/千克)'].isna().sum()}")
s["单位成本"] = s["批发价格(元/千克)"] * (1 + s["L"])
s["批发额"] = s["批发价格(元/千克)"] * s["销量(千克)"]

# ---------------------------------------------------------------- 品类日聚合
g = s.groupby(["分类名称", "销售日期"])
daily = pd.DataFrame({
    "销量": g["销量(千克)"].sum(),
    "金额": g["销售金额"].sum(),
    "批发额": g["批发额"].sum(),
}).reset_index()
daily["售价"] = daily["金额"] / daily["销量"]
daily["批发价"] = daily["批发额"] / daily["销量"]
daily["单位成本"] = daily["批发价"] * (1 + daily["分类名称"].map(lmap))
daily["L"] = daily["分类名称"].map(lmap)

cats = sorted(daily["分类名称"].unique())

# 品类三年参数：销量加权(优化用) 与 简单平均(对照论文) 两种单位成本
rows_param = []
w_cat = w.merge(a1[["单品编码", "分类名称"]], on="单品编码", how="left")
for c in cats:
    d = daily[daily["分类名称"] == c]
    C_w = d["批发额"].sum() / d["销量"].sum()
    B_w = C_w * (1 + d["L"].iloc[0])
    C_s = w_cat[w_cat["分类名称"] == c]["批发价格(元/千克)"].mean()
    B_s = C_s * (1 + d["L"].iloc[0])
    P_w = d["金额"].sum() / d["销量"].sum()
    lam = P_w / C_w - 1.0
    rows_param.append({"品类": c, "加权批发价C": C_w, "简单平均C": C_s,
                       "加权单位成本B(优化用)": B_w, "简单平均B(对照)": B_s,
                       "加权售价P0": P_w, "隐含加成率λ": lam,
                       "平均损耗率L": d["L"].iloc[0]})
param = pd.DataFrame(rows_param).set_index("品类")
param = param.round(4)
log("\n[参数] 品类三年加权与简单平均口径（简单平均可对照优秀论文）")
log(param.to_string())
log("\n[论文对照] 论文单位成本: 花菜7.57 食用菌7.20 花叶5.44 辣椒7.61 茄5.17 水生根茎11.22")

# ---------------------------------------------------------------- 第一小问：销量 vs 成本加成定价
lam_map = param["隐含加成率λ"].to_dict()
FIX_LAM = 0.30
rows_rel = []
for c in cats:
    d = daily[daily["分类名称"] == c].dropna(subset=["销量", "单位成本"]).copy()
    x = d["销量"].values
    cpp_i = d["单位成本"].values * (1 + lam_map[c])   # 隐含加成
    cpp_f = d["单位成本"].values * (1 + FIX_LAM)      # 固定30%
    r_i, p_i = _st.pearsonr(x, cpp_i)
    r_f, p_f = _st.pearsonr(x, cpp_f)
    b1, b0 = np.polyfit(x, cpp_i, 1)
    rows_rel.append({"品类": c, "Pearson_r(隐含加成)": r_i, "p值": p_i,
                     "Pearson_r(固定30%)": r_f, "截距β0": b0, "斜率β1": b1})
df_rel = pd.DataFrame(rows_rel)
log("\n[第一小问] 销量 vs 成本加成定价 CPP=单位成本×(1+加成率)（Pearson + 回归）")
log(df_rel.round(4).to_string())
log("注：隐含加成口径下 CPP≈实际售价(×比例)，回归斜率 β1<0 即“定价越高销量越低”。")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, c in zip(axes.ravel(), cats):
    d = daily[daily["分类名称"] == c].dropna(subset=["销量", "单位成本"])
    d = d.copy()
    d["CPP"] = d["单位成本"] * (1 + lam_map[c])
    ax.scatter(d["销量"], d["CPP"], s=8, alpha=0.35, color="#4C72B0")
    row = df_rel[df_rel["品类"] == c].iloc[0]
    xs = np.linspace(d["销量"].min(), d["销量"].max(), 50)
    ax.plot(xs, row["截距β0"] + row["斜率β1"] * xs, "r-", lw=1.6)
    ax.set_title(f"{c}\nr={row['Pearson_r(隐含加成)']:.3f} (p={row['p值']:.2g})",
                 fontsize=10)
    ax.set_xlabel("日销量(kg)")
    ax.set_ylabel("成本加成定价(元/kg)")
fig.suptitle("第一小问：各品类销售总量 vs 成本加成定价（隐含加成率，散点+拟合）")
fig.tight_layout(rect=[0, 0, 1, 0.96])
save(fig, "fig1_销量与成本加成定价.png")

# ---------------------------------------------------------------- 第二小问：预测 + 优化
pred_dates = pd.date_range("2023-07-01", "2023-07-07", freq="D")


def cat_series(c):
    d = daily[daily["分类名称"] == c].set_index("销售日期")["销量"]
    return d.reindex(pd.date_range("2020-07-01", "2023-06-30", freq="D"),
                     fill_value=0)


# 未来一周基准需求 S0（7 天周期指数平滑）
S0 = {}
for c in cats:
    train = cat_series(c).loc["2023-01-01":"2023-06-30"]
    try:
        fc = ExponentialSmoothing(train, trend=None, seasonal="add",
                                  seasonal_periods=7,
                                  initialization_method="estimated").fit().forecast(7)
        fc = np.asarray(fc, dtype=float)
    except Exception as e:
        log(f"  ETS 失败({c}): {e} -> 用近4周同星期均值")
        fc = np.array([train[train.index.weekday == wd].tail(4).mean()
                       for wd in range(7)])
    S0[c] = np.maximum(fc, 0.05)
S0 = pd.DataFrame(S0, index=pred_dates)

B_map = param["加权单位成本B(优化用)"].to_dict()
P0_map = param["加权售价P0"].to_dict()
L_map = param["平均损耗率L"].to_dict()
Smax_map = {c: cat_series(c).max() for c in cats}

# 需求-价格：常弹性 D(P)=S0·(P0/P)^η (η<0)。
# 对单位成本 B，单期利润 (P-B)·D(P) 的最优价 P*=B·η/(1+η)（需 η<-1），
# 并夹在 [1.1B,1.5B] 之间。η 越大(负得多)最优加成越低。
# 弹性难以精确估计 -> 做敏感性分析；主结果取 η=-4（最优加成 1/3≈33%）。
ETA_MAIN = -4.0
ETA_LIST = [-3.0, -4.0, -6.0]

rows_out = []
sens = []
for eta in ETA_LIST:
    tot = 0.0
    for c in cats:
        c_eff = float(B_map[c])
        P0 = float(P0_map[c])
        lo, hi = 1.1 * c_eff, 1.5 * c_eff
        for dt in pred_dates:
            s0 = float(S0.loc[dt, c])
            p_star = c_eff * eta / (1 + eta) if eta < -1 else hi
            p_star = float(np.clip(p_star, lo, hi))
            D = s0 * (p_star / P0) ** eta if eta != 0 else s0  # 常弹性 η<0
            D = float(np.clip(D, 0.01, float(Smax_map[c])))
            R = min(D / (1 - L_map[c]), float(Smax_map[c]))
            R = max(R, D)
            pi = D * (p_star - c_eff)
            tot += pi
            if eta == ETA_MAIN:
                rows_out.append({"日期": dt, "品类": c, "基准需求S0": s0,
                                 "最优定价P": p_star,
                                 "加成率P/B-1": p_star / c_eff - 1,
                                 "预计销量D": D, "补货量R": R,
                                 "单位成本B": c_eff, "日利润(元)": pi})
    sens.append({"需求价格弹性η": eta, "未来一周总收益(元)": tot})

out = pd.DataFrame(rows_out)
total_profit = sens[[s["需求价格弹性η"] for s in sens].index(ETA_MAIN)]["未来一周总收益(元)"]
out_piv_R = out.pivot_table(index="日期", columns="品类", values="补货量R")
out_piv_P = out.pivot_table(index="日期", columns="品类", values="最优定价P")

log("\n[敏感性] 需求价格弹性的影响（η<-1 才可能降价增利）")
log(pd.DataFrame(sens).round(1).to_string(index=False))
log(f"\n[第二小问][主方案 η={ETA_MAIN:.0f}] 2023-07-01~07-07 日补货量(kg)")
log(out_piv_R.round(1).to_string())
log(f"\n[第二小问][主方案 η={ETA_MAIN:.0f}] 最优定价(元/kg)")
log(out_piv_P.round(2).to_string())
log(f"\n[第二小问] 未来一周最大化收益合计：{total_profit:.2f} 元")
print_log = out.copy()
print_log["日期"] = print_log["日期"].dt.strftime("%Y-%m-%d")
log("\n明细(主方案):")
log(print_log.round(2).to_string(index=False))

out.to_csv(os.path.join(BASE, "q2_补货定价_未来一周.csv"), index=False,
           encoding="utf-8-sig")
param.round(4).to_csv(os.path.join(BASE, "q2_品类参数.csv"), encoding="utf-8-sig")
df_rel.round(4).to_csv(os.path.join(BASE, "q2_量价相关回归.csv"), index=False,
                       encoding="utf-8-sig")
log("\n[导出] q2_补货定价_未来一周.csv / q2_品类参数.csv / q2_量价相关回归.csv")

# 图2：未来一周补货量与定价
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 6))
out_piv_R.plot(marker="o", ax=a)
a.set_title("未来一周各品类日补货量(kg)")
b2 = out_piv_P.plot(marker="o", ax=b)
b.set_title("未来一周各品类定价(元/kg)")
fig.autofmt_xdate()
save(fig, "fig2_未来一周补货量与定价.png")

# 图3：销量预测 vs 历史
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, c in zip(axes.ravel(), cats):
    full = cat_series(c)
    tail = full.loc["2023-04-01":"2023-06-30"]
    ax.plot(tail.index, tail.values, lw=0.8, color="#4C72B0")
    ax.plot(pred_dates, S0[c].values, "o--", color="#C0392B", label="预测S0")
    ax.set_title(c, fontsize=10)
    ax.legend(fontsize=8)
fig.suptitle("未来一周基准需求预测（红：7/1~7/7）")
fig.autofmt_xdate()
fig.tight_layout(rect=[0, 0, 1, 0.96])
save(fig, "fig3_销量预测.png")

with open(os.path.join(BASE, "q2_2023C_结果汇总.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LINES))
log("\n全部完成")
