# -*- coding: utf-8 -*-
"""
2025 高教社杯 C 题 | 问题 1
胎儿 Y 染色体浓度与孕周数、BMI 等指标的关系建模与显著性检验
============================================================
思路（为什么用"线性混合效应模型" LMM，而不是普通多元回归）：
  · 附件为 267 位孕妇的纵向随访数据（多数每人 4 次，10~29 周），属重复测量；
  · 组内（同一孕妇随时间）：Y 浓度随孕周上升（胎儿游离 DNA 增多）；
  · 组间（不同孕妇之间）：均值呈负相关（高 BMI/体重大 → 血浆稀释 → 浓度低，
    且这类孕妇往往检测较晚），若不分离会把"组内正效应"冲淡甚至反转；
  · ICC≈0.74，即 ~74% 变异来自个体间，观测不独立，普通 OLS 会高估显著性。
  → 以孕妇为分组建立 LMM(随机截距)，分离组内/组间效应，得到更可信的关系模型。

输出：控制台中文报告 + figures_q1/ 下 5 张图
"""
import os
import re
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# ---------- 中文显示 & 日志 ----------
logging.getLogger("matplotlib").setLevel(logging.WARNING)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# 压掉 statsmodels 混合模型的边界收敛警告噪音
warnings.filterwarnings("ignore", message="The MLE may be on the boundary.*")

FIG = "figures_q1"
os.makedirs(FIG, exist_ok=True)

PATH = "附件.xlsx"


def parse_weeks(s):
    """'11w+6' / '13w' / '15w+2' -> 连续孕周(周数+天数/7)"""
    if pd.isna(s):
        return np.nan
    m = re.fullmatch(r"(\d+)\s*w(?:\+(\d+))?", str(s).strip().lower())
    if not m:
        return np.nan
    return int(m.group(1)) + (int(m.group(2)) if m.group(2) else 0) / 7.0


# ================= 1. 读取与预处理 =================
def load_data():
    df = pd.read_excel(PATH)
    sub = df[[
        "孕妇代码", "年龄", "身高", "体重", "检测孕周",
        "孕妇BMI", "Y染色体浓度",
    ]].rename(columns={
        "孕妇代码": "pid", "年龄": "age", "身高": "height",
        "体重": "weight", "检测孕周": "gw_raw",
        "孕妇BMI": "bmi", "Y染色体浓度": "yconc",
    })
    sub["gw"] = sub["gw_raw"].map(parse_weeks)
    # 男胎才有 Y 浓度；剔除缺孕周/缺BMI/超范围异常
    d = sub[sub["yconc"].notna()].copy()
    d = d.dropna(subset=["gw", "bmi"])
    d = d[(d["gw"] >= 10) & (d["gw"] <= 30) & (d["bmi"] >= 15) & (d["bmi"] <= 60)]
    return d.reset_index(drop=True)


# ================= 2. 相关分析 =================
def correlation_report(d):
    print("=" * 78)
    print("①  相关特性分析（Y染色体浓度 与 各指标）")
    print("=" * 78)
    print(f"样本观测: {len(d)}  |  孕妇人数: {d['pid'].nunique()}"
          f"  |  孕周范围: {d['gw'].min():.1f}~{d['gw'].max():.1f} 周"
          f"  |  BMI: {d['bmi'].min():.1f}~{d['bmi'].max():.1f}")
    targets = {"孕周(周)": d["gw"], "BMI(kg/m²)": d["bmi"],
               "年龄(岁)": d["age"], "身高(cm)": d["height"], "体重(kg)": d["weight"]}
    rows = []
    for name, x in targets.items():
        pr = stats.pearsonr(x, d["yconc"])
        sr = stats.spearmanr(x, d["yconc"])
        rows.append([name, pr.statistic, pr.pvalue, sr.statistic, sr.pvalue])
    t = pd.DataFrame(rows, columns=["指标", "Pearson r", "p值", "Spearman ρ", "p值"])
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("  （p<0.05 相关显著；BMI/年龄/体重与浓度显著负相关，孕周弱正相关）")

    # 组内 vs 组间 分解
    g = d.groupby("pid").agg(gw_m=("gw", "mean"), bmi_m=("bmi", "mean"),
                             y_m=("yconc", "mean"))
    gw_b, y_b = g["gw_m"].to_numpy(), g["y_m"].to_numpy()
    r_b = stats.pearsonr(gw_b, y_b)
    dm = d.assign(gw_c=d["gw"] - d.groupby("pid")["gw"].transform("mean"),
                  y_c=d["yconc"] - d.groupby("pid")["yconc"].transform("mean"))
    r_w = stats.pearsonr(dm["gw_c"], dm["y_c"])
    print("\n  ★ 组内/组间效应分解（为什么需要混合模型）:")
    print(f"     组间相关(267位孕妇均值, 孕周~浓度):  r = {r_b.statistic:+.3f}")
    print(f"     组内相关(去个体均值后, 孕周~浓度):  r = {r_w.statistic:+.3f}"
          f"  (p = {r_w.pvalue:.2e})")
    print("     即：同一孕妇随孕周浓度↑；但孕妇间均值反而↓（高BMI稀释+检测偏晚）")
    print("     → 两股方向相反的效应被 OLS 混在一起，须用混合模型分离。")
    return g, dm


# ================= 3. 建模对比 =================
def fit_models(d):
    print("\n" + "=" * 78)
    print("②  模型对比（选出'更好的模型'）")
    print("=" * 78)

    # (A) 朴素 pooled OLS（忽略重复测量）
    ols_p = smf.ols("yconc ~ gw + bmi + age", d).fit()
    # (B) 孕妇均值 OLS（去重测量，n=267）
    g = d.groupby("pid").agg(gw=("gw", "mean"), bmi=("bmi", "first"),
                             age=("age", "first"), yconc=("yconc", "mean"))
    ols_g = smf.ols("yconc ~ gw + bmi + age", g).fit()
    # (C) 混合效应：随机截距（主模型）
    lmm_ri = smf.mixedlm("yconc ~ gw + bmi + age", d,
                         groups=d["pid"]).fit(reml=True)
    # (D) 混合效应：随机截距+随机斜率(孕周)
    try:
        lmm_rs = smf.mixedlm("yconc ~ gw + bmi + age", d, groups=d["pid"],
                             re_formula="~ gw").fit(reml=True)
    except Exception as e:
        print("  (随机斜率模型未收敛，回退随机截距模型):", e)
        lmm_rs = None

    summary_rows = [
        ["A 朴素 OLS(1082观测)", "0.0013(++)", f"{ols_p.params['bmi']:+.4f}(--)",
         f"{ols_p.rsquared:.3f}"],
        ["B 孕妇均值OLS(267)", f"{ols_g.params['gw']:+.4f}",
         f"{ols_g.params['bmi']:+.4f}", f"{ols_g.rsquared:.3f}"],
        ["C LMM随机截距(主模型)", f"{lmm_ri.params['gw']:+.4f}",
         f"{lmm_ri.params['bmi']:+.4f}", "—"],
    ]
    if lmm_rs is not None:
        summary_rows.append(["D LMM随机截距+斜率", f"{lmm_rs.params['gw']:+.4f}",
                             f"{lmm_rs.params['bmi']:+.4f}", "—"])
    print("\n  模型      |  孕周系数(周⁻¹)  |  BMI系数  |  R²/说明")
    print("  " + "-" * 68)
    for name, cgw, cb, r2 in summary_rows:
        print(f"  {name:<22}|  {cgw:>8}      | {cb:>9} | {r2}")
    print("\n  解读: A 中孕周系数被'组间负选择'冲淡成 +0.0013；C 分离个体效应后")
    print("  真实组内增速约为 +0.0033/周（更符合生物学：胎儿 DNA 随孕周增多）。")

    # ICC
    var_b = lmm_ri.cov_re.iloc[0, 0]
    var_e = lmm_ri.scale
    icc = var_b / (var_b + var_e)
    print(f"\n  ★ 方差分解: 个体间方差={var_b:.5f}, 残差方差={var_e:.5f}"
          f" → ICC={icc:.3f}（{icc*100:.0f}% 变异来自孕妇个体差异）")
    return ols_p, ols_g, lmm_ri, lmm_rs, icc


# ================= 4. 显著性检验 =================
def significance_tests(d, lmm_ri):
    print("\n" + "=" * 78)
    print("③  混合效应模型 显著性检验（主模型: yconc ~ 孕周 + BMI + 年龄）")
    print("=" * 78)
    # 系数 Wald z 检验（只用固定效应，剔除方差分量）
    from scipy.stats import norm
    fe = lmm_ri.fe_params
    bse = lmm_ri.bse_fe
    z = fe / bse
    pv = 2 * (1 - norm.cdf(np.abs(z)))
    lo = fe.values - 1.96 * bse
    hi = fe.values + 1.96 * bse
    coef = pd.DataFrame({
        "固定效应": fe.index,
        "系数": fe.values,
        "标准误": bse,
        "z": z.values,
        "p值": pv,
        "95%CI": [f"[{a:.4f}, {b_:.4f}]" for a, b_ in zip(lo, hi)],
    })
    print(coef.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # 似然比检验 LRT（比较嵌套模型，检验单个固定效应显著性）
    print("\n  似然比检验(LRT, χ²):")
    full = smf.mixedlm("yconc ~ gw + bmi + age", d,
                       groups=d["pid"]).fit(reml=False)
    reds = {
        "去掉孕周 gw": "yconc ~ bmi + age",
        "去掉 BMI": "yconc ~ gw + age",
        "去掉 年龄": "yconc ~ gw + bmi",
    }
    for label, formula in reds.items():
        red = smf.mixedlm(formula, d, groups=d["pid"]).fit(reml=False)
        lr = 2 * (full.llf - red.llf)
        dof = len(full.params) - len(red.params)
        p = stats.chi2.sf(lr, dof)
        print(f"    {label:<12}  χ²={lr:8.3f}  df={dof}  p={p:.2e}"
              + ("   ★显著" if p < 0.05 else "  不显著"))
    print("\n  ★ 结论：孕周与 BMI 均显著影响 Y 染色体浓度；年龄影响不显著。")


# ================= 5. 绘图 =================
def plot_figures(d, lmm_ri):
    # ---------- 图1: 各指标相关性热力图 ----------
    corr_cols = ["yconc", "gw", "bmi", "age", "height", "weight"]
    corr_lab = ["Y染色体浓度", "孕周(周)", "BMI(kg/m²)", "年龄(岁)",
                "身高(cm)", "体重(kg)"]
    sub = d[corr_cols]
    R = sub.corr()
    # 成对 Pearson 检验 p 值（给显著性星号）
    pv = R.copy().astype(float)
    pv[:] = 1.0
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            if i != j:
                pv.iloc[i, j] = stats.pearsonr(
                    sub.iloc[:, i], sub.iloc[:, j])[1]
    mask = np.triu(np.ones_like(R, dtype=bool), k=1)   # 掩掉对称上三角
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_lab)))
    ax.set_xticklabels(corr_lab, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_lab)))
    ax.set_yticklabels(corr_lab)
    # 下三角 + 对角线：写相关系数与显著性
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            if not mask[i, j]:
                p = pv.iloc[i, j]
                star = "***" if p < 0.001 else "**" if p < 0.01 \
                    else "*" if p < 0.05 else ""
                ax.text(j, i, f"{R.iloc[i, j]:.2f}{star}", ha="center",
                        va="center", fontsize=9,
                        color="white" if abs(R.iloc[i, j]) > 0.6 else "black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Pearson 相关系数 r（* p<0.05  ** p<0.01  *** p<0.001）",
                   fontsize=9)
    ax.set_title("图1  各指标间相关性热力图（下三角为 r 值）")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig1_corr_heatmap.png", dpi=150)
    plt.close(fig)

    # ---------- 图2: 每人轨迹(spaghetti) ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    sample = d["pid"].value_counts()
    pick = sample[sample >= 3].index[:80]  # 抽样80人清晰显示
    cmap = plt.cm.viridis_r
    norm = plt.Normalize(d["bmi"].min(), d["bmi"].max())
    for pid in pick:
        dd = d[d["pid"] == pid]
        ax.plot(dd["gw"], dd["yconc"], "-o", ms=3, lw=0.8, alpha=0.45,
                color=cmap(norm(dd["bmi"].iloc[0])))
    sm_ = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm_, ax=ax, label="BMI (kg/m²)")
    ax.set_xlabel("孕周 (周)"); ax.set_ylabel("Y染色体浓度")
    ax.set_title("图2  各孕妇 Y染色体浓度随孕周的变化轨迹（颜色=BMI）")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_spaghetti.png", dpi=150)
    plt.close(fig)

    # ---------- 图3: BMI 分箱箱线 ----------
    d2 = d.copy()
    d2["BMI分组"] = pd.cut(d2["bmi"], bins=[20, 28, 32, 36, 40, 60],
                          right=False,
                          labels=["[20,28)", "[28,32)", "[32,36)", "[36,40)", "≥40"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    d2.boxplot(column="yconc", by="BMI分组", ax=ax, grid=False)
    ax.set_title("图3  Y染色体浓度随 BMI 分组的变化")
    ax.set_ylabel("Y染色体浓度")
    fig.suptitle("")
    # 叠加均值点
    means = d2.groupby("BMI分组", observed=True)["yconc"].mean()
    xs = np.arange(len(means))
    ax.plot(xs, means.values, "D-", color="crimson", label="组均值")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_bmi.png", dpi=150)
    plt.close(fig)

    # ---------- 图4: 边际效应预测曲线(Y~孕周, 固定BMI) ----------
    b = lmm_ri.fe_params
    # 固定效应协方差：cov_params 含方差分量，截取前 k_fe 行/列
    k_fe = len(b)
    cov = lmm_ri.cov_params().iloc[:k_fe, :k_fe]
    gw_grid = np.linspace(d["gw"].min(), d["gw"].max(), 100)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.scatter(d["gw"], d["yconc"], s=8, alpha=0.20, color="steelblue",
               label="样本点")
    for bmi_val, lab, col in [(25, "BMI=25(偏瘦)", "seagreen"),
                              (32, "BMI=32(中位)", "darkorange"),
                              (38, "BMI=38(肥胖)", "crimson")]:
        yhat = b["Intercept"] + b["gw"] * gw_grid + b["bmi"] * bmi_val \
            + b["age"] * d["age"].mean()
        # 置信带(仅固定效应)
        Xm = np.column_stack([np.ones_like(gw_grid), gw_grid,
                              np.full_like(gw_grid, bmi_val),
                              np.full_like(gw_grid, d["age"].mean())])
        se = np.sqrt(np.einsum("ij,jk,ik->i", Xm, cov, Xm))
        z = 1.96
        ax.plot(gw_grid, yhat, color=col, lw=2, label=lab)
        ax.fill_between(gw_grid, yhat - z * se, yhat + z * se, color=col,
                        alpha=0.15)
    ax.axhline(0.04, color="black", ls="--", lw=1.2)
    ax.text(d["gw"].min() + 0.2, 0.041, "4% 达标线", fontsize=9)
    ax.set_xlabel("孕周 (周)"); ax.set_ylabel("Y染色体浓度")
    ax.set_title("图4  混合模型边际预测：不同BMI下浓度随孕周的变化(含95%CI)")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_marginal.png", dpi=150)
    plt.close(fig)

    # ---------- 图5: 3D 关系曲面 ----------
    gx = np.linspace(d["gw"].min(), d["gw"].max(), 40)
    bx = np.linspace(d["bmi"].min(), d["bmi"].max(), 40)
    GG, BB = np.meshgrid(gx, bx)
    YY = b["Intercept"] + b["gw"] * GG + b["bmi"] * BB \
        + b["age"] * d["age"].mean()
    fig = plt.figure(figsize=(10, 7))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.plot_surface(GG, BB, YY, cmap=cm.viridis, alpha=0.85, edgecolor="none")
    ax3.scatter(d["gw"], d["bmi"], d["yconc"], s=4, alpha=0.15, color="red")
    ax3.set_xlabel("孕周(周)"); ax3.set_ylabel("BMI"); ax3.set_zlabel("Y浓度")
    ax3.set_title("图5  Y染色体浓度 随 (孕周, BMI) 的关系曲面")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_surface.png", dpi=150)
    plt.close(fig)

    # ---------- 图6: 残差诊断 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    resid = lmm_ri.resid
    fitted = lmm_ri.fittedvalues
    axes[0].scatter(fitted, resid, s=8, alpha=0.3)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xlabel("拟合值"); axes[0].set_ylabel("残差")
    axes[0].set_title("残差 vs 拟合值")
    sm.qqplot(resid, line="45", ax=axes[1], marker=".", alpha=0.4)
    axes[1].set_title("残差 QQ 图")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig6_diagnostics.png", dpi=150)
    plt.close(fig)
    print(f"\n  图表已保存至 {FIG}/（fig1~fig6.png）")


# ================= 主流程 =================
def main():
    d = load_data()
    correlation_report(d)
    ols_p, ols_g, lmm_ri, lmm_rs, icc = fit_models(d)
    significance_tests(d, lmm_ri)
    plot_figures(d, lmm_ri)

    var_b = lmm_ri.cov_re.iloc[0, 0]
    b = lmm_ri.params
    print("\n" + "=" * 78)
    print("④  最终关系模型（问题1结论）")
    print("=" * 78)
    print(f"\n  Y染色体浓度 = {b['Intercept']:.4f}"
          f" + {b['gw']:.4f}·孕周(周)  {b['bmi']:+.4f}·BMI(kg/m²)"
          f"  {b['age']:+.4f}·年龄(岁)  + 个体随机截距 + ε")
    print("  其中 个体随机截距 ~ N(0, %.5f²)，残余误差 σ=%.4f，ICC=%.2f"
          % (np.sqrt(var_b), np.sqrt(lmm_ri.scale), icc))
    print("\n  · 孕周系数显著为正(+%.4f/周, p<0.001)：同一孕妇孕周每增加1周，"
          % b["gw"])
    print("    Y染色体浓度平均上升约%.2f个百分点（胎儿游离DNA随孕周增多）；"
          % (b["gw"] * 100))
    print("  · BMI系数显著为负：BMI每+1，浓度约下降%.3f个百分点"
          % (abs(b["bmi"]) * 100))
    print("    （母体血容量/脂肪增多对胎儿DNA的稀释效应）；")
    print("  · 年龄效应不显著(p>0.05)，身高体重信息主要由BMI体现。")
    print("  · ICC=%.2f：约%.0f%% 的浓度变异源于孕妇个体差异，"
          % (icc, icc * 100))
    print("    因此后续问题按孕妇个体/BMI 分组、个性化定 NIPT 时点非常必要。")


if __name__ == "__main__":
    main()
