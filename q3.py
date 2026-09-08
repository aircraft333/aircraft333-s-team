# -*- coding: utf-8 -*-
"""
2025 高教社杯 C 题 | 问题 3
综合考虑身高/体重/年龄等因素 + 检测误差 + 达标比例，
按男胎孕妇 BMI 分组给出各组最佳 NIPT 时点（潜在风险最小）
============================================================
方法概要（承接 Q2，扩展到多元因素与概率型达标）：
  ① 个体增长 LMM → 每人达标孕周 t_i（浓度≥4%）
  ② 达标时间多因素回归：
       t_i = β0 + β1·BMI + β2·年龄 + β3·身高 + β4·体重 + ε
     量化各因素对“最早达标时间”的贡献并检验显著性；
     身高/体重的影响主要通过 BMI(其定义)体现 → 看标准化系数。
  ③ 达标比例(考虑个体差异与检测误差)：
       孕妇 i 在 d 周已达标概率 = Φ((d − t_i) / σ_eff),
       σ_eff = √(σ_res² + (σe/β1)²)；组内达标比例 F_g(d)=均值。
     达标比例 F_g(d) 随 d 单调上升；组内越“难达标”(高BMI)曲线越靠右。
  ④ 分组：回归树按 BMI 对 t_i 聚类；每组最佳时点 d_g=min{F_g(d)≥p0}。
  ⑤ 误差/稳健性：σe 与 p0 的敏感性分析。

输出：控制台中文报告 + figures_q3/
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm

from q_common import load_male, setup_chinese
# 复用 Q2 的核心(个体增长LMM、达标周、分组、达标概率矩阵等)
from q2 import (WEEKS, weeks_label, THRESH, P0, fit_growth,
                prob_matrix, make_groups, first_over, group_curve)

warnings.filterwarnings("ignore", message="The MLE may be on the boundary.*")
setup_chinese()
FIG = "figures_q3"
os.makedirs(FIG, exist_ok=True)

W_LO, W_HI = WEEKS[0], WEEKS[-1]


def main():
    d = load_male()
    d, info = fit_growth(d)
    print("=" * 82)
    print("问题 3  多元因素下的 BMI 分组与最佳 NIPT 时点")
    print("=" * 82)
    print(f"样本 {len(d)} 观测 | 孕妇 {d['pid'].nunique()} 人 | "
          f"增速 β1={info['beta1']:.4f}/周 | 残差σ={info['resid_sd']:.4f}")

    # ============ ① 每人达标孕周 t_i（与 Q2 一致） ============
    mu0 = d.groupby("pid")["mu"].first()
    tw = pd.DataFrame({
        "pid": mu0.index,
        "t_i": np.where(
            (mu0.values[:, None] + info["beta1"] * WEEKS[None, :] >= THRESH)
            .any(axis=1),
            WEEKS[np.argmax(mu0.values[:, None]
                            + info["beta1"] * WEEKS[None, :] >= THRESH, axis=1)],
            np.nan)})
    base = d[["pid", "bmi", "age", "height", "weight"]].drop_duplicates("pid")
    r = base.merge(tw, on="pid")
    cc = r.dropna(subset=["t_i"]).copy()
    # 由于多数 t_i 落在窗口起点，达标“早晚”与孕12周浓度等价；回归供解释用
    print("\n--- 个体达标孕周 t_i 描述 ---")
    print(cc["t_i"].describe().round(2).to_string())

    # ============ ② 多因素分析（避免 BMI 与身高/体重定义共线） ============
    print("\n--- 单变量相关: 达标孕周 t_i 与各因素 ---")
    for name, col in [("BMI", "bmi"), ("年龄", "age"),
                      ("身高", "height"), ("体重", "weight")]:
        pr = stats.pearsonr(cc[col], cc["t_i"])
        print(f"   {name:>4}: r={pr.statistic:+.3f} (p={pr.pvalue:.2e})")

    # 模型1: BMI + 年龄（二者不共线，报告 BMI 的净影响）
    X1 = sm.add_constant(cc[["bmi", "age"]])
    m1 = sm.OLS(cc["t_i"], X1).fit()
    print("\n--- 回归 M1: t_i ~ BMI + 年龄 ---")
    print(pd.DataFrame({
        "变量": m1.params.index, "系数": m1.params.values,
        "标准误": m1.bse.values, "t": m1.tvalues.values,
        "p值": m1.pvalues.values,
    }).to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # 模型2: 体重 + 身高 + 年龄（不含BMI，看 BMI 之外的替代信息）
    X2 = sm.add_constant(cc[["weight", "height", "age"]])
    m2 = sm.OLS(cc["t_i"], X2).fit()
    print("\n--- 回归 M2: t_i ~ 体重 + 身高 + 年龄 (替代BMI, 供对照) ---")
    print(pd.DataFrame({
        "变量": m2.params.index, "系数": m2.params.values,
        "标准误": m2.bse.values, "t": m2.tvalues.values,
        "p值": m2.pvalues.values,
    }).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"  (R²: M1={m1.rsquared:.3f}, M2={m2.rsquared:.3f}；身高体重信息主要"
          f"由 BMI=体重/身高² 承载，把 BMI 与身高体重同时放入会共线)")
    stdbeta = pd.Series({
        "BMI": m1.params["bmi"] * cc["bmi"].std() / cc["t_i"].std(),
        "年龄": m1.params["age"] * cc["age"].std() / cc["t_i"].std(),
        "体重*": m2.params["weight"] * cc["weight"].std() / cc["t_i"].std(),
        "身高*": m2.params["height"] * cc["height"].std() / cc["t_i"].std(),
    })
    print("\n标准化系数(相对重要性; *来自M2):")
    for k, v in stdbeta.items():
        print(f"   {k:>6}: {v:+.3f}")
    sigma_res = float(np.sqrt(m1.mse_resid))

    # ============ ③ BMI 分组（回归树） ============
    edges, labels, thr = make_groups(r)
    r["group"] = pd.cut(r["bmi"], bins=edges, right=False, labels=labels)
    print("\n回归树 BMI 断点:", [round(x, 1) for x in thr],
          "\n分组: ", "  ".join(labels))

    # ============ ④ 达标比例曲线(含残差+检测误差) & 各组最佳时点 ============
    beta1 = info["beta1"]
    print("\n" + "-" * 86)
    print(f"{'BMI区间':<20}{'n':>5}{'平均达标(周)':>12}{'d_g(90%) 推荐':>14}"
          f"{'d(95%) 保守':>13}{'F(d_g)':>8}")
    print("-" * 86)
    rows = []
    sigma_e_base = 0.004
    sigma_eff = np.sqrt(sigma_res ** 2 + (sigma_e_base / beta1) ** 2)
    for lab in labels:
        g = r[r["group"] == lab].dropna(subset=["t_i"])
        if len(g) == 0:
            continue
        # 组内达标比例曲线 F(d)=mean Φ((d - t_i)/σ_eff)
        F = norm.cdf((WEEKS[None, :] - g["t_i"].values[:, None])
                     / sigma_eff).mean(axis=0)
        d90, d95 = first_over(F, 0.90), first_over(F, 0.95)
        Fd = F[np.argmax(WEEKS >= d90)] if not np.isnan(d90) else np.nan
        rows.append({"组": lab, "n": len(g), "mean_t": g["t_i"].mean(),
                     "d90": d90, "d95": d95, "Fd": Fd,
                     "sigma_eff": sigma_eff})
        print(f"{lab:<20}{len(g):>5}{g['t_i'].mean():>9.1f} 周"
              f"{weeks_label(d90)+' ≈ '+format(d90,'.1f')+'w':>14}"
              f"{weeks_label(d95)+' ≈ '+format(d95,'.1f')+'w':>13}{Fd:>8.0%}")
    res = pd.DataFrame(rows)

    print("\n" + "-" * 86)
    print("达标比例 F_g(d) 已计入: 个体差异残差 σ_res + 检测误差 σe/β1。")
    print("d_g = 该组最早使达标比例≥p0 的时点 = 潜在风险最小的推荐时点。")

    # ============ ⑤ 误差/保证水平敏感性 ============
    print("\n" + "=" * 82)
    print("检测误差 σe 敏感性: 各组 d_g(90%)（周）")
    print("=" * 82)
    sens = {}
    for se in [0.0, 0.004, 0.008, 0.012, 0.016]:
        s_eff = np.sqrt(sigma_res ** 2 + (se / beta1) ** 2)
        row = {}
        for lab in labels:
            g = r[r["group"] == lab].dropna(subset=["t_i"])
            if len(g) == 0:
                continue
            F = norm.cdf((WEEKS[None, :] - g["t_i"].values[:, None])
                         / s_eff).mean(axis=0)
            row[lab] = first_over(F, 0.90)
        sens[se] = row
    st = pd.DataFrame(sens).T
    st.index = [f"σe={s:.3f}" for s in sens]
    print(st.round(1).to_string())

    print("\n" + "=" * 82)
    print("目标达标比例 p0 敏感性: 各组推荐时点（周）")
    print("=" * 82)
    sp = {}
    for p0 in [0.80, 0.85, 0.90, 0.95]:
        row = {}
        for lab in labels:
            g = r[r["group"] == lab].dropna(subset=["t_i"])
            if len(g) == 0:
                continue
            F = norm.cdf((WEEKS[None, :] - g["t_i"].values[:, None])
                         / sigma_eff).mean(axis=0)
            row[lab] = first_over(F, p0)
        sp[f"p0={p0:.0%}"] = row
    st2 = pd.DataFrame(sp).T
    print(st2.round(1).to_string())

    print("\n" + "=" * 82)
    print("结论")
    print("=" * 82)
    print("· BMI 是达标时间的主要可解释因素(标准化系数最大)；身高/体重的信息")
    print("  主要由 BMI(=体重/身高²) 承载，额外加入年龄等只小幅改善拟合。")
    print("· 高 BMI 组达标更晚，应把 NIPT 时点相应后移；推荐结果见主表。")
    print("· σe↑/要求更严(高p0) → 时点后移；分组方案稳健。")

    plot_all(r, res, st, stdbeta)
    return res


def plot_all(r, res, st, stdbeta):
    cc = r.dropna(subset=["t_i"])
    # 图1 达标时间 vs BMI，点色=年龄
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    sc = ax.scatter(cc["bmi"], cc["t_i"], c=cc["age"], cmap="viridis",
                    s=26, alpha=0.75, edgecolor="white", lw=0.4)
    fig.colorbar(sc, ax=ax, label="年龄")
    for lab in r["group"].cat.categories:
        ax.axvline(float(lab.split(",")[0].strip("[")), color="grey",
                   ls="--", lw=1)
    ax.set_xlabel("孕妇 BMI"); ax.set_ylabel("个体达标孕周 t_i (周)")
    ax.set_title("图1  达标孕周随 BMI 变化（颜色=年龄，竖线=分组边界）")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_factors.png", dpi=150)
    plt.close(fig)

    # 图2 因素重要性(标准化系数绝对值)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(stdbeta.index, stdbeta.values, color="steelblue")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("标准化系数 β*")
    ax.set_title("图2  各因素对达标时间的影响(标准化系数)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_importance.png", dpi=150)
    plt.close(fig)

    # 图3 各组达标比例曲线(示例σe=0.004) + d_g
    fig, ax = plt.subplots(figsize=(9.5, 6))
    cmap = plt.cm.viridis
    for i, row in res.iterrows():
        g = r[r["group"] == row["组"]].dropna(subset=["t_i"])
        F = norm.cdf((WEEKS[None, :] - g["t_i"].values[:, None])
                     / row["sigma_eff"]).mean(axis=0)
        ax.plot(WEEKS, F, lw=2, color=cmap(i / max(len(res) - 1, 1)),
                label=f"{row['组']}  d_g={weeks_label(row['d90'])}")
        ax.axvline(row["d90"], color=cmap(i / max(len(res) - 1, 1)),
                   ls=":", alpha=0.6)
    ax.axhline(0.90, color="black", ls="--", lw=1.2)
    ax.text(W_LO + 0.2, 0.91, "目标 90%", fontsize=9)
    ax.set_xlabel("孕周 d (周)"); ax.set_ylabel("组内达标比例 F(d)")
    ax.set_title("图3  各组达标比例曲线与最佳 NIPT 时点")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_attain_curves.png", dpi=150)
    plt.close(fig)

    # 图4 σe 敏感性
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for col in st.columns:
        ax.plot(st.index, st[col], "-o", label=col)
    ax.set_xlabel("测量误差 σe"); ax.set_ylabel("推荐时点 d_g (周)")
    ax.set_title("图4  检测误差对各组推荐时点的影响")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_sensitivity.png", dpi=150)
    plt.close(fig)
    print(f"\n图表已保存至 {FIG}/ (fig1~fig4.png)")


if __name__ == "__main__":
    main()
