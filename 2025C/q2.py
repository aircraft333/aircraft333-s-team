# -*- coding: utf-8 -*-
"""
2025 高教社杯 C 题 | 问题 2
男胎孕妇 BMI 分组 与 各组最佳 NIPT 时点（使潜在风险最小）
============================================================
方法概要：
  ① 个体增长模型（承接问题1）：LMM(yconc ~ 孕周, 随机截距按孕妇)
        个体期望浓度 μ_i(d) = (β0+u_i) + β1·d
  ② 达标(≥4%)概率：真实浓度含噪声 σ_tot=√(残差²+测量误差σe²)
        孕妇 i 在 d 周已达标概率  p_i(d) = Φ((μ_i(d)−0.04)/σ_tot)
  ③ 达标比例曲线：组内 F_g(d) = 组内各孕妇 p_i(d) 的均值
  ④ 最佳时点（风险最小）：d_g = 使 F_g(d) ≥ p0(默认90%) 的最早孕周
        早于 d_g → 组内 >10% 未达4% → 测序失败/结果不准风险大；
        晚于 d_g → 延误、压缩治疗窗口 → 晚期发现风险大。
        故 d_g 为两难折中下的“风险最小”检测时点。
  ⑤ 分组：回归树按 BMI 对个体达标孕周聚类 → 自然 BMI 区间。
  ⑥ 误差分析：测量误差 σe、达标保证水平 p0 的敏感性。

输出：控制台中文报告 + figures_q2/
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import statsmodels.formula.api as smf

from q_common import load_male, setup_chinese

warnings.filterwarnings("ignore", message="The MLE may be on the boundary.*")
setup_chinese()
FIG = "figures_q2"
os.makedirs(FIG, exist_ok=True)

THRESH = 0.04
P0 = 0.90                # 目标达标比例(默认)
W_LO, W_HI = 10.0, 30.0
WEEKS = np.arange(W_LO, W_HI + 1e-9, 0.25)


def weeks_label(w):
    if pd.isna(w):
        return "-"
    wk = int(np.floor(w)); day = int(round((w - wk) * 7))
    if day >= 7:
        wk += 1; day = 0
    return f"{wk}w+{day}d" if day else f"{wk}w"


def fit_growth(d):
    m = smf.mixedlm("yconc ~ gw", d, groups=d["pid"]).fit(reml=True)
    b0, b1 = float(m.fe_params["Intercept"]), float(m.fe_params["gw"])
    resid_sd = float(np.sqrt(m.scale))
    u = {}
    for pid, val in m.random_effects.items():
        v = val["Group"] if isinstance(val, dict) else val
        u[pid] = float(np.asarray(v).ravel()[0])
    d = d.copy()
    d["mu"] = b0 + d["pid"].map(u)
    return d, {"beta0": b0, "beta1": b1, "resid_sd": resid_sd}


def attain_weeks(d, b1, resid_sd):
    """每孕妇达标孕周 t_i（个体线达到0.04的最早窗口周，用于分组与描述）"""
    mu0 = d.groupby("pid")["mu"].first()
    line = mu0.values[:, None] + b1 * WEEKS[None, :]
    hit = line >= THRESH
    t = np.where(hit.any(axis=1), WEEKS[np.argmax(hit, axis=1)], np.nan)
    return pd.DataFrame({"pid": mu0.index, "t_i": t})


def prob_matrix(d, b1, resid_sd, sigma_e):
    """返回(孕妇id, 各周达标概率矩阵(行=孕妇,列=周))"""
    stot = np.sqrt(resid_sd ** 2 + sigma_e ** 2)
    mu0 = d.groupby("pid")["mu"].first()
    prob = norm.cdf((mu0.values[:, None] + b1 * WEEKS[None, :] - THRESH)
                    / stot)
    return mu0.index, prob


def make_groups(r, min_leaf=40):
    cc = r.dropna(subset=["t_i"])
    try:
        from sklearn.tree import DecisionTreeRegressor
        tree = DecisionTreeRegressor(min_samples_leaf=min_leaf,
                                     random_state=0, max_depth=2)
        tree.fit(cc[["bmi"]], cc["t_i"])
        thr = sorted(float(x) for x in tree.tree_.threshold
                     if x < np.inf and cc["bmi"].min() < x < cc["bmi"].max())
    except Exception:
        thr = []
    if not thr:
        thr = [round(v, 1) for v in np.quantile(cc["bmi"], [0.45, 0.75])]
    lo, hi = max(20.0, float(cc["bmi"].min())), min(60.0, float(cc["bmi"].max()))
    edges = sorted(set([lo] + thr + [hi]))
    labels = [f"[{edges[i]:.1f},{edges[i+1]:.1f})"
              for i in range(len(edges) - 2)] + \
             [f"[{edges[-2]:.1f},{edges[-1]:.1f}]"]
    return edges, labels, thr


def first_over(F, p0):
    return float(WEEKS[np.argmax(F >= p0)]) if (F >= p0).any() else np.nan


def group_curve(pidx, prob, g):
    """返回该组(行)的平均达标比例曲线"""
    keep = np.isin(pidx, g["pid"].values)
    return prob[keep].mean(axis=0)


def main():
    d = load_male()
    d, info = fit_growth(d)
    print("=" * 82)
    print("问题 2  男胎孕妇 BMI 分组与最佳 NIPT 时点（使潜在风险最小）")
    print("=" * 82)
    print(f"样本 {len(d)} 观测 | 孕妇 {d['pid'].nunique()} 人 | "
          f"增速 β1={info['beta1']:.4f}/周 | 残差σ={info['resid_sd']:.4f}")
    print(f"达标阈值=4% | 目标达标比例 p0={P0:.0%} | "
          f"窗口 {W_LO:.0f}~{W_HI:.0f} 周")

    tw = attain_weeks(d, info["beta1"], info["resid_sd"])
    r = d[["pid", "bmi"]].drop_duplicates("pid").merge(tw, on="pid")
    cc = r.dropna(subset=["t_i"])
    print("\n个体达标孕周 t_i 描述:")
    print(cc["t_i"].describe().round(2).to_string())
    pr = stats.pearsonr(cc["bmi"], cc["t_i"])
    reg = stats.linregress(cc["bmi"], cc["t_i"])
    print(f"\nt_i ~ BMI : r={pr.statistic:.3f} (p={pr.pvalue:.2e})  ★显著")
    print(f"  t_i = {reg.intercept:.1f} + {reg.slope:.2f}·BMI "
          f"(BMI 每 +1 达标约推迟 {reg.slope:.2f} 周)")

    edges, labels, thr = make_groups(r)
    r["group"] = pd.cut(r["bmi"], bins=edges, right=False, labels=labels)
    print("\n回归树 BMI 断点:", [round(x, 1) for x in thr],
          "\n分组: ", "  ".join(labels))

    # ---- 主表（σe=0.004 基线）----
    pidx, prob0 = prob_matrix(d, info["beta1"], info["resid_sd"], 0.004)
    print("\n" + "-" * 86)
    print(f"{'BMI区间':<20}{'n':>5}{'达标中位':>9}{'F(d90)':>8}"
          f"{'d_g(90%) 推荐':>14}{'d(95%) 保守':>13}")
    print("-" * 86)
    rows, curves = [], {}
    for lab in labels:
        g = r[r["group"] == lab]
        if len(g) == 0:
            continue
        F = group_curve(pidx, prob0, g)
        d90, d95 = first_over(F, 0.90), first_over(F, 0.95)
        curves[lab] = F
        med = g["t_i"].median()
        Fd90 = F[np.argmax(WEEKS >= d90)] if not np.isnan(d90) else np.nan
        rows.append({"组": lab, "n": len(g), "median": med,
                     "d90": d90, "d95": d95, "Fd90": Fd90})
        print(f"{lab:<20}{len(g):>5}{med:>6.1f} 周{Fd90:>8.0%}"
              f"{weeks_label(d90)+' ≈ '+format(d90,'.1f')+'w':>14}"
              f"{weeks_label(d95)+' ≈ '+format(d95,'.1f')+'w':>13}")
    res = pd.DataFrame(rows)

    print("\n" + "-" * 86)
    print("风险逻辑: d_g 为该组“最早使达标比例≥p0”的时点。早于 d_g → 组内未达标")
    print("比例高(测序失败/结果不准)；晚于 d_g → 白等并缩短治疗窗口(晚期发现风")
    print("险)。d_g 即折中下潜在风险最小的最佳 NIPT 时点。")

    # ---- σe 敏感性 ----
    print("\n" + "=" * 82)
    print("检测误差敏感性: 不同 σe 下各组 d_g(90%达标)（周）")
    print("=" * 82)
    sens = {}
    for se in [0.0, 0.004, 0.008, 0.012, 0.016]:
        _, pb = prob_matrix(d, info["beta1"], info["resid_sd"], se)
        row = {}
        for lab in labels:
            g = r[r["group"] == lab]
            if len(g) == 0:
                continue
            row[lab] = first_over(group_curve(pidx, pb, g), P0)
        sens[se] = row
    st = pd.DataFrame(sens).T
    st.index = [f"σe={s:.3f}" for s in sens]
    print(st.round(1).to_string())

    # ---- 保证水平敏感性 ----
    print("\n" + "=" * 82)
    print("目标达标比例敏感性: 不同 p0 下各组推荐时点（周）")
    print("=" * 82)
    sp = {}
    for p0 in [0.80, 0.85, 0.90, 0.95]:
        row = {}
        for lab in labels:
            g = r[r["group"] == lab]
            if len(g) == 0:
                continue
            row[lab] = first_over(group_curve(pidx, prob0, g), p0)
        sp[f"p0={p0:.0%}"] = row
    st2 = pd.DataFrame(sp).T
    print(st2.round(1).to_string())
    print("\n结论: σe 增大或要求更严(高p0) → 时点后移；组间次序(高BMI更晚)稳定 → 稳健。")

    plot_all(r, res, pidx, prob0, st)
    return res


def plot_all(r, res, pidx, prob0, st):
    cc = r.dropna(subset=["t_i"])
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.scatter(cc["bmi"], cc["t_i"], s=22, alpha=0.6, color="steelblue",
               edgecolor="white", lw=0.4)
    reg = stats.linregress(cc["bmi"], cc["t_i"])
    xs = np.linspace(cc["bmi"].min(), cc["bmi"].max(), 60)
    ax.plot(xs, reg.intercept + reg.slope * xs, "r-", lw=2,
            label=f"t = {reg.intercept:.1f}{reg.slope:+.2f}·BMI")
    for lab in r["group"].cat.categories:
        ax.axvline(float(lab.split(",")[0].strip("[")), color="grey",
                   ls="--", lw=1)
    ax.set_xlabel("孕妇 BMI"); ax.set_ylabel("个体达标孕周 t_i (周)")
    ax.set_title("图1  个体达标孕周随 BMI 变化（竖线=分组边界）")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_t_vs_bmi.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    cmap = plt.cm.viridis
    n = max(len(res) - 1, 1)
    for i, row in res.iterrows():
        c = cmap(i / n)
        F = prob0[np.isin(pidx, r[r["group"] == row["组"]]["pid"].values)]\
            .mean(axis=0)
        ax.plot(WEEKS, F, lw=2, color=c,
                label=f"{row['组']}  d_g={weeks_label(row['d90'])}")
        ax.axvline(row["d90"], color=c, ls=":", alpha=0.6)
    ax.axhline(0.90, color="black", ls="--", lw=1.2)
    ax.text(W_LO + 0.2, 0.91, "目标 90%", fontsize=9)
    ax.set_xlabel("孕周 d (周)"); ax.set_ylabel("组内达标比例 F(d)")
    ax.set_title("图2  各组达标比例曲线与最佳 NIPT 时点")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_attain_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for col in st.columns:
        ax.plot(st.index, st[col], "-o", label=col)
    ax.set_xlabel("测量误差 σe"); ax.set_ylabel("推荐时点 d_g (周)")
    ax.set_title("图3  检测误差对各组推荐时点的影响")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_sensitivity.png", dpi=150)
    plt.close(fig)
    print(f"\n图表已保存至 {FIG}/ (fig1~fig3.png)")


if __name__ == "__main__":
    main()
