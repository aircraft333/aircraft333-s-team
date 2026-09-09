# -*- coding: utf-8 -*-
"""
2025 高教社杯 C 题 | 问题 4
女胎异常的判定方法（以 13/18/21 号染色体非整倍体 AB 列为判定结果）
============================================================
背景：女胎无 Y 染色体，不能用 Y 浓度判定。本表以 AB(染色体的非整倍体)
为标签(阳性=T13/T18/T21 等，67/605)，用 X染色体浓度/Z 值、13/18/21 的
Z 值与 GC 含量、读段数及比例、BMI 等建立异常判定模型。

方法：
  ① 单变量判别力(AUC)扫描 → 选特征（本数据里 X染色体浓度判别力最强）
  ② 标准化 Logistic 回归(L2, 类别均衡)，5 折分层 CV：
       - 用 Out-of-Fold 预测算 ROC-AUC / PR-AUC / 混淆矩阵
       - 由 OOF 的 Youden 及“特异度≥95%”确定操作阈值
  ③ 全量数据重训得“判定公式” logit(S)=β0+Σβx，P≥τ 判为异常
  ④ Bootstrap 给出 AUC 95% 置信区间；给出 ROC/PR/特征重要性图

注意：本附件为构造数据，各染色体 Z 值在正常/异常间差异很小，模型主要
依赖 X 染色体浓度；结论按“方法框架 + 实测性能”如实报告。

输出：控制台中文报告 + figures_q4/
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from q_common import load_female, setup_chinese

warnings.filterwarnings("ignore")
setup_chinese()
FIG = "figures_q4"
os.makedirs(FIG, exist_ok=True)

FEATURES = [
    "xc",            # X染色体浓度
    "zx",            # X染色体Z值
    "z13", "z18", "z21",
    "gc", "gc13", "gc18", "gc21",
    "n_raw", "map_ratio", "dup_ratio", "n_uniq", "filter_ratio",
    "bmi", "age", "gw",
]


def univariate_auc(y, x):
    x = np.asarray(x, float)
    m = ~np.isnan(x)
    y, x = y[m], x[m]
    if len(np.unique(y)) < 2 or x.std() == 0:
        return np.nan
    from sklearn.metrics import roc_auc_score
    a = roc_auc_score(y, x)
    # 方向：若 AUC<0.5 反转，报告 max(AUC,1-AUC) 与方向
    return max(a, 1 - a), a < 0.5


def main():
    df = load_female()
    df = df.dropna(subset=FEATURES + ["label"])
    y = df["label"].values
    print("=" * 82)
    print("问题 4  女胎异常判定方法")
    print("=" * 82)
    print(f"样本 {len(df)} | 阳性(非整倍体) {int(y.sum())} | "
          f"阴性 {int((1-y).sum())}")

    # ① 单变量判别力
    print("\n--- 单变量判别力(AUC, 已取≥0.5方向) ---")
    tab = []
    for f in FEATURES:
        a, rev = univariate_auc(y, df[f].values)
        tab.append([f, a, "越小越异常" if rev else "越大越异常"])
    t = pd.DataFrame(tab, columns=["特征", "AUC", "方向"]).sort_values(
        "AUC", ascending=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    strong = t[t["AUC"] >= 0.6]
    print("\n判别力较强(AUC≥0.6)的特征:", "  ".join(strong["特征"]) if
          len(strong) else "（无，见下方说明）")

    # ② 标准化逻辑回归 + 5折CV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 roc_curve, precision_recall_curve,
                                 confusion_matrix)
    X = df[FEATURES].values.astype(float)
    X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
    # 均值填补缺失(若还有)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = np.nanmean(X[:, j])

    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                               max_iter=2000))])

    # 全量拟合（用于系数/重要性展示）
    pipe.fit(X, y)
    lr = pipe.named_steps["lr"]
    sc = pipe.named_steps["sc"]
    coef_std = lr.coef_[0]                 # 标准化系数
    OR = np.exp(coef_std)                  # X每+1个SD的几率比

    # 5折分层CV → OOF 预测
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y)); oof_roc = []
    for tr, te in skf.split(X, y):
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
        oof_roc.append(roc_auc_score(y[te], oof[te]))
    auc_oof = roc_auc_score(y, oof)
    ap_oof = average_precision_score(y, oof)
    print("\n" + "=" * 82)
    print("5 折分层 CV 结果 (Out-of-Fold)")
    print("=" * 82)
    print(f"  ROC-AUC: 每折 {['%.3f' % a for a in oof_roc]} | "
          f"OOF 总 AUC = {auc_oof:.3f}")
    print(f"  PR-AUC (阳性为少数): {ap_oof:.3f}")

    # OOF ROC 阈值
    fpr, tpr, thr = roc_curve(y, oof)
    youden = tpr - fpr
    iy = int(np.argmax(youden))
    th_youden = thr[iy]
    # 特异度≥95% 的阈值
    ok = np.where(fpr <= 0.05)[0]
    th_spec = thr[ok[-1]] if len(ok) else thr[iy]
    print(f"\n  操作阈值(Youden): P≥{th_youden:.3f}  → "
          f"灵敏度={tpr[iy]:.3f} 特异度={1-fpr[iy]:.3f}")
    for ts in [0.10, 0.05]:
        ok = np.where(fpr <= ts)[0]
        if len(ok):
            j = ok[-1]
            print(f"  操作阈值(特异度≥{1-ts:.0%}): P≥{thr[j]:.3f}  → "
                  f"灵敏度={tpr[j]:.3f} 特异度={1-fpr[j]:.3f}")

    # ③ 全量模型的判定公式与 OR
    print("\n" + "=" * 82)
    print("判定公式（标准化特征 Logistic 回归, 全量重训）")
    print("=" * 82)
    beta0 = lr.intercept_[0]
    coef_df = pd.DataFrame({
        "特征": FEATURES, "标准化系数β": coef_std,
        "OR(每+1SD)": OR,
    }).sort_values("标准化系数β", key=abs, ascending=False)
    print(coef_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  logit(P) = {beta0:.3f} + Σ β_j·x_j*（x*为标准化特征）")
    print("  判定: P = 1/(1+e^-logit) ≥ τ → 判为异常(建议 τ 用 Youden 阈值)")
    print("\n  判读: X染色体浓度系数为负(越负越异常, OR 偏离1最大)；13/18/21")
    print("  Z 值与 GC/读段类特征在本数据中贡献很小 → 方法框架+实测如实报告。")

    # ④ Bootstrap AUC 95%CI
    print("\n--- Bootstrap 95% CI of OOF AUC ---")
    rng = np.random.default_rng(0)
    boots = []
    n = len(y)
    for _ in range(500):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], oof[idx]))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  AUC={auc_oof:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]")

    # ⑤ 判定效果(Youden 阈值)混淆矩阵
    cm = confusion_matrix(y, (oof >= th_youden).astype(int))
    print("\n--- OOF 判定混淆矩阵 (τ=Youden=%.3f) ---" % th_youden)
    print(pd.DataFrame(cm, index=["真阴性", "真阳性"],
                       columns=["判正常", "判异常"]).to_string())
    tn, fp, fn, tp = cm.ravel()
    print(f"  灵敏度={tp/(tp+fn):.3f} 特异度={tn/(tn+fp):.3f} "
          f"PPV={tp/(tp+fp):.3f} NPV={tn/(tn+fn):.3f}")

    plot_figures(y, oof, X, FEATURES, coef_std, th_youden)
    return {"AUC": auc_oof, "CI": [lo, hi], "threshold": th_youden}


def plot_figures(y, oof, X, features, coef_std, th_youden):
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    fpr, tpr, _ = roc_curve(y, oof)
    prec, rec, _ = precision_recall_curve(y, oof)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, "b-", lw=2,
                 label=f"模型 (AUC={auc(fpr, tpr):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="随机")
    axes[0].set_xlabel("假阳性率 1-特异度")
    axes[0].set_ylabel("真阳性率 灵敏度")
    axes[0].set_title("图1  ROC 曲线 (女胎异常判定)")
    axes[0].legend()
    axes[1].plot(rec, prec, "r-", lw=2)
    axes[1].set_xlabel("召回率 灵敏度")
    axes[1].set_ylabel("精确率 PPV")
    axes[1].set_title("图2  PR 曲线 (类别不平衡视角)")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_roc_pr.png", dpi=150)
    plt.close(fig)

    # 特征重要性(标准化系数绝对值)
    fig, ax = plt.subplots(figsize=(8, 6))
    order = np.argsort(np.abs(coef_std))
    ax.barh(np.array(features)[order], coef_std[order],
            color=np.where(coef_std[order] > 0, "steelblue", "crimson"))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("标准化系数 (蓝=正向, 红=负向)")
    ax.set_title("图3  逻辑回归标准化系数（女胎判定）")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_coef.png", dpi=150)
    plt.close(fig)

    # X浓度 分布 by label
    j = FEATURES.index("xc")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(X[y == 0, j], bins=30, alpha=0.5, label="正常", color="steelblue")
    ax.hist(X[y == 1, j], bins=30, alpha=0.6, label="异常(非整倍体)",
            color="crimson")
    ax.set_xlabel("X染色体浓度 (标准化)"); ax.set_ylabel("人数")
    ax.set_title("图4  X染色体浓度 在正常/异常女胎中的分布")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_xc.png", dpi=150)
    plt.close(fig)
    print(f"\n图表已保存至 {FIG}/ (fig1~fig4)")


if __name__ == "__main__":
    main()
