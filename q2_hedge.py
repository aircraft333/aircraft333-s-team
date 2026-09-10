# -*- coding: utf-8 -*-
"""
问题二改进：分时段自适应安全裕量（滚动分位数）
=====================================================================
原方案：全天统一加一个标量裕量（hedge = 300 kW），对 144 个时段一视同仁。
问题：预测误差的方差在一天内差别很大（中午光伏波动大、深夜几乎为 0），
      统一裕量会在误差小的时段白花钱、在误差大的时段又不够。

更优的理论依据（报童/Newsvendor）：
    在第 t 时段多买 1 kWh 的成本是 π_t，收益是「少付 5π_t 紧急购电费」乘以其发生概率。
    于是「多买」的临界条件是
        5·π_t · P(缺口)  >  π_t   ⟺   P(缺口) > 1/5
    即：最优裕量 = 该时段净负荷预测误差的 80% 分位数。
    注意价格 π_t 在两侧约掉了 —— 最优分位水平与电价无关，只由 5 倍惩罚比决定。

本脚本用滚动经验分位数构造该裕量（严格只用历史误差）：
    x[i,t] = Quantile_{q}{ ε[j,t] : j ∈ 历史窗口 }
    ε[j,t] = (L-G)|_{j,t} − (L-G)^{预测}|_{j,t}
然后快筛 (分位数 q, 窗口长度 W)，再对最优者跑完整 LP 核对。
"""
import numpy as np
import pandas as pd

from config import *
from q2 import build_forecast, solve_day

# ==================== 配置 ====================
SPEC = "wday:7|ma:3"                 # q2_tune 寻优得到的最优预测规格
OUT_START = "2025-02-01"
TXT = "q2_裕量优化.txt"

GRID_Q = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
GRID_WIN = [0, 30, 60, 90, 180]      # 0 = 用全部历史
BASE_HEDGE = 300.0                   # 原方案的标量裕量，作对照


def rolling_quantile(err, q, win):
    """滚动经验分位数：X[i] = Quantile_q({err[j] : j ∈ [i-win, i)})，只用历史

    err : (D, T) 逐时段的净负荷预测误差
    win : 回看窗口天数；0 表示用全部历史
    返回 X : (D, T) 逐时段裕量 (kW)
    """
    D, T = err.shape
    X = np.zeros((D, T))
    for i in range(D):
        s = 0 if win == 0 else max(0, i - win)
        h = err[s:i]
        if h.shape[0]:
            X[i] = np.quantile(h, q, axis=0)
    return X


def daily_sums(err, x, pi, net_real, net_pred, mask):
    """算代理指标：计划购电费与紧急购电费上界（日均）"""
    plan = np.maximum(0.0, net_pred + x)
    deficit = np.maximum(0.0, net_real - net_pred - x)
    c_plan = np.sum(pi[None, :] * plan * DT_H) / mask.sum()
    c_emg = np.sum(5.0 * pi[None, :] * deficit * DT_H) / mask.sum()
    return c_plan, c_emg


def main():
    A1 = att1_arrays(load_att1())
    pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
    dates, L, G = load_att2()
    net_real = L - G
    F_L, F_G = build_forecast(L, G, L1, G1, SPEC)
    net_pred = F_L - F_G
    err = net_real - net_pred                      # (D, T) 净负荷预测误差
    mask = np.asarray(dates >= pd.Timestamp(OUT_START))
    ev = err[mask]

    rule("【1】误差的时段分布特征")
    log(f"预测规格 = {SPEC}；评价区间 {OUT_START} 起 {int(mask.sum())} 天")
    log("净负荷预测误差按 4 小时时段的统计（kW）：")
    log(f"{'时段':<12s} {'均值':>10s} {'标准差':>10s} {'80%分位':>10s} {'95%分位':>10s}")
    for k, (s0, s1) in enumerate([(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]):
        blk = ev[:, s0:s1].ravel()
        log(f"{['0-4','4-8','8-12','12-16','16-20','20-24'][k]:<12s} "
            f"{blk.mean():>10.1f} {blk.std():>10.1f} "
            f"{np.quantile(blk, 0.8):>10.1f} {np.quantile(blk, 0.95):>10.1f}")
    log(f"\n全天统一标量裕量只能取一个数，而各时段 80% 分位从 "
        f"{min(np.quantile(ev[:, s0:s1], 0.8).mean() for s0, s1 in [(0,24),(24,48),(48,72),(72,96),(96,120),(120,144)]):.0f} "
        f"到 {max(np.quantile(ev[:, s0:s1], 0.8).mean() for s0, s1 in [(0,24),(24,48),(48,72),(72,96),(96,120),(120,144)]):.0f} kW 不等")

    rule("【2】快筛：分位数 × 回看窗口（代理指标，日均元/天）")
    rows = []
    for win in GRID_WIN:
        for q in GRID_Q:
            x = rolling_quantile(err, q, win)
            c_plan, c_emg = daily_sums(err, x, pi, net_real, net_pred, mask)
            rows.append({"win": win, "q": q, "plan": c_plan, "emg": c_emg,
                         "total": c_plan + c_emg,
                         "x_mean": float(x[mask].mean()), "x_max": float(x[mask].max())})
    x0 = np.full_like(net_pred, BASE_HEDGE)
    b_plan, b_emg = daily_sums(err, x0, pi, net_real, net_pred, mask)
    log(f"对照：原标量裕量 {BASE_HEDGE:g} kW → 计划 {b_plan:,.0f} + 紧急 {b_emg:,.0f} "
        f"= {b_plan + b_emg:,.0f} 元/天\n")
    rows.sort(key=lambda r: r["total"])
    log(f"{'排名':>4s} {'窗口':>6s} {'分位':>6s} {'裕量均值kW':>11s} {'裕量最大kW':>11s} "
        f"{'计划购电费':>11s} {'紧急购电费':>11s} {'合计':>11s} {'较原方案':>10s}")
    for k, r in enumerate(rows[:15], 1):
        log(f"{k:>4d} {r['win']:>6d} {r['q']:>6.2f} {r['x_mean']:>11.0f} {r['x_max']:>11.0f} "
            f"{r['plan']:>11,.0f} {r['emg']:>11,.0f} {r['total']:>11,.0f} "
            f"{r['total'] - (b_plan + b_emg):>+10,.0f}")

    best = rows[0]
    log(f"\n最优：回看窗口 {best['win']} 天（0=全部历史），分位数 q = {best['q']:.2f}，"
        f"裕量均值 {best['x_mean']:.0f} kW、最大 {best['x_max']:.0f} kW")
    write_report(TXT)


if __name__ == "__main__":
    main()
