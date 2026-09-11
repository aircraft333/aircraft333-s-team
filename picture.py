# -*- coding: utf-8 -*-
"""2026 高教社杯 C 题 · 问题四：波动电价下的跨日套利机理与储能容量瓶颈分析

=================================================================================
优化说明：
1. 加深“逐日独立规划（Daily LP）”折线颜色（高对比度深蓝 #0D3B66），加粗线条，确保清晰醒目。
2. 调整坐标轴 Y 轴上限与图例布局（横排多列 + 顶部留白），彻底避免图例遮挡曲线。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp

# ============================ 基础物理参数 ============================
DT_MIN = 10
DT_H = DT_MIN / 60.0  # 1/6 小时
N_SLOT = 144  # 每天 144 个时段
P_RATE = 5000.0  # 最大充放电功率 kW
SOC_MIN = 1200.0  # 最低储电量 kWh
SOC_MAX = 10800.0  # 最高储电量 kWh
SOC0 = 6000.0  # 初始储电量 kWh
ETA = 0.90  # 充放电效率

# ============================ 数据读取函数 ============================
def load_all_data():
    """读取附件1、附件2、附件4 的真实数据"""
    df1 = pd.read_excel("附件1.xlsx")
    L1 = df1.iloc[:, 2].to_numpy(float)
    G1 = df1.iloc[:, 3].to_numpy(float)

    df2_load = pd.read_excel("附件2.xlsx", sheet_name="小区负载")
    df2_pv = pd.read_excel("附件2.xlsx", sheet_name="光伏发电实际功率")
    dates = pd.to_datetime(df2_load.iloc[:, 0])
    L_all = df2_load.iloc[:, 1 : 1 + N_SLOT].to_numpy(float)
    G_all = df2_pv.iloc[:, 1 : 1 + N_SLOT].to_numpy(float)

    df4 = pd.read_excel("附件4.xlsx")
    PI_all = df4.iloc[:, 1 : 1 + N_SLOT].to_numpy(float)

    return dates, L_all, G_all, PI_all, L1, G1

# ============================ 预测与安全裕量（与 q4.py 一致） ============================
def build_forecast_and_hedge(L, G, L1, G1, q=0.76, win=30):
    """日前预测：负荷=上周同期(lag=7)；光伏=前3天均值；裕量=历史误差分位数"""
    D, T = L.shape
    F_L = np.zeros_like(L)
    for i in range(D):
        F_L[i] = L[i - 7] if i >= 7 else L1

    F_G = np.zeros_like(G)
    for i in range(D):
        if i == 0:
            F_G[i] = G1
        else:
            k = min(i, 3)
            F_G[i] = G[i - k : i].mean(axis=0)

    err = (L - G) - (F_L - F_G)
    Hedge = np.zeros_like(err)
    for i in range(D):
        if i < 7:
            Hedge[i] = 0.0
        else:
            hist_start = max(0, i - win)
            Hedge[i] = np.quantile(err[hist_start:i], q, axis=0)

    L_plan = F_L + Hedge
    G_plan = F_G
    return L_plan, G_plan

# ============================ 优化模型求解器 ============================
def solve_multi_day_lp(pi_vec, L_vec, G_vec, E_start, unlimited_capacity=False):
    """多日（如3天）联合 LP 模型"""
    N = len(pi_vec)
    m = pulp.LpProblem("MultiDay_Joint_LP", pulp.LpMinimize)

    b = [pulp.LpVariable(f"b_{t}", lowBound=0) for t in range(N)]
    c = [
        pulp.LpVariable(f"c_{t}", lowBound=0, upBound=P_RATE) for t in range(N)
    ]
    d = [
        pulp.LpVariable(f"d_{t}", lowBound=0, upBound=P_RATE) for t in range(N)
    ]
    s = [pulp.LpVariable(f"s_{t}", lowBound=0) for t in range(N)]

    e_upper = None if unlimited_capacity else SOC_MAX
    E = [
        pulp.LpVariable(f"E_{t}", lowBound=SOC_MIN, upBound=e_upper)
        for t in range(N)
    ]

    # 目标函数：最小化总购电费用
    m += pulp.lpSum(pi_vec[t] * b[t] * DT_H for t in range(N))

    # 约束条件
    for t in range(N):
        m += b[t] + G_vec[t] + d[t] - c[t] - s[t] == L_vec[t]
        prev_E = E_start if t == 0 else E[t - 1]
        m += E[t] == prev_E + (ETA * c[t] - d[t] / ETA) * DT_H

    m.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[m.status] != "Optimal":
        raise RuntimeError("求解未找到最优解")

    b_res = np.array([v.value() for v in b])
    c_res = np.array([v.value() for v in c])
    d_res = np.array([v.value() for v in d])
    E_res = np.array([v.value() for v in E])
    cost_series = pi_vec * b_res * DT_H  # 每个时段的购电费（元）

    return {
        "b": b_res,
        "c": c_res,
        "d": d_res,
        "E": E_res,
        "cost_series": cost_series,
        "cost_total": float(np.sum(cost_series)),
        "E_end": E_res[-1],
    }

def solve_daily_seq(pi_mat, L_mat, G_mat, E_start):
    """逐日规划（每天只看当天 144 时段，初值按天顺延）"""
    days = pi_mat.shape[0]
    b_all, c_all, d_all, E_all, cost_s_all = [], [], [], [], []
    E_curr = E_start

    for d in range(days):
        res = solve_multi_day_lp(
            pi_mat[d],
            L_mat[d],
            G_mat[d],
            E_curr,
            unlimited_capacity=False,
        )
        b_all.append(res["b"])
        c_all.append(res["c"])
        d_all.append(res["d"])
        E_all.append(res["E"])
        cost_s_all.append(res["cost_series"])
        E_curr = res["E_end"]

    cost_concat = np.concatenate(cost_s_all)
    return {
        "b": np.concatenate(b_all),
        "c": np.concatenate(c_all),
        "d": np.concatenate(d_all),
        "E": np.concatenate(E_all),
        "cost_series": cost_concat,
        "cost_total": float(np.sum(cost_concat)),
    }

# ============================ 核心绘图与分析主程序 ============================
def main():
    print(">>> 正在加载附件数据并构建预测体系...")
    dates, L, G, PI, L1, G1 = load_all_data()
    L_plan, G_plan = build_forecast_and_hedge(L, G, L1, G1)

    # 选取最经典的 3 天窗口：2025-04-25 至 2025-04-27
    target_start_date = "2025-04-25"
    win_days = 3
    s_idx = np.where(dates == pd.Timestamp(target_start_date))[0][0]
    e_idx = s_idx + win_days

    span_dates = list(dates[s_idx:e_idx])
    pi_win = PI[s_idx:e_idx]
    L_win = L_plan[s_idx:e_idx]
    G_win = G_plan[s_idx:e_idx]

    pi_flat = pi_win.ravel()
    L_flat = L_win.ravel()
    G_flat = G_win.ravel()

    print(
        f"\n================ 典型跨日分析窗口（{target_start_date} 起 {win_days} 天）================"
    )
    for i, d in enumerate(span_dates):
        print(
            f"  第 {i+1} 天 ({d.strftime('%Y-%m-%d')}): 均价 = {pi_win[i].mean():.4f} 元/kWh, 最低 = {pi_win[i].min():.4f}, 最高 = {pi_win[i].max():.4f}"
        )

    # 1. 模型 A：逐日独立规划 (Daily LP)
    res_daily = solve_daily_seq(pi_win, L_win, G_win, E_start=SOC0)

    # 2. 模型 B：3天联合规划（常规储能容量 1.2k~10.8k kWh）
    res_multi_cap = solve_multi_day_lp(
        pi_flat, L_flat, G_flat, E_start=SOC0, unlimited_capacity=False
    )

    # 3. 模型 C：3天联合规划（解除储能最大容量限制 E_max = 无穷大）
    res_multi_unlim = solve_multi_day_lp(
        pi_flat, L_flat, G_flat, E_start=SOC0, unlimited_capacity=True
    )

    print(
        f"\n================ {win_days} 天规划结果与经济效益对比 ================"
    )
    print(
        f"1. 模型 A（逐日分别规划）        : 总购电费 = {res_daily['cost_total']:,.2f} 元"
    )
    print(
        f"2. 模型 B（跨日联合规划·正常容量）: 总购电费 = {res_multi_cap['cost_total']:,.2f} 元  (相对逐日节省: {res_daily['cost_total'] - res_multi_cap['cost_total']:,.2f} 元, {(res_daily['cost_total'] - res_multi_cap['cost_total'])/res_daily['cost_total']:.3%})"
    )
    print(
        f"3. 模型 C（跨日联合规划·容量无限）: 总购电费 = {res_multi_unlim['cost_total']:,.2f} 元  (相对逐日节省: {res_daily['cost_total'] - res_multi_unlim['cost_total']:,.2f} 元, {(res_daily['cost_total'] - res_multi_unlim['cost_total'])/res_daily['cost_total']:.3%})"
    )

    # ============================ 绘制 4 联对比折线图 ============================
    print("\n>>> 正在绘制购电费用、电价与储能状态对比折线图...")
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
        "Arial",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    time_steps = np.arange(len(pi_flat))  # 3 天共 432 个时段
    cost_daily = res_daily["cost_series"]
    cost_cap = res_multi_cap["cost_series"]
    cost_unlim = res_multi_unlim["cost_series"]

    # 累计购电费
    cum_daily = np.cumsum(cost_daily)
    cum_cap = np.cumsum(cost_cap)
    cum_unlim = np.cumsum(cost_unlim)

    # 调色盘定义：模型 A 加深为深海蓝，提高粗细
    C_DAILY = "#0D3B66"  # 加深加粗深海蓝
    C_CAP = "#2E7D32"  # 墨绿色虚线
    C_UNLIM = "#8E24AA"  # 紫罗兰实线
    C_PRICE = "#C62828"  # 电价深红

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

    # ------------------ 子图 1：实时波动电价 ------------------
    ax1 = axes[0]
    ax1.plot(
        time_steps, pi_flat, color=C_PRICE, lw=1.8, label="实时外网电价 (元/kWh)"
    )
    ax1.set_ylabel("电价\n(元/kWh)", fontsize=11)
    ax1.set_title(
        f"波动电价跨日套利机理与储能容量瓶颈分析（{target_start_date} 起连续 3 天）",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax1.grid(True, linestyle="--", alpha=0.5)
    # 顶部留出空间放置图例
    y_max1 = max(pi_flat)
    ax1.set_ylim(-0.05, y_max1 * 1.30)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=10)

    # ------------------ 子图 2：时段购电费用 ------------------
    ax2 = axes[1]
    # 模型 A 底层深色加粗
    ax2.plot(
        time_steps,
        cost_daily,
        color=C_DAILY,
        lw=2.0,
        alpha=1.0,
        label="模型 A: 逐日独立规划 (Daily LP)",
    )
    ax2.plot(
        time_steps,
        cost_cap,
        color=C_CAP,
        lw=1.5,
        linestyle="--",
        alpha=0.95,
        label="模型 B: 跨日联合规划 (常规容量 1.2k~10.8k kWh)",
    )
    ax2.plot(
        time_steps,
        cost_unlim,
        color=C_UNLIM,
        lw=1.5,
        alpha=0.9,
        label="模型 C: 跨日联合规划 (解除容量上限限制)",
    )
    ax2.set_ylabel("时段购电费\n(元/10min)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    # 扩大 Y 轴上限 38%，使用 3 列水平排布图例，完全不挡波峰
    y_max2 = max(max(cost_daily), max(cost_cap), max(cost_unlim))
    ax2.set_ylim(-20, y_max2 * 1.38)
    ax2.legend(
        loc="upper left",
        ncol=3,
        framealpha=0.92,
        fontsize=9.5,
        columnspacing=1.0,
    )

    # ------------------ 子图 3：累计购电费用 ------------------
    ax3 = axes[2]
    ax3.plot(
        time_steps,
        cum_daily,
        color=C_DAILY,
        lw=2.0,
        label=f"模型 A 累计电费 (终值: {cum_daily[-1]:,.0f}元)",
    )
    ax3.plot(
        time_steps,
        cum_cap,
        color=C_CAP,
        lw=1.6,
        linestyle="--",
        label=f"模型 B 累计电费 (终值: {cum_cap[-1]:,.0f}元)",
    )
    ax3.plot(
        time_steps,
        cum_unlim,
        color=C_UNLIM,
        lw=1.8,
        label=f"模型 C 累计电费 (终值: {cum_unlim[-1]:,.0f}元)",
    )
    ax3.set_ylabel("累计购电费\n(元)", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)
    # 累计电费递增，图例放在左上角空白区
    y_max3 = max(cum_daily[-1], cum_cap[-1], cum_unlim[-1])
    ax3.set_ylim(-1000, y_max3 * 1.25)
    ax3.legend(
        loc="upper left",
        ncol=3,
        framealpha=0.92,
        fontsize=9.5,
        columnspacing=1.0,
    )

    # ------------------ 子图 4：储能电量 SOC 变化 ------------------
    ax4 = axes[3]
    ax4.plot(
        time_steps,
        res_daily["E"],
        color=C_DAILY,
        lw=2.0,
        label="模型 A 储电量 (kWh)",
    )
    ax4.plot(
        time_steps,
        res_multi_cap["E"],
        color=C_CAP,
        lw=1.6,
        linestyle="--",
        label="模型 B 储电量 (kWh)",
    )
    ax4.plot(
        time_steps,
        res_multi_unlim["E"],
        color=C_UNLIM,
        lw=1.8,
        label="模型 C 储电量 (无限容量)",
    )
    ax4.axhline(
        y=SOC_MAX,
        color="#757575",
        linestyle=":",
        lw=1.6,
        label=f"常规储能容量上限 ({SOC_MAX:,.0f} kWh)",
    )
    ax4.set_ylabel("储电量 SOC\n(kWh)", fontsize=11)
    ax4.set_xlabel("时间（按天切分，每天 144 个 10 分钟时段）", fontsize=11)
    ax4.grid(True, linestyle="--", alpha=0.5)
    y_max4 = max(res_multi_unlim["E"].max(), SOC_MAX)
    ax4.set_ylim(-1000, y_max4 * 1.30)
    ax4.legend(
        loc="upper left",
        ncol=4,
        framealpha=0.92,
        fontsize=9.5,
        columnspacing=0.8,
    )

    # 分割线与 X 轴刻度标注
    for day_i in range(win_days):
        tick_pos = day_i * N_SLOT
        for ax in axes:
            ax.axvline(x=tick_pos, color="#424242", linestyle="--", alpha=0.35)

    plt.xticks(
        ticks=[i * N_SLOT + 72 for i in range(win_days)],
        labels=[
            f"{d.strftime('%Y-%m-%d')}\n(均价 {pi_win[i].mean():.3f} 元/kWh)"
            for i, d in enumerate(span_dates)
        ],
        fontsize=10.5,
    )

    plt.tight_layout()
    output_fig = "q4_3天购电费用与储能对比折线图.png"
    plt.savefig(output_fig, dpi=300)
    print(f">>> 对比折线图已成功保存至：{output_fig}")
    plt.show()

if __name__ == "__main__":
    main()