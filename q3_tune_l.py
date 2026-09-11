# -*- coding: utf-8 -*-
"""
q3.py 超参数自动网格搜索脚本 (PV_MIX × PV_HIST_N)
=====================================================================
- 搜索区间:
    PV_MIX:    [0.2, 0.7], 步长 0.1
    PV_HIST_N: [2, 7],     步长 1
- 特性:
    1. 完全不改动 q3.py 任何源码；
    2. 一次性加载数据，复用底层预报与 LP 求解器；
    3. 自动计算 36 组参数下的考核期总电费并寻找最优解；
    4. 导出 tune_results.csv，打印热力图矩阵，并生成最优结果表格。

⚠ 【口径必须对齐】否则搜出来的最优与 result3.xlsx 不可比：
    · 负载自适应  adapt = q3.ADAPT（= 0.2）—— 旧版本写死了 adapt=0.0，已修正；
    · 安全裕量    按 q3.HEDGE_MODE 自动选：quantile -> 逐日分位数矩阵；
                  scalar -> 全天固定 q3.HEDGE kW；
    · 决策时刻与汇总区间同 q3.py（2/1 起 334 天）。

⏱ 【耗时可观】36 组 × 全年 365 天 × 0.32 s ≈ 70 分钟。
  若只需要粗排，可把 mix_candidates 改成 [0.3, 0.4, 0.5] 并把 hist_candidates 改成 [2, 3, 4]。
"""

import os
import time
import numpy as np
import pandas as pd

# 导入原程序与配置
import q3
from config import *

# ---- 运行控制 ----
# DAYS=None -> 跑全部 365 天（约 70 分钟 / 36 组）
# DAYS=(60, 190) -> 【粗筛模式】只跑这 130 天（约 24 分钟），
#                  先用热力图排出前几名，再把上面两个候选列表缩小、DAYS 改回 None 跑全年
DAYS = None

def run_single_year(df3, dates, L, G, pi, L1, mix, hist_n, Xh=None):
    """封装单次全年的闭环滚动求解流程

    Xh : (D, 144) 的逐日安全裕量矩阵（由 q3.build_hedge_q3 给出）；
         None 时退回 q3.HEDGE 标量（仅用于旧口径复现）
    """
    decide_h = q3.DECIDE_H
    D = len(dates)

    # 1. 构造光伏和负荷的逐槽预报矩阵 (len(decide_h), D, 144)
    pv_fc = q3.build_pv_forecast(
        df3, dates, G, decide_h=decide_h, mix=mix, mix_hist=hist_n
    )
    load_fc = q3.build_load_forecast(
        L, L1, dates, decide_h=decide_h, lag=q3.LOAD_LAG, adapt=q3.ADAPT
    )

    # 2. 逐日滚动求解
    recs = []
    current_E0 = SOC0  # 初始储电量 6000.0 kWh

    for d in range(D):
        # 提取当日各决策时刻的预报
        Lf_day = load_fc[:, d, :]
        Gf_day = pv_fc[:, d, :]

        # 调用 q3 单日求解器
        res_day = q3.run_day(
            pi=pi,
            Lf_all=Lf_day,
            Gf_all=Gf_day,
            L=L[d],
            G=G[d],
            E_start=current_E0,
            decide_h=decide_h,
            hedge=(Xh[d] if Xh is not None else q3.HEDGE),
        )
        res_day["date"] = pd.Timestamp(dates[d])
        recs.append(res_day)

        # 状态递推：次日 0:00 储电量接续当日 24:00 实际储电量
        current_E0 = res_day["E24"]

    # 3. 统计汇总考核期 (2025.2.1 - 2025.12.31) 结果
    #    注意：q3.summarize 的 cost_total = cost_dev + cost_emg
    #    （cost_dev 已含计划部分，不能再加 cost_plan，否则重复计入）
    summary = q3.summarize(recs, out_start=q3.OUT_START)
    return recs, summary

def run_tuning():
    print("=" * 72)
    print("      启动 q3.py 超参数网格搜索 (PV_MIX × PV_HIST_N)")
    print("=" * 72)

    # 设定网格搜索参数空间
    mix_candidates = [round(x, 1) for x in np.arange(0.2, 0.71, 0.1)]
    hist_candidates = list(range(2, 8))
    total_runs = len(mix_candidates) * len(hist_candidates)

    print(f"搜索空间: PV_MIX    = {mix_candidates}")
    print(f"          PV_HIST_N = {hist_candidates}")
    print(f"共计待评估组合数: {total_runs} 组\n")

    # 一次性预加载数据（大幅加快后续速度）
    print("[1/3] 正在加载基础数据...")
    A1 = att1_arrays(load_att1())
    pi, L1 = A1["price"], A1["load"]
    dates, L, G = load_att2()
    df3 = load_att3()
    ACC = int(np.argmax(np.asarray(dates >= pd.Timestamp(q3.OUT_START))))
    print(f"      数据 {len(dates)} 天；全年评价区间 {q3.OUT_START} 起 "
          f"{len(dates) - ACC} 天")
    if DAYS is not None:                     # 粗筛模式：只跑一个区块
        d0, d1 = DAYS
        print(f"      【粗筛模式】 只跑 {dates[d0].date()} ~ {dates[d1 - 1].date()}"
              f"（{d1 - d0} 天）—— 先用它排序，再对前几名把 DAYS 改回 None 跑全年")
        dates, L, G = dates[d0:d1], L[d0:d1], G[d0:d1]
        ACC = int(np.argmax(np.asarray(dates >= pd.Timestamp(q3.OUT_START))))

    # ---------- 安全裕量：与 q3.py 同口径（跑 36 次，但只算一次）----------
    if q3.HEDGE_MODE == "quantile":
        Xh = q3.build_hedge_q3(L, L1, dates)
        hedge_desc = (f"逐时段取历史 {q3.HEDGE_WIN} 天负载预报误差的 "
                      f"{q3.HEDGE_PARAM:.2f} 分位数")
    else:
        Xh = np.full((len(dates), N_SLOT), float(q3.HEDGE))
        hedge_desc = f"全天固定 {q3.HEDGE:g} kW"
    print(f"【口径】负载自适应 adapt={q3.ADAPT:g}；安全裕量 {hedge_desc}；"
          f"决策时刻 {q3.DECIDE_H}")
    print("        （与 q3.py 主口径一致，结果可直接和 result3.xlsx 对比）")
    print("数据加载完成，开始执行全局优化搜索...\n")

    results = []
    start_time = time.time()
    best_cost = float("inf")
    best_combo = None
    best_recs = None

    run_idx = 0
    print(
        f"{'序号':<4} | {'PV_MIX':<8} | {'HIST_N':<8} | {'总购电费(元)':<14} | {'紧急购电(kWh)':<14} | {'耗时':<6}"
    )
    print("-" * 72)

    for mix in mix_candidates:
        for hist_n in hist_candidates:
            run_idx += 1
            t0 = time.time()

            # 运行单次全年仿真
            recs, summary = run_single_year(
                df3=df3,
                dates=dates,
                L=L,
                G=G,
                pi=pi,
                L1=L1,
                mix=mix,
                hist_n=hist_n,
                Xh=Xh,
            )

            tot_cost = summary["cost_total"]
            emg_kwh = summary["q_emg"]
            elapsed = time.time() - t0

            # 记录当前组合
            record = {
                "PV_MIX": mix,
                "PV_HIST_N": hist_n,
                "cost_total": tot_cost,
                "cost_plan": summary["cost_plan"],
                "cost_dev": summary["cost_dev"],
                "cost_emg": summary["cost_emg"],
                "q_emg": emg_kwh,
                "n_emg_days": summary["n_emg_days"],
                "q_adj": summary["q_adj"],
            }
            results.append(record)

            # 更新最优标记
            is_best = False
            if tot_cost < best_cost:
                best_cost = tot_cost
                best_combo = (mix, hist_n)
                best_recs = recs
                is_best = True

            best_flag = " ★" if is_best else ""
            print(
                f"{run_idx:4d} | {mix:<8.1f} | {hist_n:<8d} | {tot_cost:14,.2f} | {emg_kwh:14,.2f} | {elapsed:5.1f}s{best_flag}"
            )

    total_elapsed = time.time() - start_time
    print("-" * 72)
    print(f"网格搜索完成！总耗时: {total_elapsed / 60:.2f} 分钟\n")

    # 4. 汇总为 DataFrame 并保存 CSV
    df_res = pd.DataFrame(results)
    csv_file = "tune_results.csv"
    df_res.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"[2/3] 全量参数评测数据已导出至: {os.path.abspath(csv_file)}")

    # 5. 打印二维费用对比热力网格
    pivot_cost = df_res.pivot(
        index="PV_HIST_N", columns="PV_MIX", values="cost_total"
    )
    print("\n【总购电费用二维对比表 (元)】:")
    print(pivot_cost.map(lambda x: f"{x:,.0f}"))

    pivot_emg = df_res.pivot(
        index="PV_HIST_N", columns="PV_MIX", values="q_emg"
    )
    print("\n【紧急购电量二维对比表 (kWh)】:")
    print(pivot_emg.map(lambda x: f"{x:,.1f}"))

    # 6. 报告全局最优组合
    best_row = df_res.loc[df_res["cost_total"].idxmin()]
    print("\n" + "=" * 72)
    print("                  ★ 全局最优参数组合 ★")
    print("=" * 72)
    print(f"  最优光伏混合权重 PV_MIX    : {best_row['PV_MIX']:.1f}")
    print(f"  最优滑动平均天数 PV_HIST_N : {int(best_row['PV_HIST_N'])} 天")
    print(f"  ---------------------------------------------")
    print(f"  最低考核期总购电费用       : {best_row['cost_total']:,.2f} 元")
    print(f"  计划购电费用               : {best_row['cost_plan']:,.2f} 元")
    print(f"  调整结算费用               : {best_row['cost_dev']:,.2f} 元")
    print(f"  紧急购电费用               : {best_row['cost_emg']:,.2f} 元")
    print(
        f"  累计紧急购电量             : {best_row['q_emg']:,.2f} kWh (共 {int(best_row['n_emg_days'])} 天发生)"
    )
    print("=" * 72)

    # 7. 导出最优解 Excel
    opt_xlsx = "result3_optimal.xlsx"
    print(f"\n[3/3] 正在写出最优参数结果文件至: {opt_xlsx} ...")
    try:
        q3.write_xlsx(best_recs, opt_xlsx)
        print("写入完成！可以直接用于提交或结果分析。")
    except Exception as err:
        print(f"写出 Excel 时提示: {err}（费用与参数已完整保存在 CSV 中）")

if __name__ == "__main__":
    run_tuning()