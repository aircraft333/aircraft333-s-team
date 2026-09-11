# -*- coding: utf-8 -*-
"""
2026 高教社杯 C 题 · 问题二：全年每天 0:00 制定计划购电策略
=====================================================================
题目要点
    · 每天的电价相同（用附件1 的固定分时电价），小区负载和光伏随时间变化（附件2，365 天）
    · 微网 0:00 制定当天 144 个 10 分钟时段的计划购电策略
    · 微网提供的电能不可低于小区负载，若低于负载需紧急购电，电价为交易时刻电价的 5 倍
    · 除紧急购电费用外，其他购电费用均按计划购电量计算
    · 储能 0:00 初值逐日连续（题目未要求每天首尾电量相同），1/1 0:00 为 6000 kWh
    · 结果写入 result2.xlsx（计划购电量 / 充放电量 / 紧急购电量 三个工作表）

为什么会出现紧急购电
    0:00 制定计划时只能依据「日前信息」，而实际运行值是附件2 的当天实际值。
    两组数据之间的偏差就是缺口的唯一来源。数据事实（见 README/论文）：
        附件1 的负载 ≡ 附件2 的列均值（最大绝对差 5e-5）
        附件1 的光伏 ≡ 附件2 的列均值（最大绝对差 0.04）
        附件1 的电价 ≡ 附件4 的列均值（最大绝对差 5e-5）
    → 附件1 就是「基准日 / 典型日」曲线；其光伏列名为「光伏发电预测功率」，
      附件2 的光伏列名为「光伏发电实际功率」，正好构成「预测 vs 实际」。

计划依据与可调参数（命令行：python q2.py [method] [param] [hedge]）
    核心假设：每天 0:00 制定计划时，只能掌握「历史数据」（1…d−1 天），
              当天的负载和光伏尚未发生、不可知，必须用预测代替。

    method（日前预测方法，均严格只用历史）：
        ma       : 前 n 天移动平均，n = param（n=1 即持续性预测）      ← 默认
        ewma     : 指数加权移动平均，α = param（α→1 退化为持续性）
        wday     : 同星期预测，lag = param（默认 7，取上周同一天）
        att1_pv  : 负载用当天实际值、光伏用附件1 基准日（对照）
        att1_all : 负载与光伏都用附件1 基准日（气候态平均，精度最差对照）
        perfect  : 直接用当天实际值（完美预测，紧急购电恒为 0，理论下界）
    param：上述方法各自的参数
    hedge：安全裕量（kW），加到负载预测上。因紧急购电电价为 5 倍，
           计划应「多买一点」对冲缺口风险（报童模型：缺口概率 > 1/5 时值得提前多买）。
           参数寻优见 q2_tune.py。

时间口径
    第 t 个时段（t = 0…143）覆盖 [10t, 10t+10] 分钟，t=0 即 0:00-0:10，
    t=143 即 23:50-24:00（与附件2 时间戳 T 对应的区间 [T-10min, T] 一致）。
"""
import sys

import openpyxl
import pulp

from config import *

# ==================== 运行配置（可由命令行覆盖） ====================
#   python q2.py [spec] [hedge]
#       spec  : 日前预测规格（严格只用历史数据），见 build_forecast 注释
#               "ma:1"            持续性预测（默认）
#               "ma:7"            前 7 天移动平均
#               "wday:7"          上周同一天
#               "wday:7|ma:1"     负载用 wday:7、光伏用 ma:1
#               "att1:0"          附件1 基准日曲线
#               "perfect"         当天实际值（理论下界，仅作对照）
#       hedge : 安全裕量 (kW)，加到负载预测上以对冲 5 倍价紧急购电
#               带 win 参数时退为分位数模式，详见 build_hedge
#
# 默认值 = 由 q2_tune.py 参数寻优得到的最优方案：
#     负载用上周同一天（周周期强）、光伏用前 3 天均值
# 不带参数运行会写官方文件名 result2.xlsx；带参数运行写 result2_<tag>.xlsx，不相互覆盖。
FC_SPEC = "wday:7|ma:3"

# 安全裕量（对冲 5 倍价紧急购电；报童模型：缺口概率 > 1/k 即值得提前多买，k = 5 → q* = 0.8）
# 实测（当前逐槽被动平衡口径，全年 334 天，见 q2_敏感性分析.txt）：
#      hedge =   0 kW 固定      → 15,228,314.3 元（紧急购电 303.3 万元）
#      hedge = 300 kW 固定      → 14,121,654.0 元（紧急购电  63.3 万元）
#      报童分位 q = 0.80（均值 300 kW）→ 14,001,975.7 元
#      报童分位 q = 0.75（均值 238 kW）→ 13,983,038.5 元   ← 采用，比固定 300 kW 省 13.9 万元
# 分位数曲线在 q ∈ [0.75, 0.80] 上极平坦（仅差 0.14%），且 q=0.80 正是报童解析值，
# 说明采用分位数规则是稳健的、而非对评价年份的过拟合。
# 裕量是当前口径下最大的单一杠杆：逐槽规则下储能被动吸收全部偏差、不会主动留余量，
# 一旦电池提前见底，后续缺口全部按 5π 结算，必须靠「多买一点」把缺口概率压下去。
# 注：q2_参数寻优.txt 是旧口径（储能按计划执行）下跑的，结论不能直接沿用。
HEDGE_MODE = "quantile"        # "scalar"：全天统一值；"quantile"：逐时段取历史误差分位数
HEDGE_PARAM = 0.75              # scalar -> 裕量 kW；quantile -> 分位数水平
HEDGE_WIN = 60                  # quantile 模式的回看窗口天数（0 = 全部历史）

USE_RT = True                    # True: 储能在日内实时再调度（第二阶段）
                                 # False: 储能严格按计划充放电，偏差全由紧急购电填

OUT_START = "2025-02-01"          # 结果输出起始日期（模板要求 2025.2.1-12.31）
TXT_Q2 = "q2_结果汇总.txt"
KEY_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]   # 题目指定日期

IDX4H = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]  # 6 个 4 小时时段
LAB4H = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"]
IDX_REP = [60, 72, 84, 96, 108, 120]                                    # 表1 的 6 个代表时段
LAB_REP = ["10:00-10:10", "12:00-12:10", "14:00-14:10", "16:00-16:10", "18:00-18:10", "20:00-20:10"]


# ==================== 单日计划 LP ====================
def solve_day(pi, L_plan, G_plan, E_start):
    """给定电价、计划依据的负载/光伏、储能当日初值，求最优计划

        min  Σ_t π_t · b_t · Δt
        s.t. b_t + G_t + d_t − c_t − s_t = L_t              (功率平衡)
             0 ≤ b_t,  0 ≤ c_t, d_t ≤ P̄,  s_t ≥ 0           (变量范围)
             E_t = E_{t−1} + (η·c_t − d_t/η)·Δt               (储电量状态转移)
             E_min ≤ E_t ≤ E_max                             (SOC 安全区间)

    说明：不使用充放电互斥 0-1 变量。因 η<1，同时充放电必然造成 η² 的净损耗，
          可证存在 c_t·d_t = 0 的最优解，故 LP 松弛不损失最优值，且规模小、求解快。

    返回 dict(b, c, d, s, E)，单位 kW / kWh，长度 T=144。
    """
    T = N_SLOT
    m = pulp.LpProblem("q2_day", pulp.LpMinimize)
    b = [pulp.LpVariable(f"b{t}", lowBound=0) for t in range(T)]
    c = [pulp.LpVariable(f"c{t}", lowBound=0, upBound=P_RATE) for t in range(T)]
    d = [pulp.LpVariable(f"d{t}", lowBound=0, upBound=P_RATE) for t in range(T)]
    s = [pulp.LpVariable(f"s{t}", lowBound=0) for t in range(T)]
    E = [pulp.LpVariable(f"E{t}", lowBound=SOC_MIN, upBound=SOC_MAX) for t in range(T)]

    m += pulp.lpSum(pi[t] * b[t] * DT_H for t in range(T))
    for t in range(T):
        m += b[t] + G_plan[t] + d[t] - c[t] - s[t] == L_plan[t]
        prev = E_start if t == 0 else E[t - 1]
        m += E[t] == prev + (ETA * c[t] - d[t] / ETA) * DT_H

    m.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[m.status] != "Optimal":
        raise RuntimeError(f"单日计划求解失败：{pulp.LpStatus[m.status]}")
    return {k: np.array([v.value() for v in arr])
            for k, arr in (("b", b), ("c", c), ("d", d), ("s", s), ("E", E))}


def build_hedge(err, mode="quantile", param=0.85, win=60):
    """构造逐时段安全裕量矩阵 X (D, T)，严格只用历史信息

    理论依据（报童模型）：第 t 时段多买 1 kWh 的成本是 π_t、收益是「少付 5π_t 紧急购电
    费」乘其发生概率，临界条件为 5π_t·P(缺口) > π_t ⟺ P(缺口) > 1/5。
    注意 π_t 在两侧约掉 —— 最优分位水平与电价无关，只由 5 倍惩罚比决定。

    mode="scalar"   : 全天 144 个时段统一取 param kW（旧方案）
    mode="quantile" : 第 i 天第 t 时段取历史误差 ε[j,t] (j ∈ 回看窗口) 的 param 分位数

    err : (D, T) 净负荷预测误差 (L−G)|实 − (L−G)|预
    """
    D, T = err.shape
    if mode != "quantile":
        return np.full((D, T), float(param))
    X = np.zeros((D, T))
    for i in range(D):
        s = 0 if win == 0 else max(0, i - win)
        h = err[s:i]
        if h.shape[0]:
            X[i] = np.quantile(h, param, axis=0)
    return X


def solve_rt(pi, L_real, G_real, b_fixed, E_start):
    """实时再调度（第二阶段）：计划购电量已按计划结算并固定，日内实时调度储能

        min  Σ_t 5·π_t · e_t · Δt
        s.t. e_t ≥ L_t − G_t − b_t − d_t + c_t,   e_t ≥ 0
             0 ≤ c_t, d_t ≤ P̄，SOC 区间与状态转移同第一阶段

    返回 dict(e, c, d, E)，单位 kW / kWh
    """
    T = N_SLOT
    m = pulp.LpProblem("q2_rt", pulp.LpMinimize)
    c = [pulp.LpVariable(f"rc{t}", lowBound=0, upBound=P_RATE) for t in range(T)]
    d = [pulp.LpVariable(f"rd{t}", lowBound=0, upBound=P_RATE) for t in range(T)]
    e = [pulp.LpVariable(f"re{t}", lowBound=0) for t in range(T)]
    E = [pulp.LpVariable(f"rE{t}", lowBound=SOC_MIN, upBound=SOC_MAX) for t in range(T)]
    m += pulp.lpSum(5.0 * pi[t] * e[t] * DT_H for t in range(T))
    for t in range(T):
        m += e[t] >= L_real[t] - G_real[t] - b_fixed[t] - d[t] + c[t]
        prev = E_start if t == 0 else E[t - 1]
        m += E[t] == prev + (ETA * c[t] - d[t] / ETA) * DT_H
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[m.status] != "Optimal":
        raise RuntimeError(f"实时再调度求解失败：{pulp.LpStatus[m.status]}")
    return {k: np.array([v.value() for v in arr])
            for k, arr in (("e", e), ("c", c), ("d", d), ("E", E))}


def seg_range(mask):
    """把布尔序列合并成连续区间（第 t 个时段覆盖 [10t, 10t+10] 分钟）

    返回 [(起分钟, 止分钟), ...]
    """
    segs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start * 10, i * 10))
            start = None
    if start is not None:
        segs.append((start * 10, len(mask) * 10))
    return segs


def fmt_span(a, b):
    """(起分钟, 止分钟) -> '10:00-10:30'"""
    return f"{fmt(a)}-{fmt(b)}"


def _forecast_one(X, X1, spec):
    """单个序列的日前预测；spec 形如 'ma:1' / 'ewma:0.3' / 'wday:7' / 'att1:0'"""
    meth, _, par = spec.partition(":")
    meth = meth.strip()
    par = par.strip()
    if meth == "att1":                       # 附件1 基准日（气候态平均）
        return np.tile(X1, (X.shape[0], 1))
    if meth == "ewma":
        return fc_ewma(X, float(par) if par else 0.3, X1)
    if meth == "wday":
        return fc_wday(X, int(float(par)) if par else 7, X1)
    return fc_ma(X, max(1, int(float(par)) if par else 1), X1)      # ma（默认）


def build_forecast(L, G, L1, G1, spec="ma:1"):
    """按 spec 构建 (D, T) 的日前预测矩阵（严格只用历史数据）

    spec 语法：
        "ma:1"            负载与光伏都用前 1 天（持续性预测）
        "ma:7"            前 7 天移动平均
        "wday:7"          取上周同一天
        "wday:7|ma:1"     竖线分隔：左边负载、右边光伏，可用不同方法
        "att1:0"          附件1 基准日曲线（气候态平均）
        "perfect"         直接用当天实际值（理论下界，不参与寻优排序）
    返回 (F_L, F_G)；第 0 天无历史，统一用附件1 基准日兜底。
    """
    s_l, _, s_g = spec.partition("|")
    s_l, s_g = s_l.strip(), (s_g or s_l).strip()
    if s_l == "perfect":
        return L, G
    return _forecast_one(L, L1, s_l), _forecast_one(G, G1, s_g)


# ==================== 主流程 ====================
def main():
    # ---------- 0. 命令行参数 ----------
    global FC_SPEC, HEDGE_MODE, HEDGE_PARAM, HEDGE_WIN, TXT_Q2
    is_default = (len(sys.argv) == 1)            # 无参数 = 主方案，写 result2.xlsx
    if len(sys.argv) > 1:
        FC_SPEC = sys.argv[1]
    if len(sys.argv) > 2:                        # 第 3 参 -> 标量裕量 (kW)
        HEDGE_MODE, HEDGE_PARAM = "scalar", float(sys.argv[2])
    if len(sys.argv) > 3:                        # 第 4 参 -> 分位数 + 回看窗口
        HEDGE_MODE, HEDGE_PARAM, HEDGE_WIN = "quantile", float(sys.argv[2]), int(sys.argv[3])
    tag = FC_SPEC.replace(":", "").replace("|", "-")
    tag += (f"_h{HEDGE_PARAM:g}" if HEDGE_MODE == "scalar"
            else f"_q{HEDGE_PARAM:g}w{HEDGE_WIN}")
    tag += "_rt" if USE_RT else "_plan"          # 储能口径：实时平衡 / 按计划执行
    TXT_Q2 = f"q2_结果汇总_{tag}.txt"

    # ---------- 1. 数据 ----------
    A1 = att1_arrays(load_att1())
    pi, L1, G1 = A1["price"], A1["load"], A1["pv"]      # 附件1：电价 / 负载 / 光伏预测
    dates, L, G = load_att2()                            # 附件2：365 天实际负载与实际光伏
    F_L, F_G = build_forecast(L, G, L1, G1, FC_SPEC)
    err = (L - G) - (F_L - F_G)                          # 净负荷预测误差（仅用于构造裕量）
    X = build_hedge(err, HEDGE_MODE, HEDGE_PARAM, HEDGE_WIN)

    rule("【1】数据与假设")
    log("关键假设：每天 0:00 制定计划时只能掌握历史数据（1…d−1 天），当天数据未知")
    log(f"日前预测规格 = {FC_SPEC}；电价采用附件1 的固定分时电价曲线")
    log("购电量在 0:00 定死后不可更改；储能在日内实时再调度以吸收预测偏差"
        if USE_RT else "购电量与储能充放电均在 0:00 定死，偏差全部由紧急购电填补")
    if HEDGE_MODE == "scalar":
        log(f"安全裕量 = 全天统一 {HEDGE_PARAM:g} kW")
    else:
        log(f"安全裕量 = 逐时段取历史误差的 {HEDGE_PARAM:.2f} 分位数"
            f"（回看 {HEDGE_WIN} 天；日均裕量 {X.mean():.0f} kW，最大 {X.max():.0f} kW）")
    log(f"电价：采用附件1 的分时电价曲线（每天相同），日均 {pi.mean():.4f} 元/kWh，"
        f"峰谷 {pi.min():.4f} ~ {pi.max():.4f}")
    log(f"储能：容量 {E_CAP:.0f} kWh，SOC {SOC_MIN:.0f}~{SOC_MAX:.0f} kWh，"
        f"功率 {P_RATE:.0f} kW，效率 {ETA:.0%}，起始 {SOC0:.0f} kWh")
    log(f"天数：{len(dates)} 天（{dates[0].date()} ~ {dates[-1].date()}），"
        f"逐日储能初值连续；结果输出 {OUT_START} 起")

    # ---------- 2. 逐日：制定计划 → 实际执行 → 紧急购电 ----------
    rule("【2】逐日求解")
    E_start = SOC0
    recs = []
    for i, day in enumerate(dates):
        L_plan = F_L[i] + X[i]           # 负载预测 + 逐时段安全裕量
        G_plan = F_G[i]
        plan = solve_day(pi, L_plan, G_plan, E_start)

        if USE_RT:
            # 实际执行：购电量已按计划结算并固定，储能按逐槽物理必然规则实时平衡
            sim = simulate_dispatch(L[i], G[i], plan["b"], E_start)
            deficit = sim["e"]
            E_next = sim["E"][-1]
            q_ch, q_dis, E_traj = sim["c"] * DT_H, sim["d"] * DT_H, sim["E"]
        else:
            # 实际执行：储能严格按计划充放电，偏差全部由紧急购电填补
            supply = G[i] + plan["b"] + plan["d"] - plan["c"]
            deficit = np.maximum(0.0, L[i] - supply)             # kW
            E_next = plan["E"][-1]
            q_ch, q_dis, E_traj = plan["c"] * DT_H, plan["d"] * DT_H, plan["E"]

        rec = {
            "date": day, "E0": E_start, "E24": E_next,
            "q_buy": plan["b"] * DT_H, "q_ch": q_ch, "q_dis": q_dis,
            "E": E_traj, "q_emg": deficit * DT_H,
            "cost": float(np.sum(pi * plan["b"] * DT_H)),
            "cost_emg": float(np.sum(5.0 * pi * deficit * DT_H)),
        }
        recs.append(rec)
        E_start = E_next
        if (i + 1) % 60 == 0:
            log(f"    已完成 {i + 1}/{len(dates)} 天 …")

    # ---------- 3. 汇总 ----------
    out = [r for r in recs if r["date"] >= pd.Timestamp(OUT_START)]
    tot_buy = sum(r["q_buy"].sum() for r in out)
    tot_cost = sum(r["cost"] for r in out)
    tot_emg = sum(r["q_emg"].sum() for r in out)
    tot_cost_emg = sum(r["cost_emg"] for r in out)
    n_emg_days = sum(1 for r in out if r["q_emg"].sum() > 1e-6)

    rule("【3】全年结果汇总（2025.2.1 - 12.31）")
    log(f"天数 = {len(out)}；计划购电量 = {tot_buy:,.1f} kWh；计划购电费 = {tot_cost:,.1f} 元")
    log(f"紧急购电量 = {tot_emg:,.1f} kWh（占计划购电量 {tot_emg / tot_buy:.2%}）；"
        f"紧急购电费 = {tot_cost_emg:,.1f} 元")
    log(f"总购电费 = {tot_cost + tot_cost_emg:,.1f} 元（紧急购电占 "
        f"{tot_cost_emg / (tot_cost + tot_cost_emg):.1%}）")
    log(f"出现紧急购电的天数 = {n_emg_days}/{len(out)} 天")
    log(f"期末（12/31 24:00）储电量 = {out[-1]['E24']:,.1f} kWh")

    # ---------- 4. 题目指定日期 ----------
    rule("【4】题目指定日期的结果（表1 / 表2 / 表3 格式）")
    for ds in KEY_DATES:
        r = next((x for x in recs if x["date"] == pd.Timestamp(ds)), None)
        if r is None:
            continue
        log(f"\n—— {ds} ——")
        log("  表1 微网购电量")
        log("      时间段           购电量(kWh)      时间段           购电量(kWh)")
        for j in range(0, 6, 2):
            a = f"{LAB_REP[j]:<14s} {r['q_buy'][IDX_REP[j]]:>12.2f}"
            b = f"{LAB_REP[j + 1]:<14s} {r['q_buy'][IDX_REP[j + 1]]:>12.2f}"
            log(f"      {a}      {b}")
        log(f"      全天购电量 {r['q_buy'].sum():,.2f} kWh    全天购电费 {r['cost']:,.2f} 元")
        log("  表2 储能充放电量")
        for k, (s0, s1) in enumerate(IDX4H):
            log(f"      {LAB4H[k]:<12s} 充电 {r['q_ch'][s0:s1].sum():>10.2f} kWh   "
                f"放电 {r['q_dis'][s0:s1].sum():>10.2f} kWh")
        log(f"      0:00 储电量 {r['E0']:,.2f} kWh    24:00 储电量 {r['E24']:,.2f} kWh")
        log("  表3 紧急购电")
        segs = seg_range(r["q_emg"] > 1e-6)
        if not segs:
            log("      无紧急购电")
        else:
            for a, b in segs:
                q = r["q_emg"][a // 10:b // 10].sum()
                log(f"      {fmt_span(a, b):<16s} {q:>10.2f} kWh")

    write_report(TXT_Q2)

    # ---------- 5. 写入结果文件 ----------
    # 主方案（att1_pv）写官方文件名 result2.xlsx，对照口径加后缀以免互相覆盖
    result_path = resolve(OUT2 if is_default else f"result2_{tag}.xlsx")
    wb = openpyxl.load_workbook(resolve(TPL2))

    # --- 5.1 计划购电量：334 天 × 144 时段 + 全天购电量 / 全天购电费 ---
    ws = wb["计划购电量"]
    for i, r in enumerate(out):
        row = i + 2
        for t in range(N_SLOT):
            ws.cell(row=row, column=2 + t, value=round(float(r["q_buy"][t]), 4))
        ws.cell(row=row, column=146, value=round(float(r["q_buy"].sum()), 4))
        ws.cell(row=row, column=147, value=round(float(r["cost"]), 4))

    # --- 5.2 充放电量：每天 6 个 4 小时时段 + 0:00 / 24:00 储电量 ---
    ws = wb["充放电量"]
    for i, r in enumerate(out):
        for k, (s0, s1) in enumerate(IDX4H):
            row = 2 + i * 6 + k
            if k == 0:
                ws.cell(row=row, column=1, value=r["date"].to_pydatetime())
            ws.cell(row=row, column=2, value=LAB4H[k])
            ws.cell(row=row, column=3, value=round(float(r["q_ch"][s0:s1].sum()), 4))
            ws.cell(row=row, column=4, value=round(float(r["q_dis"][s0:s1].sum()), 4))
            if k < 2:
                ws.cell(row=row, column=5, value="00:00" if k == 0 else "24:00")
                ws.cell(row=row, column=6,
                        value=round(float(r["E0"] if k == 0 else r["E24"]), 4))

    # --- 5.3 紧急购电量：日期 / 购电时间段 / 购电量 ---
    ws = wb["紧急购电量"]
    row = 2
    for r in out:
        segs = seg_range(r["q_emg"] > 1e-6)
        if not segs:                       # 当天无紧急购电，仅写日期
            ws.cell(row=row, column=1, value=r["date"].to_pydatetime())
            row += 1
            continue
        for j, (a, b) in enumerate(segs):
            if j == 0:
                ws.cell(row=row, column=1, value=r["date"].to_pydatetime())
            ws.cell(row=row, column=2, value=fmt_span(a, b))
            ws.cell(row=row, column=3, value=round(float(r["q_emg"][a // 10:b // 10].sum()), 4))
            row += 1

    wb.save(result_path)
    print(f"\n结果已写入 {result_path}（{TXT_Q2} 为文字汇总）")

if __name__ == "__main__":
    main()
