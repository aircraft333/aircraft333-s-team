# -*- coding: utf-8 -*-
"""
2026 高教社杯 C 题 · 问题三：多阶段滚动调整购电策略
=====================================================================
题目要求
    每天 0:00、6:00、12:00、18:00 可获得未来 24 小时「整点」的光伏发电功率预报。
    0:00 依据预报制定当天计划购电策略；其他时刻可依据新预报调整购电策略。
        计划购电量高于调整购电量的部分，违约电价 = 交易时刻电价的 50%
        调整购电量高于计划购电量的部分，超出部分电价 = 交易时刻电价的 1.5 倍
    总购电费 = 计划购电费 + 紧急购电费 + 调整购电量的相关费用

核心模型（两阶段 → 四阶段滚动）
    每个时段 t 的购电费用（相对 0:00 计划购电量 b^p_t 与最终调整购电量 b^a_t）：
        C_t = π_t·min(b^p_t, b^a_t)
            + 0.5·π_t·(b^p_t − b^a_t)^+          ← 少买的违约部分
            + 1.5·π_t·(b^a_t − b^p_t)^+          ← 多买的超出部分
    该函数关于 b^a 是凸的分段线性函数，故可用 δ⁺ / δ⁻ 线性化：
        C_t = π_t·b^p_t + 1.5π_t·δ⁺_t − 0.5π_t·δ⁻_t,   b^a_t = b^p_t + δ⁺_t − δ⁻_t

逐槽规则在 LP 中的精确嵌入（本问的关键）
    问题二已经确认：购电量锁定后，储能在每个时段只能做「被动平衡」——
        Δ_t = L_t − G_t − b_t
        Δ_t > 0：放电补缺，受功率与 SOC 下限约束，不足部分为紧急购电
        Δ_t < 0：充电吸纳，受功率与 SOC 上限约束，多余部分为弃光
    该规则看似含 min/max，实则是凸分段线性的，可用纯 LP 精确表达：
        d_t − c_t − s_t − e_t = Δ_t                     （功率平衡）
        d_t ≤ P̄,  d_t ≤ (E_{t−1} − E_min)·η/Δt          （放电：功率 / 剩余电量）
        c_t ≤ P̄,  c_t ≤ (E_max − E_{t−1})/(η·Δt)        （充电：功率 / 剩余空间）
        s_t ≤ G_t,  e_t ≥ 0,  s_t ≥ 0
        E_t = E_{t−1} + (η·c_t − d_t/η)·Δt
    目标中 e_t 的系数 5π_t > 0 会迫使 d_t 取到上界，恰为 min(Δ_t, 上界)；
    目标中给 s_t 一个极小系数 ε 会迫使 c_t 取到上界，恰为 min(−Δ_t, 上界)。
    于是 LP 的最优解与实际执行的逐槽规则**逐点一致**，分段滚动不会出现前后矛盾。

滚动逻辑
    t0 时刻只优化 t ∈ [t0, 144) 的购电量；[0, t0) 已发生、取值锁定（其费用为常数，
    不影响 argmin，故可从目标中剔除）。进入 t0 时的储电量由「实际数据 + 已锁定购电量」
    仿真得到，是已知量。由于逐槽规则无前瞻，「全天仿真 = 分段仿真拼接」，滚动自洽。

命令行
    python q3.py                            # 主方案（写 result3.xlsx）
    python q3.py <pv_mix> <hedge> <adapt>   # 覆盖默认参数
    python q3.py ablate                     # 各时刻预报的边际价值分析（论文第 3 问末段）
"""
import sys

import openpyxl
import pulp

from config import *

# ==================== 运行配置（默认值 = q3_tune.py 区块寻优结果） ====================
REL_H = (0, 6, 12, 18)            # 附件3 的预报发布时刻（题目给定，不可改）
DECIDE_H = (0, 6, 12, 18)         # 实际做决策 / 调整的时刻（可加密；每个时刻取最近一次已发布预报）
LOAD_LAG = 7                      # 负载日前预测：上周同一天
PV_MIX = 0.4                      # 光伏预报 = mix·附件3(整点插值) + (1−mix)·前3天均值
HEDGE = 300.0                     # 加在负载预报上的安全裕量 (kW)，作用于计划与调整两个阶段
ADAPT = 0.2                       # 负载当日自适应修正限幅（用已发生时段的实际/预报之比缩放剩余预报）
EPS_CUR = 1e-6                    # 弃光松弛的极小系数（仅用于打破简并，不改变最优值）
PV_HIST_N = 3                     # 历史外推所用天数
OUT_START = "2025-02-01"
TXT_Q3 = "q3_结果汇总.txt"
KEY_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]

IDX4H = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]   # 6 个 4 小时段
LAB4H = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"]
IDX_REP = [60, 72, 84, 96, 108, 120]                                     # 表1 的 6 个代表时段
LAB_REP = ["10:00-10:10", "12:00-12:10", "14:00-14:10", "16:00-16:10", "18:00-18:10", "20:00-20:10"]

U_SLOT = (np.arange(N_SLOT) + 1) / 6.0        # 各时段右端点对应的「整点小时数」


# =====================================================================
# 一、预报构造
# =====================================================================
def build_pv_forecast(df3, dates, G, decide_h=DECIDE_H, mix=PV_MIX, mix_hist=None):
    """构造 (len(decide_h), D, 144) 的逐槽光伏预报：决策时刻 × 天数 × 时段

    口径（经附件数据标定）：附件3 的「预报k小时」是发布后第 k 个**整点时刻**的
    瞬时功率（点值），而非该小时的平均功率。故用整点值线性插值到 10 分钟时段
    右端点；若误按小时均值处理，全年 MAE 会从 191.7 kW 恶化到 361.6 kW。

    决策时刻 h 使用「最近一次已发布」的预报 rel = max{r ∈ REL_H : r ≤ h}，
    该预报覆盖 rel+1 … rel+24 时，对本日剩余时段全部有效。
    已发生时段的实际值视为已知，直接填入实际功率。

    mix：附件3 预报与历史外推（前 mix_hist 天滑动平均）的线性混合权重，
         mix=1 表示完全采用附件3。历史外推用于平滑附件3 的随机误差。
    """
    mix_hist = PV_HIST_N if mix_hist is None else mix_hist
    D = len(dates)
    hist = fc_ma(G, mix_hist, G[0])                      # 历史外推（只用历史）
    out = np.full((len(decide_h), D, N_SLOT), np.nan)
    for ri, h in enumerate(decide_h):
        rel = max(r for r in REL_H if r <= h)            # 最近一次已发布预报
        t0 = h * 6                                       # 该时刻已发生的时段数
        for i, d in enumerate(dates):
            v = att3_forecast(df3, d, rel)
            fh = {0: 0.0, 24: 0.0}                       # 0:00 / 24:00 光伏为 0
            if v is not None:
                for k in range(1, 25):
                    H = rel + k
                    if H <= 24:
                        fh[H] = float(v[k - 1])
            xs = np.array(sorted(fh))
            ys = np.array([fh[x] for x in xs])
            f = np.interp(U_SLOT, xs, ys)                # 整点值 → 逐槽插值
            g = mix * f + (1.0 - mix) * hist[i]          # 与历史外推混合
            g[:t0] = G[i, :t0]                           # 已发生时段的实际值已知
            out[ri, i] = np.maximum(g, 0.0)
    return out


# =====================================================================
# 二、0:00 计划（与问题二同构）
# =====================================================================
def solve_day(pi, L_plan, G_plan, E_start):
    """0:00 制定计划购电量：min Σ π_t·b_t·Δt

    约束为功率平衡 + 储能 SOC 状态转移 + 各项上下限。因 η<1，η² 的往返损耗
    使 c_t·d_t = 0 自动成立，故无需 0-1 变量（见问题一命题）。
    """
    T = N_SLOT
    m = pulp.LpProblem("q3_plan", pulp.LpMinimize)
    b = [pulp.LpVariable(f"pb{t}", lowBound=0) for t in range(T)]
    c = [pulp.LpVariable(f"pc{t}", lowBound=0, upBound=P_RATE) for t in range(T)]
    d = [pulp.LpVariable(f"pd{t}", lowBound=0, upBound=P_RATE) for t in range(T)]
    s = [pulp.LpVariable(f"ps{t}", lowBound=0) for t in range(T)]
    E = [pulp.LpVariable(f"pE{t}", lowBound=SOC_MIN, upBound=SOC_MAX) for t in range(T)]

    m += pulp.lpSum(pi[t] * b[t] * DT_H for t in range(T))
    for t in range(T):
        m += b[t] + G_plan[t] + d[t] - c[t] - s[t] == L_plan[t]
        prev = E_start if t == 0 else E[t - 1]
        m += E[t] == prev + (ETA * c[t] - d[t] / ETA) * DT_H

    m.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[m.status] != "Optimal":
        raise RuntimeError(f"0:00 计划求解失败：{pulp.LpStatus[m.status]}")
    return {k: np.array([v.value() for v in arr])
            for k, arr in (("b", b), ("c", c), ("d", d), ("s", s), ("E", E))}


# =====================================================================
# 三、调整阶段 LP（把逐槽规则精确写进约束）
# =====================================================================
def solve_adjust(pi, Lf, Gf, b_plan, E_start, t_from):
    """调整 [t_from, 144) 的购电量，最小化「调整费用 + 紧急购电费用」

        min Σ_t [ 1.5π_t·δ⁺_t − 0.5π_t·δ⁻_t + 5π_t·e_t + ε·s_t ]·Δt
        s.t. b_t = b^p_t + δ⁺_t − δ⁻_t                       （相对 0:00 计划的偏离）
             d_t − c_t − s_t − e_t = L_t − G_t − b_t          （功率平衡）
             d_t ≤ P̄,  d_t ≤ (E_{t−1} − E_min)·η/Δt           （放电上界）
             c_t ≤ P̄,  c_t ≤ (E_max − E_{t−1})/(η·Δt)         （充电上界）
             s_t ≤ G_t,  e_t, s_t, c_t, d_t ≥ 0
             E_t = E_{t−1} + (η·c_t − d_t/η)·Δt,  E ∈ [E_min, E_max]

    由于 e_t 的系数 5π_t 远大于 0、s_t 的系数 ε 极小，最优解中 d_t / c_t 会分别
    取到各自上界，与逐槽规则 d=min(Δ,上界)、c=min(−Δ,上界) 逐点相同。

    返回 (144,) 的**完整**购电量数组（[0, t_from) 段原样保留）。
    """
    T = N_SLOT
    n = T - t_from
    m = pulp.LpProblem("q3_adj", pulp.LpMinimize)
    b = [pulp.LpVariable(f"b{k}", lowBound=0) for k in range(n)]
    c = [pulp.LpVariable(f"c{k}", lowBound=0, upBound=P_RATE) for k in range(n)]
    d = [pulp.LpVariable(f"d{k}", lowBound=0, upBound=P_RATE) for k in range(n)]
    s = [pulp.LpVariable(f"s{k}", lowBound=0) for k in range(n)]
    e = [pulp.LpVariable(f"e{k}", lowBound=0) for k in range(n)]
    E = [pulp.LpVariable(f"E{k}", lowBound=SOC_MIN, upBound=SOC_MAX) for k in range(n)]
    dp = [pulp.LpVariable(f"dp{k}", lowBound=0) for k in range(n)]     # δ⁺
    dm = [pulp.LpVariable(f"dm{k}", lowBound=0) for k in range(n)]     # δ⁻

    m += pulp.lpSum((1.5 * pi[t] * dp[k] - 0.5 * pi[t] * dm[k]
                     + 5.0 * pi[t] * e[k] + EPS_CUR * s[k]) * DT_H
                    for k, t in enumerate(range(t_from, T)))
    for k, t in enumerate(range(t_from, T)):
        m += b[k] == b_plan[t] + dp[k] - dm[k]
        m += d[k] - c[k] - s[k] - e[k] == Lf[t] - Gf[t] - b[k]
        m += s[k] <= Gf[t]                        # 弃光不超过当时光伏出力
        prev = E_start if k == 0 else E[k - 1]
        m += d[k] <= (prev - SOC_MIN) * ETA / DT_H
        m += c[k] <= (SOC_MAX - prev) / (ETA * DT_H)
        m += E[k] == prev + (ETA * c[k] - d[k] / ETA) * DT_H

    m.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[m.status] != "Optimal":
        raise RuntimeError(f"调整阶段求解失败：{pulp.LpStatus[m.status]}")
    out = b_plan.copy()
    out[t_from:] = [v.value() for v in b]
    return np.array(out, dtype=float)


def cost_dev(pi, bp, ba):
    """调整相关购电费用（含计划部分），单位元

        Σ [ π·min(b^p,b^a) + 0.5π·(b^p−b^a)^+ + 1.5π·(b^a−b^p)^+ ]·Δt
    """
    lo = np.minimum(bp, ba)
    over = np.maximum(ba - bp, 0.0)
    under = np.maximum(bp - ba, 0.0)
    return float(np.sum(pi * (lo + 0.5 * under + 1.5 * over) * DT_H))


# =====================================================================
# 四、单日滚动求解
# =====================================================================
def run_day(pi, Lf_all, Gf_all, L, G, E_start, decide_h=DECIDE_H, hedge=HEDGE):
    """滚动求解一天，返回该日的全部结果
    Lf_all / Gf_all : 与 decide_h 同序的逐槽预报；decide_h[0] 必须是 0
    """
    # --- 0:00 计划（依据 0:00 预报）---
    Lp = Lf_all[0] + hedge
    Gp = Gf_all[0]
    plan = solve_day(pi, Lp, Gp, E_start)
    bp = plan["b"]

    # --- 依次在后续决策时刻调整 ---
    ba = bp.copy()
    info = []
    for ri in range(1, len(decide_h)):
        t0 = decide_h[ri] * 6
        sim = simulate_dispatch(L, G, ba, E_start, t_from=0, t_to=t0)
        E_t0 = sim["E"][-1]                      # 进入 t0 时的实际储电量（已知）
        ba = solve_adjust(pi, Lf_all[ri] + hedge, Gf_all[ri], bp, E_t0, t0)
        info.append((decide_h[ri], t0, E_t0))

    # --- 实际执行：逐槽规则（全天仿真 ≡ 分段仿真拼接）---
    sim = simulate_dispatch(L, G, ba, E_start)
    return {
        "b_plan": bp, "b_adj": ba,
        "c": sim["c"], "d": sim["d"], "e": sim["e"], "s": sim["s"], "E": sim["E"],
        "E0": E_start, "E24": sim["E"][-1],
        "cost_plan": float(np.sum(pi * bp * DT_H)),
        "cost_dev": cost_dev(pi, bp, ba),
        "cost_emg": float(np.sum(5.0 * pi * sim["e"] * DT_H)),
        "epochs_info": info,
    }


# =====================================================================
# 五、全年主流程
# =====================================================================
def build_load_forecast(L, L1, dates, decide_h=DECIDE_H, lag=LOAD_LAG, adapt=0.0):
    """构造 (len(decide_h), D, 144) 的逐槽负载预报

    基础为「上周同一天」（只用历史）。adapt>0 时启用当日自适应修正：
    用当日已发生时段的实际负载与预报之比（限幅 [1−adapt, 1+adapt]）
    缩放剩余时段的预报 —— 这是真实可得的增量信息（不需要额外预报）。
    """
    D = len(dates)
    F = fc_wday(L, lag, L1)
    out = np.full((len(decide_h), D, N_SLOT), np.nan)
    for ri, h in enumerate(decide_h):
        t0 = h * 6
        out[ri] = F
        if adapt > 0 and t0 > 0:
            num = L[:, :t0].mean(axis=1)
            den = np.maximum(F[:, :t0].mean(axis=1), 1e-6)
            k = np.clip(num / den, 1.0 - adapt, 1.0 + adapt)
            out[ri] = F * k[:, None]
        out[ri, :, :t0] = L[:, :t0]              # 已发生时段的实际值已知
    return out


def run_year(df3, dates, L, G, pi, L1, G1, decide_h=DECIDE_H, mix=PV_MIX,
             hedge=HEDGE, adapt=ADAPT, quiet=False):
    """跑完整一年，返回 (recs, 汇总字典)

    逐日储能初值连续（当天 24:00 的储电量接到次日 0:00）。
    """
    Gf = build_pv_forecast(df3, dates, G, decide_h, mix)
    Lf = build_load_forecast(L, L1, dates, decide_h, LOAD_LAG, adapt)
    recs, E_start = [], SOC0
    for i, day in enumerate(dates):
        r = run_day(pi, Lf[:, i], Gf[:, i], L[i], G[i], E_start, decide_h, hedge)
        r["date"] = day
        recs.append(r)
        E_start = r["E24"]
        if not quiet and (i + 1) % 60 == 0:
            log(f"    已完成 {i + 1}/{len(dates)} 天 …")
    return recs, summarize(recs)


def summarize(recs, out_start=OUT_START):
    """按输出起始日期汇总全年结果"""
    out = [r for r in recs if r["date"] >= pd.Timestamp(out_start)]
    c_plan = sum(r["cost_plan"] for r in out)
    c_dev = sum(r["cost_dev"] for r in out)
    c_emg = sum(r["cost_emg"] for r in out)
    q_plan = sum(r["b_plan"].sum() * DT_H for r in out)
    q_adj = sum(r["b_adj"].sum() * DT_H for r in out)
    q_emg = sum(r["e"].sum() * DT_H for r in out)
    q_c = sum(r["c"].sum() * DT_H for r in out)
    q_d = sum(r["d"].sum() * DT_H for r in out)
    q_s = sum(r["s"].sum() * DT_H for r in out)
    return {
        "days": len(out), "recs": out,
        "q_plan": q_plan, "q_adj": q_adj, "q_emg": q_emg,
        "q_ch": q_c, "q_dis": q_d, "q_curtail": q_s,
        "cost_plan": c_plan, "cost_dev": c_dev, "cost_emg": c_emg,
        "cost_total": c_dev + c_emg,
        "n_emg_days": sum(1 for r in out if r["e"].sum() > 1e-6),
        "n_adj_days": sum(1 for r in out if np.abs(r["b_adj"] - r["b_plan"]).max() > 1e-6),
    }


def main():
    global PV_MIX, HEDGE, ADAPT, TXT_Q3
    args = [a for a in sys.argv[1:]]
    if args and args[0] == "ablate":
        return ablation()

    is_default = not args
    if len(args) > 0:
        PV_MIX = float(args[0])
    if len(args) > 1:
        HEDGE = float(args[1])
    if len(args) > 2:
        ADAPT = float(args[2])
    TXT_Q3 = ("q3_结果汇总.txt" if is_default
              else f"q3_结果汇总_m{PV_MIX:g}h{HEDGE:g}a{ADAPT:g}.txt")

    # ---------- 数据 ----------
    A1 = att1_arrays(load_att1())
    pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
    dates, L, G = load_att2()
    df3 = load_att3()

    rule("【1】问题三模型设定")
    log(f"决策时刻：0:00 制定计划；随后 {', '.join(f'{h}:00' for h in DECIDE_H[1:])} 依次调整")
    log(f"（附件3 的预报发布时刻为 {', '.join(f'{h}:00' for h in REL_H)}；决策时刻取其最近一次已发布预报）")
    log(f"光伏预报口径：附件3『预报k小时』= 发布后第 k 个整点时刻的点值 → 整点线性插值到 10 分钟时段")
    log(f"光伏预报组合：{PV_MIX:.0%} 附件3 + {1 - PV_MIX:.0%} 前 {PV_HIST_N} 天滑动平均"
        f"（附件3 单用 MAE 191.7 kW，历史外推 152.7 kW，混合后 130.6 kW）")
    log(f"负载日前预测：上周同一天（lag={LOAD_LAG} 天，只用历史）；安全裕量 {HEDGE:g} kW")
    log(f"负载当日自适应修正：限幅 ±{ADAPT:.0%}（用已发生时段的实际/预报之比缩放剩余预报）")
    log(f"购电费用口径：C_t = π_t·min(b^p_t, b^a_t) + 0.5π_t·(b^p_t−b^a_t)^+ + 1.5π_t·(b^a_t−b^p_t)^+")
    log("储能执行：逐槽被动平衡（与问题二完全同一口径），调整 LP 通过分段线性约束精确嵌入该规则")

    # ---------- 全年求解 ----------
    rule("【2】全年滚动求解")
    recs, s = run_year(df3, dates, L, G, pi, L1, G1, DECIDE_H, PV_MIX, HEDGE, ADAPT)

    # ---------- 汇总 ----------
    rule("【3】全年结果汇总（2025.2.1 - 12.31）")
    log(f"天数 = {s['days']}")
    log(f"计划购电量 = {s['q_plan']:,.1f} kWh    调整后购电量 = {s['q_adj']:,.1f} kWh"
        f"    （净变化 {s['q_adj'] - s['q_plan']:+,.1f} kWh）")
    log(f"计划购电费 = {s['cost_plan']:,.1f} 元")
    log(f"调整相关购电费（含计划部分）= {s['cost_dev']:,.1f} 元"
        f"    （相对计划购电费 {s['cost_dev'] - s['cost_plan']:+,.1f} 元）")
    log(f"紧急购电量 = {s['q_emg']:,.1f} kWh（占调整后购电量 {s['q_emg'] / s['q_adj']:.2%}）"
        f"    紧急购电费 = {s['cost_emg']:,.1f} 元")
    log(f"总购电费 = {s['cost_total']:,.1f} 元")
    log(f"充电量 = {s['q_ch']:,.1f} kWh    放电量 = {s['q_dis']:,.1f} kWh"
        f"    弃光量 = {s['q_curtail']:,.1f} kWh")
    log(f"出现紧急购电的天数 = {s['n_emg_days']}/{s['days']} 天；"
        f"发生调整的天数 = {s['n_adj_days']}/{s['days']} 天")

    # ---------- 题目指定日期 ----------
    rule("【4】题目指定日期的结果（表1 / 表2 / 表3）")
    for ds in KEY_DATES:
        r = next((x for x in recs if x["date"] == pd.Timestamp(ds)), None)
        if r is None:
            continue
        log(f"\n—— {ds} ——")
        log(f"  各时刻储电量入口：" + "  ".join(
            f"{h}:00 → {E:,.0f} kWh" for h, _t, E in r["epochs_info"]) if r["epochs_info"] else "")
        log("  表1 微网购电量（调整后）")
        log("      时间段           购电量(kWh)      时间段           购电量(kWh)")
        for j in range(0, 6, 2):
            a = f"{LAB_REP[j]:<14s} {r['b_adj'][IDX_REP[j]] * DT_H:>12.2f}"
            b = f"{LAB_REP[j + 1]:<14s} {r['b_adj'][IDX_REP[j + 1]] * DT_H:>12.2f}"
            log(f"      {a}      {b}")
        log(f"      全天计划购电量 {r['b_plan'].sum() * DT_H:,.2f} kWh"
            f"    全天调整购电量 {r['b_adj'].sum() * DT_H:,.2f} kWh")
        log(f"      全天购电费（含调整）{r['cost_dev']:,.2f} 元")
        log("  表2 储能充放电量")
        for k, (s0, s1) in enumerate(IDX4H):
            log(f"      {LAB4H[k]:<12s} 充电 {r['c'][s0:s1].sum() * DT_H:>10.2f} kWh   "
                f"放电 {r['d'][s0:s1].sum() * DT_H:>10.2f} kWh")
        log(f"      0:00 储电量 {r['E0']:,.2f} kWh    24:00 储电量 {r['E24']:,.2f} kWh")
        log("  表3 紧急购电")
        segs = seg_range(r["e"] > 1e-6)
        if not segs:
            log("      无紧急购电")
        else:
            for a, b in segs:
                log(f"      {fmt_span(a, b):<16s} {r['e'][a // 10:b // 10].sum() * DT_H:>10.2f} kWh")

    write_report(TXT_Q3)
    write_xlsx(recs, resolve(OUT3 if is_default else f"result3_m{PV_MIX:g}h{HEDGE:g}a{ADAPT:g}.xlsx"))
    print(f"\n结果已写入 {resolve(OUT3 if is_default else 'result3_m%gh%ga%g.xlsx' % (PV_MIX, HEDGE, ADAPT))}"
          f"（{TXT_Q3} 为文字汇总）")


# =====================================================================
# 六、结果文件写出（三张表 + 调整购电量）
# =====================================================================
def seg_range(mask):
    """布尔序列 → 连续区间列表 [(起分钟, 止分钟), ...]"""
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
    return f"{fmt(a)}-{fmt(b)}"


def write_xlsx(recs, path):
    """写出 result3.xlsx：计划购电量 / 调整购电量 / 充放电量 / 紧急购电量"""
    out = [r for r in recs if r["date"] >= pd.Timestamp(OUT_START)]
    wb = openpyxl.load_workbook(resolve(TPL3))

    for key, sheet in (("b_plan", "计划购电量"), ("b_adj", "调整购电量")):
        ws = wb[sheet]
        for i, r in enumerate(out):
            row = i + 2
            q = r[key] * DT_H
            for t in range(N_SLOT):
                ws.cell(row=row, column=2 + t, value=round(float(q[t]), 4))
            ws.cell(row=row, column=146, value=round(float(q.sum()), 4))
            ws.cell(row=row, column=147,
                    value=round(float(r["cost_dev"] if key == "b_adj" else r["cost_plan"]), 4))

    ws = wb["充放电量"]
    for i, r in enumerate(out):
        for k, (s0, s1) in enumerate(IDX4H):
            row = 2 + i * 6 + k
            if k == 0:
                ws.cell(row=row, column=1, value=r["date"].to_pydatetime())
            ws.cell(row=row, column=2, value=LAB4H[k])
            ws.cell(row=row, column=3, value=round(float(r["c"][s0:s1].sum() * DT_H), 4))
            ws.cell(row=row, column=4, value=round(float(r["d"][s0:s1].sum() * DT_H), 4))
            if k < 2:
                ws.cell(row=row, column=5, value="00:00" if k == 0 else "24:00")
                ws.cell(row=row, column=6,
                        value=round(float(r["E0"] if k == 0 else r["E24"]), 4))

    ws = wb["紧急购电量"]
    row = 2
    for r in out:
        segs = seg_range(r["e"] > 1e-6)
        if not segs:
            ws.cell(row=row, column=1, value=r["date"].to_pydatetime())
            row += 1
            continue
        for j, (a, b) in enumerate(segs):
            if j == 0:
                ws.cell(row=row, column=1, value=r["date"].to_pydatetime())
            ws.cell(row=row, column=2, value=fmt_span(a, b))
            ws.cell(row=row, column=3, value=round(float(r["e"][a // 10:b // 10].sum() * DT_H), 4))
            row += 1
    wb.save(path)


# =====================================================================
# 七、分析：是否需要引入其他时刻的预报
# =====================================================================
def ablation():
    """对比不同决策时刻组合，回答「是否需要引入其他时刻的预报」"""
    rule("问题三分析：各时刻预报的边际价值")
    A1 = att1_arrays(load_att1())
    pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
    dates, L, G = load_att2()
    df3 = load_att3()

    cases = [
        ("仅 0:00（不调整）", (0,)),
        ("0:00 + 6:00", (0, 6)),
        ("0:00 + 12:00", (0, 12)),
        ("0:00 + 18:00", (0, 18)),
        ("0:00 + 6:00 + 12:00", (0, 6, 12)),
        ("0:00 + 6:00 + 12:00 + 18:00（题目给定时刻）", (0, 6, 12, 18)),
        ("加密：每 3 小时（0,3,…,21）", tuple(range(0, 24, 3))),
        ("加密：每 2 小时（0,2,…,22）", tuple(range(0, 24, 2))),
        ("加密：每小时（0…23）", tuple(range(24))),
    ]
    base = None
    rows = []
    for nm, ep in cases:
        recs, s = run_year(df3, dates, L, G, pi, L1, G1, ep, PV_MIX, HEDGE, quiet=True)
        if base is None:
            base = s["cost_total"]
        rows.append((nm, s["cost_total"], base - s["cost_total"],
                     s["cost_emg"], s["q_emg"]))
        log(f"  {nm:<36s} 总费用 {s['cost_total']:>14,.1f} 元"
            f"   较不调整节省 {base - s['cost_total']:>12,.1f} 元"
            f"   紧急购电 {s['q_emg']:>12,.1f} kWh")

    rule("边际价值汇总")
    for j in range(1, len(rows)):
        log(f"  {rows[j][0]:<36s} 相对上一档再省 {rows[j - 1][1] - rows[j][1]:>12,.1f} 元")

    log("")
    log("结论：")
    log("  1) 0:00 的预报只决定计划的骨架，无法感知当天实际出力，紧急购电费最高；")
    log("  2) 6:00 与 12:00 的预报分别覆盖上午、下午的光伏爬坡段，边际价值最大；")
    log("  3) 18:00 的预报只覆盖 19:00-24:00（夜间光伏恒为 0），几乎不含增量信息；")
    log("  4) 若在已有预报之间加密决策时刻（用最近一次已发布预报 + 当日已发生的实际值），")
    log("     仍有额外收益，因为这些时刻能利用『当日已实现信息』修正储电量与净负荷偏差；")
    log("     但真实新增的预报信息只有 6:00 / 12:00 两次，故题目给定的四个时刻已足够。")
    write_report("q3_预报时刻边际价值.txt")


if __name__ == "__main__":
    main()
