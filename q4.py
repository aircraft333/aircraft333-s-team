# -*- coding: utf-8 -*-
"""
2026 高教社杯 C 题 · 问题四：波动电价下重算问题二、问题三
=====================================================================
题目要求
    实际上外网的电价也是实时波动的。根据附件2 的小区负载与光伏发电、附件3 的光伏发电预报、
    附件4 的电价数据，在波动电价下重新计算问题 2 和问题 3，结果分别写入
    result4-2.xlsx（对应问题 2）与 result4-3.xlsx（对应问题 3）。

附件4 的数据特征（实测）
    · 形状 365 × 144，与附件1 逐时段一一对应（列均值 ≡ 附件1 电价，最大绝对差 5.2e-5）
    · 逐日均价 0.5378 ~ 0.9290（std 0.1036），日内峰谷比均值 **5.74**（附件1 为 3.76）
    · ⇒ 电价 = 附件1 的分时形状 + 逐日整体水平漂移 + 逐槽噪声
      因此「谷充峰放」的日内套利依然存在，同时新出现**跨日套利**空间：
      4/26 日均 0.5378 与 1/6 日均 0.9290 相差 73%。
      但实测 7 日窗口的跨日套利只值 0.411%（逐日 1,629,250.9 vs 联合 1,622,550.5），
      原因是储能可用容量仅 9600 kWh、而日均净负荷约 7 万 kWh，跨日搬运受容量硬约束。

关键假设
    · 当天 0:00 已能获知当天的完整电价曲线（日前市场提前公布），
      故计划 LP 使用当天实际电价，不需要对电价做预测 —— 与题目「根据附件4 的电价数据」一致。
    · 其余口径（逐槽被动平衡执行、安全裕量、附件3 整点预报为点值）与问题二、三完全一致。

结果文件
    result4-2.xlsx：计划购电量 / 充放电量 / 紧急购电量               （对应问题 2）
    result4-3.xlsx：计划购电量 / 调整购电量 / 充放电量 / 紧急购电量   （对应问题 3）

命令行
    python q4.py             # 重算问题二、三并写出两个结果文件
    python q4.py multiday    # 额外做「逐日 LP vs 多日联合 LP」分析，量化跨日套利价值
"""
import sys

import openpyxl
import pulp

import q2
import q3
from config import *

# ==================== 运行配置 ====================
OUT_START = "2025-02-01"
TXT_Q4 = "q4_结果汇总.txt"
KEY_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]

IDX4H = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]
LAB4H = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"]
IDX_REP = [60, 72, 84, 96, 108, 120]
LAB_REP = ["10:00-10:10", "12:00-12:10", "14:00-14:10", "16:00-16:10", "18:00-18:10", "20:00-20:10"]


# =====================================================================
# 一、附件4：逐日电价
# =====================================================================
def load_att4():
    """读取附件4，返回 (365, 144) 的逐日电价矩阵

    列口径与附件1、附件2 完全一致：第 k 列的时间戳为右端点，对应区间 [T-10min, T]。
    """
    df = pd.read_excel(resolve(ATT4))
    P = df.iloc[:, 1:1 + N_SLOT].to_numpy(float)
    assert P.shape == (365, N_SLOT), f"附件4 结构异常：{P.shape}"
    return P


# =====================================================================
# 二、问题二的口径：0:00 计划 + 逐槽执行
# =====================================================================
def run_day_q2(pi, L_plan, G_plan, L, G, E_start):
    """波动电价下的单日：0:00 计划（用当天电价）→ 逐槽被动平衡执行"""
    plan = q2.solve_day(pi, L_plan, G_plan, E_start)
    bp = plan["b"]
    sim = simulate_dispatch(L, G, bp, E_start)
    return {
        "b_plan": bp, "b_adj": bp,
        "c": sim["c"], "d": sim["d"], "e": sim["e"], "s": sim["s"], "E": sim["E"],
        "E0": E_start, "E24": sim["E"][-1],
        "cost_plan": float(np.sum(pi * bp * DT_H)),
        "cost_dev": float(np.sum(pi * bp * DT_H)),
        "cost_emg": float(np.sum(5.0 * pi * sim["e"] * DT_H)),
        "info": [],
    }


# =====================================================================
# 三、全年求解（口径 = 问题二 或 问题三）
# =====================================================================
def run_year(PI, dates, L, G, L1, G1, df3, mode="q2"):
    """mode="q2" 对应问题二（仅 0:00 计划）；mode="q3" 对应问题三（四阶段滚动）

    PI 为 (D, 144) 的逐日电价；每一问都用同一套因果规则，保证结果可归因。
    """
    D = len(dates)
    if mode == "q3":
        Gf = q3.build_pv_forecast(df3, dates, G, q3.DECIDE_H, q3.PV_MIX)
        Lf = q3.build_load_forecast(L, L1, dates, q3.DECIDE_H, q3.LOAD_LAG, q3.ADAPT)
        # 裕量口径与 q3.py 保持一致（分位数模式给逐日矩阵，标量模式给常数）
        X3 = (q3.build_hedge_q3(L, L1, dates) if q3.HEDGE_MODE == "quantile"
              else q3.HEDGE)
    else:
        F_L, F_G = q2.build_forecast(L, G, L1, G1, q2.FC_SPEC)
        err = (L - G) - (F_L - F_G)
        X = q2.build_hedge(err, q2.HEDGE_MODE, q2.HEDGE_PARAM, q2.HEDGE_WIN)
        Lp = F_L + X                                   # 负载预测 + 安全裕量
        Gp = F_G

    recs, E_start = [], SOC0
    for i, day in enumerate(dates):
        pi = PI[i]                                     # 当天电价（0:00 已知）
        if mode == "q3":
            h = X3 if np.isscalar(X3) else X3[i]
            r = q3.run_day(pi, Lf[:, i], Gf[:, i], L[i], G[i], E_start,
                           q3.DECIDE_H, h)
        else:
            r = run_day_q2(pi, Lp[i], Gp[i], L[i], G[i], E_start)
        r["info"] = r.get("epochs_info", r.get("info", []))   # 统一键名
        r["date"] = day
        recs.append(r)
        E_start = r["E24"]
        if (i + 1) % 60 == 0:
            log(f"    已完成 {i + 1}/{D} 天 …")
    return recs, q3.summarize(recs, OUT_START)


# =====================================================================
# 四、结果文件写出
# =====================================================================
def write_xlsx(recs, path, tpl, with_adj):
    """写出结果文件；with_adj=True 时多写一张「调整购电量」表"""
    out = [r for r in recs if r["date"] >= pd.Timestamp(OUT_START)]
    wb = openpyxl.load_workbook(resolve(tpl))

    pairs = [("b_plan", "计划购电量", "cost_plan")]
    if with_adj:
        pairs.append(("b_adj", "调整购电量", "cost_dev"))
    for key, sheet, ckey in pairs:
        ws = wb[sheet]
        for i, r in enumerate(out):
            row = i + 2
            q = r[key] * DT_H
            for t in range(N_SLOT):
                ws.cell(row=row, column=2 + t, value=round(float(q[t]), 4))
            ws.cell(row=row, column=146, value=round(float(q.sum()), 4))
            ws.cell(row=row, column=147, value=round(float(r[ckey]), 4))

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
        segs = q3.seg_range(r["e"] > 1e-6)
        if not segs:
            ws.cell(row=row, column=1, value=r["date"].to_pydatetime())
            row += 1
            continue
        for j, (a, b) in enumerate(segs):
            if j == 0:
                ws.cell(row=row, column=1, value=r["date"].to_pydatetime())
            ws.cell(row=row, column=2, value=q3.fmt_span(a, b))
            ws.cell(row=row, column=3,
                    value=round(float(r["e"][a // 10:b // 10].sum() * DT_H), 4))
            row += 1
    save_wb(wb, path)


# =====================================================================
# 五、论文用表：题目指定日期
# =====================================================================
def dump_key_dates(recs, title, with_adj):
    rule(title)
    for ds in KEY_DATES:
        r = next((x for x in recs if x["date"] == pd.Timestamp(ds)), None)
        if r is None:
            continue
        log(f"\n—— {ds} ——")
        inf = r.get("info") or r.get("epochs_info") or []
        if inf:
            log("  各时刻储电量入口：" + "  ".join(
                f"{h}:00 → {E:,.0f} kWh" for h, _t, E in inf))
        log("  表1 微网购电量")
        log("      时间段           购电量(kWh)      时间段           购电量(kWh)")
        for j in range(0, 6, 2):
            a = f"{LAB_REP[j]:<14s} {r['b_adj'][IDX_REP[j]] * DT_H:>12.2f}"
            b = f"{LAB_REP[j + 1]:<14s} {r['b_adj'][IDX_REP[j + 1]] * DT_H:>12.2f}"
            log(f"      {a}      {b}")
        if with_adj:
            log(f"      全天计划购电量 {r['b_plan'].sum() * DT_H:,.2f} kWh"
                f"    全天调整购电量 {r['b_adj'].sum() * DT_H:,.2f} kWh")
        log(f"      全天购电量 {r['b_adj'].sum() * DT_H:,.2f} kWh"
            f"    全天购电费 {r['cost_dev']:,.2f} 元")
        log("  表2 储能充放电量")
        for k, (s0, s1) in enumerate(IDX4H):
            log(f"      {LAB4H[k]:<12s} 充电 {r['c'][s0:s1].sum() * DT_H:>10.2f} kWh   "
                f"放电 {r['d'][s0:s1].sum() * DT_H:>10.2f} kWh")
        log(f"      0:00 储电量 {r['E0']:,.2f} kWh    24:00 储电量 {r['E24']:,.2f} kWh")
        log("  表3 紧急购电")
        segs = q3.seg_range(r["e"] > 1e-6)
        if not segs:
            log("      无紧急购电")
        else:
            for a, b in segs:
                log(f"      {q3.fmt_span(a, b):<16s}"
                    f" {r['e'][a // 10:b // 10].sum() * DT_H:>10.2f} kWh")


def dump_summary(tag, s, pi_mean):
    rule(f"【{tag}】全年结果汇总（2025.2.1 - 12.31）")
    log(f"天数 = {s['days']}；电价均值 = {pi_mean:.4f} 元/kWh")
    log(f"计划购电量 = {s['q_plan']:,.1f} kWh"
        + (f"    调整后购电量 = {s['q_adj']:,.1f} kWh"
           f"（净 {s['q_adj'] - s['q_plan']:+,.1f}）" if abs(s['q_adj'] - s['q_plan']) > 1e-6
           else ""))
    log(f"计划购电费 = {s['cost_plan']:,.1f} 元")
    if abs(s['cost_dev'] - s['cost_plan']) > 1e-6:
        log(f"调整相关购电费（含计划部分）= {s['cost_dev']:,.1f} 元"
            f"（相对计划 {s['cost_dev'] - s['cost_plan']:+,.1f}）")
    log(f"紧急购电量 = {s['q_emg']:,.1f} kWh（占调整后购电量 "
        f"{s['q_emg'] / max(s['q_adj'], 1e-9):.2%}）；紧急购电费 = {s['cost_emg']:,.1f} 元")
    log(f"总购电费 = {s['cost_total']:,.1f} 元")
    log(f"充电量 = {s['q_ch']:,.1f} kWh    放电量 = {s['q_dis']:,.1f} kWh"
        f"    弃光量 = {s['q_curtail']:,.1f} kWh")
    log(f"出现紧急购电的天数 = {s['n_emg_days']}/{s['days']} 天")
    return s["cost_total"]


# =====================================================================
# 六、主流程
# =====================================================================
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "multiday":
        return multiday()

    # ---------- 数据 ----------
    A1 = att1_arrays(load_att1())
    pi1, L1, G1 = A1["price"], A1["load"], A1["pv"]      # 附件1：基准电价 / 负载 / 光伏预测
    dates, L, G = load_att2()
    df3 = load_att3()
    PI = load_att4()

    rule("【1】问题四模型设定")
    log("题目要求：用附件4 的波动电价重新计算问题二与问题三")
    log("假设：当天 0:00 已获知当天完整电价曲线（日前市场提前公布），故电价无需预测，直接用当天实际值")
    log("其余口径与问题二、三完全一致：逐槽被动平衡执行、附件3 整点预报按点值口径插值")
    log(f"附件4 电价：全局 {PI.min():.4f} ~ {PI.max():.4f}，均值 {PI.mean():.4f} 元/kWh")
    log(f"            逐日均价 {PI.mean(axis=1).min():.4f} ~ {PI.mean(axis=1).max():.4f}"
        f"（std {PI.mean(axis=1).std():.4f}）")
    log(f"            日内峰谷比：均值 {(PI.max(axis=1) / np.maximum(PI.min(axis=1), 1e-6)).mean():.2f}"
        f"（附件1 固定电价下为 {pi1.max() / pi1.min():.2f}）")
    log(f"问题二口径：日前预测 {q2.FC_SPEC} + 安全裕量 "
        + (f"逐时段历史 {q2.HEDGE_WIN} 天误差的 {q2.HEDGE_PARAM:.2f} 分位数"
           if q2.HEDGE_MODE == "quantile" else f"固定 {q2.HEDGE_PARAM:g} kW"))
    log(f"问题三口径：光伏预报 {q3.PV_MIX:.0%} 附件3 + {1 - q3.PV_MIX:.0%} 历史；"
        f"裕量 "
        + (f"逐时段历史 {q3.HEDGE_WIN} 天负载误差的 {q3.HEDGE_PARAM:.2f} 分位数"
           if q3.HEDGE_MODE == "quantile" else f"固定 {q3.HEDGE:g} kW")
        + f"；负载自适应 ±{q3.ADAPT:.0%}；决策时刻 {q3.DECIDE_H}")

    # ---------- 重算问题二 ----------
    rule("【2】波动电价下重算问题二")
    recs2, s2 = run_year(PI, dates, L, G, L1, G1, df3, "q2")
    c2 = dump_summary("问题四·对应问题二", s2, PI[dates >= pd.Timestamp(OUT_START)].mean())

    # ---------- 重算问题三 ----------
    rule("【3】波动电价下重算问题三")
    recs3, s3 = run_year(PI, dates, L, G, L1, G1, df3, "q3")
    c3 = dump_summary("问题四·对应问题三", s3, PI[dates >= pd.Timestamp(OUT_START)].mean())

    rule("【4】固定电价 vs 波动电价")
    log(f"  {'口径':<22s} {'固定电价(附件1)':>18s} {'波动电价(附件4)':>18s} {'变化':>16s}")
    fixed = {"对应问题二": 13949108.5, "对应问题三": 13589362.4}   # 来自 q2.py / q3.py 全年结果
    q2_plan_kwh = 21442248.8                          # 问题二全年计划购电量（用于算平均单价）
    for nm, c, f in (("对应问题二", c2, fixed["对应问题二"]),
                     ("对应问题三", c3, fixed["对应问题三"])):
        log(f"  {nm:<22s} {f:>18,.1f} {c:>18,.1f} {c - f:>+16,.1f} 元"
            f"（{(c / f - 1):+.2%}）")
    log("说明：附件4 的逐日均价与附件1 相同（0.7662），但日内峰谷比特偏高、且逐日漂移，")
    log("      计划必须逐日贴合当天电价曲线，因此总费用与固定电价下有系统性差异。")

    # 参考基准：无储能，直接按当天现货价购买净负荷（排除储能调度的影响，只看电价本身）
    pi1 = att1_arrays(load_att1())["price"]
    msk = dates >= pd.Timestamp(OUT_START)
    netd = np.maximum(L - G, 0.0)[msk]
    bfix = float(np.sum(np.tile(pi1, (netd.shape[0], 1)) * netd * DT_H))
    bflx = float(np.sum(PI[msk] * netd * DT_H))
    log(f"  参考（无储能、按现货价直接买净负荷）：固定电价 {bfix:,.1f} 元"
        f" / 波动电价 {bflx:,.1f} 元（{bflx / bfix - 1:+.2%}）")
    log(f"  实际平均购电单价：固定电价 {fixed['对应问题二'] / q2_plan_kwh:.4f} 元/kWh"
        f" / 波动电价 {c2 / s2['q_plan']:.4f} 元/kWh")

    # ---------- 先写结果文件（避免报告出错导致交付物丢失）----------
    p2, p3 = resolve(OUT4_2), resolve(OUT4_3)
    write_xlsx(recs2, p2, TPL4_2, with_adj=False)
    write_xlsx(recs3, p3, TPL4_3, with_adj=True)
    print(f"结果已写入\n  {p2}（对应问题二）\n  {p3}（对应问题三）")

    # ---------- 论文用表 ----------
    try:
        dump_key_dates(recs2, "【5】对应问题二 —— 题目指定日期的结果（表1 / 表2 / 表3）", False)
        dump_key_dates(recs3, "【6】对应问题三 —— 题目指定日期的结果（表1 / 表2 / 表3）", True)
    except Exception as e:                                  # 报告失败不能影响结果文件
        log(f"（指定日期表格生成失败：{type(e).__name__}: {e}）")

    write_report(TXT_Q4)
    print(f"（{TXT_Q4} 为文字汇总）")


# =====================================================================
# 七、分析：波动电价下的跨日套利（逐日 LP vs 多日联合 LP）
# =====================================================================
def solve_joint(pi_mat, L, G, E_start):
    """多日联合 LP：min Σ π·b·Δt，储能在窗口内自由跨日调度（只在窗口起点给定初值）

    返回 dict(b, c, d, s, E)，长度 T×n_days
    """
    nd, T = pi_mat.shape
    n = nd * T
    pi = pi_mat.ravel()
    Lf, Gf = L.ravel(), G.ravel()
    m = pulp.LpProblem("joint", pulp.LpMinimize)
    b = [pulp.LpVariable(f"b{i}", lowBound=0) for i in range(n)]
    c = [pulp.LpVariable(f"c{i}", lowBound=0, upBound=P_RATE) for i in range(n)]
    d = [pulp.LpVariable(f"d{i}", lowBound=0, upBound=P_RATE) for i in range(n)]
    s = [pulp.LpVariable(f"s{i}", lowBound=0) for i in range(n)]
    E = [pulp.LpVariable(f"E{i}", lowBound=SOC_MIN, upBound=SOC_MAX) for i in range(n)]
    m += pulp.lpSum(pi[i] * b[i] * DT_H for i in range(n))
    for i in range(n):
        m += b[i] + Gf[i] + d[i] - c[i] - s[i] == Lf[i]
        prev = E_start if i == 0 else E[i - 1]
        m += E[i] == prev + (ETA * c[i] - d[i] / ETA) * DT_H
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[m.status] != "Optimal":
        raise RuntimeError(pulp.LpStatus[m.status])
    out = {k: np.array([v.value() for v in arr]) for k, arr in (("b", b), ("E", E))}
    out["cost"] = float(np.sum(pi * out["b"] * DT_H))
    out["E_end"] = float(out["E"][-1])
    return out


def multiday(win=7, n_win=6):
    """在若干窗口中对比「逐日 LP」与「多日联合 LP」，量化跨日套利价值"""
    A1 = att1_arrays(load_att1())
    L1, G1 = A1["load"], A1["pv"]
    dates, L, G = load_att2()
    PI = load_att4()

    rule(f"波动电价下的跨日套利：逐日 LP vs {win} 日联合 LP")
    log("为保证可比，两者都用**实际** 净负荷与**实际** 电价，即不存在预测误差：")
    log("  逐日 LP：每天 0:00 独立决策，储能只能日内调度，SOC 逐日连续")
    log("  联合 LP：窗口内储能跨日自由调度，只在窗口起点给定初值，期末仍自由")
    log("两者差别只在于「决策视界」，因此差额就是跨日套利的价值。")
    log("")

    # 挑几个窗口：电价最低 / 最高 / 中位附近的起始日
    dm = PI.mean(axis=1)
    starts = sorted({int(np.clip(s, 0, len(dates) - win - 1))
                     for s in (int(np.argmin(dm)), int(np.argmax(dm)),
                               len(dates) // 3, len(dates) // 2, 240, 300)})
    starts = starts[:n_win]

    tot = [0.0, 0.0]
    log(f"  {'窗口起始':<12s} {'逐日LP':>14s} {'联合LP':>14s} {'节省':>12s} {'相对':>9s}"
        f"  {'窗口日均价':>10s}")
    for s0 in starts:
        win_days = np.arange(s0, s0 + win)
        # 逐日 LP：每天 0:00 独立决策（用实际净负荷，SOC 逐日连续）
        E, cd = SOC0, 0.0
        for i in win_days:
            pl = q2.solve_day(PI[i], L[i], G[i], E)
            cd += float(np.sum(PI[i] * pl["b"] * DT_H))
            E = pl["E"][-1]
        # 多日联合 LP
        jr = solve_joint(PI[win_days], L[win_days], G[win_days], SOC0)
        log(f"  {str(dates[s0].date()):<12s} {cd:>14,.1f} {jr['cost']:>14,.1f}"
            f" {cd - jr['cost']:>12,.1f} {(cd - jr['cost']) / cd:>8.2%}"
            f"  {PI[win_days].mean():>10.4f}")
        tot[0] += cd
        tot[1] += jr["cost"]

    log("")
    log(f"  合计：逐日 LP {tot[0]:,.1f} 元；联合 LP {tot[1]:,.1f} 元；"
        f"节省 {tot[0] - tot[1]:,.1f} 元（{(tot[0] - tot[1]) / tot[0]:.3%}）")
    log("")
    log("结论：")
    log("  1) 固定电价（附件1）下逐日 LP 与多日联合 LP 只差 0.139%（见 q2_多日LP对比.txt）：")
    log("     每天的谷峰结构完全相同，储能无法把能量搬到另一天去获益；")
    log("  2) 波动电价（附件4）下逐日均价在 0.54~0.93 之间漂移，跨日套利空间确实被打开，")
    log("     但 7 日窗口的联合优化也只多省 0.411%（约为固定电价的 3 倍，绝对量仍然很小）；")
    log("  3) 根本原因是储能可用容量仅 9600 kWh，而日均净负荷约 7 万 kWh，")
    log("     「跨日搬运能量」受容量硬约束，因此逐日 + 日内套利已经捕获了绝大部分价值；")
    log("  4) 实用结论：波动电价下仍可采用逐日决策（问题二 / 问题三的模型），")
    log("     多日联合优化只是理论上更完整的形式，实际增益有限。")
    write_report("q4_跨日套利分析.txt")


if __name__ == "__main__":
    main()
