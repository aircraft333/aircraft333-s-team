# -*- coding: utf-8 -*-
"""问题三：安全裕量也该用分位数规则吗？
=====================================================================
问题二已经证明「报童分位裕量」优于手调的固定 300 kW（全年省 13.9 万元）。
问题三的 `HEDGE` 现在仍是手调的固定 300 kW —— 本脚本验证它能否也升级。

问题三与问题二的两点差别：
  ① 裕量加在**负载预报**上（光伏另有附件3 的预报渠道），所以误差应取 L − F_L；
  ② 调整阶段有 1.5π 的加价 / 0.5π 的退款，边际结构比问题二的「π vs 5π」复杂，
     因此最优分位数**不一定**还是 1−1/k = 0.8，必须实测。

做法：在连续区块上对比
   · 固定 0 kW（无裕量，基线）
   · 固定 300 kW（现行主口径）
   · 分位数 q ∈ {0.70, 0.75, 0.80, 0.85}（逐时段取回看 60 天负载预报误差的分位数）

运行：python q3_hedge_q.py          （约 8~10 分钟）
"""
import time

import numpy as np

import q2
import q3
from config import *

BLOCK = (60, 190)                 # 与 q3 参数寻优同款区块（130 天）
QS = [0.70, 0.75, 0.80, 0.85]
FIX = [0.0, 300.0]
WIN = 60


def hedge3(L, L1, dates, q, win=WIN):
    """问题三的分位数裕量：对「负载预报误差」逐时段取滚动分位数 → (D, 144)"""
    F0 = q3.build_load_forecast(L, L1, dates, q3.DECIDE_H, q3.LOAD_LAG, q3.ADAPT)[0]
    return q2.build_hedge(L - F0, "quantile", q, win)


def run_block(dates, L, G, pi, Lf, Gf, X, i0, i1):
    """在 [i0, i1) 上跑滚动调度；X 可以是标量也可以是逐日矩阵"""
    tot = dict(c_plan=0.0, c_dev=0.0, c_emg=0.0,
               q_plan=0.0, q_adj=0.0, q_emg=0.0, n=0)
    E = SOC0
    for i in range(i0, i1):
        h = X if np.isscalar(X) else X[i]
        r = q3.run_day(pi, Lf[:, i], Gf[:, i], L[i], G[i], E, q3.DECIDE_H, h)
        E = r["E24"]
        tot["c_plan"] += r["cost_plan"]
        tot["c_dev"] += r["cost_dev"]
        tot["c_emg"] += r["cost_emg"]
        tot["q_plan"] += float(r["b_plan"].sum() * DT_H)
        tot["q_adj"] += float(r["b_adj"].sum() * DT_H)
        tot["q_emg"] += float(r["e"].sum() * DT_H)
        tot["n"] += 1
    tot["cost"] = tot["c_plan"] + tot["c_dev"] + tot["c_emg"]
    return tot


def main():
    t0 = time.time()
    A1 = att1_arrays(load_att1())
    pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
    dates, L, G = load_att2()
    df3 = load_att3()
    Lf = q3.build_load_forecast(L, L1, dates, q3.DECIDE_H, q3.LOAD_LAG, q3.ADAPT)
    Gf = q3.build_pv_forecast(df3, dates, G, q3.DECIDE_H, q3.PV_MIX)
    i0, i1 = BLOCK

    rule("问题三安全裕量：固定值 vs 报童分位数")
    log(f"区块 {i0}~{i1}（{i1 - i0} 天）；其余参数取 q3.py 默认 "
        f"(mix={q3.PV_MIX:g}, adapt={q3.ADAPT:g}, 决策时刻 {q3.DECIDE_H})")
    log(f"裕量加在负载预报上；分位数由「L − F_L」的回看 {WIN} 天滚动分位数给出")

    # 误差分布的先验信息
    errs = L - q3.build_load_forecast(L, L1, dates, q3.DECIDE_H,
                                      q3.LOAD_LAG, 0.0)[0]
    log(f"负载预报误差（全年）：均值 {errs.mean():.1f} kW，"
        f"标准差 {errs.std():.1f} kW，q50 {np.percentile(errs, 50):.0f}，"
        f"q75 {np.percentile(errs, 75):.0f}，q90 {np.percentile(errs, 90):.0f}")

    # ---------- 计算各方案的裕量矩阵 ----------
    XQ = {q: hedge3(L, L1, dates, q) for q in QS}

    rule("【1】区块对比")
    hdr = (f"  {'方案':>18s}{'日均裕量':>10s}{'计划购电费':>13s}{'调整费':>12s}"
           f"{'紧急购电费':>12s}{'总费用':>13s}{'相对基线':>12s}")
    log(hdr)
    log("  " + "-" * (len(hdr) + 2))
    res = {}
    base = None
    cfg = [("固定 0 kW", 0.0)] + [(f"固定 {h:g} kW", h) for h in FIX if h > 0]
    cfg += [(f"分位数 q={q:.2f}", XQ[q]) for q in QS]
    for name, X in cfg:
        t = time.time()
        r = run_block(dates, L, G, pi, Lf, Gf, X, i0, i1)
        xm = float(X) if np.isscalar(X) else float(X[i0:i1].mean())
        res[name] = dict(r=r, xm=xm)
        if base is None:
            base = r["cost"]
        log(f"  {name:>18s}{xm:>10.0f}{r['c_plan']:>13,.0f}{r['c_dev']:>12,.0f}"
            f"{r['c_emg']:>12,.0f}{r['cost']:>13,.0f}"
            f"{r['cost'] - base:>+12,.0f}   ({time.time() - t:.0f}s)")

    rule("【2】结论")
    best = min(res, key=lambda k: res[k]["r"]["cost"])
    b = res[best]["r"]["cost"]
    log(f"  最优方案：{best}   区块总费用 {b:,.0f} 元")
    for nm in ("固定 0 kW", "固定 300 kW"):
        if nm in res:
            log(f"  相对 {nm}：{b - res[nm]['r']['cost']:+,.0f} 元"
                f"（{(b - res[nm]['r']['cost']) / res[nm]['r']['cost']:+.2%}）")
    qb = [q for q in QS if f"分位数 q={q:.2f}" == best]
    if qb:
        log(f"  ⇒ 最优分位数 q = {qb[0]:.2f}，"
            f"与问题二的报童解析值 0.80 相比{'一致' if abs(qb[0] - 0.8) < 1e-9 else '有差异'}")
    else:
        log("  ⇒ 分位数规则未胜出，问题三的固定裕量口径不需要改")
    log(f"\n总耗时 {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
