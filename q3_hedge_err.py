# -*- coding: utf-8 -*-
"""问题三：安全裕量的「误差口径」对照 —— 用负载误差还是净负荷误差？

背景（也是文档里被追问概率最高的一点）
    问题二的裕量建在**净负荷**误差上：
        ε = (L − G) − (F_L − F_G)
    问题三的裕量建在**负载**误差上：
        ε = L − F_L
    理由（写在 q3.build_hedge_q3 的 docstring 里）：问题三的光伏已有附件3 这条
    预报渠道，若再把光伏误差算进裕量，等于同一个误差被防两次，裕量会偏大。

    但这条理由**此前没有做过对照实验**。本脚本补上：
    在完全相同的其他口径下，只替换裕量的误差定义，比较全年总购电费。

    同时因为误差的方差变大（净负荷误差 = 负载误差 − 光伏误差，两者的方差叠加），
    报童模型的最优分位也会受影响，故对净负荷口径额外扫几个分位。

口径
    其余一律取 q3.py 现行默认（mix=0.4、adapt=0.2、决策时刻 (0,6,12,18)、
    回看窗口 60 天），只用 run_year 的 hedge 参数替换裕量矩阵 —— 无需改动主程序。

输出：控制台 + q3_裕量误差口径.txt
"""
import time

import numpy as np
import pandas as pd

import q2
import q3
from config import *   # noqa: F401,F403

TXT = resolve("q3_裕量误差口径.txt")
WIN = 60
arm_report(TXT)                  # 长跑安全网：中途异常也不丢已算出的报告


def main():
    t0 = time.time()
    A1 = att1_arrays(load_att1())
    pi, L1 = A1["price"], A1["load"]
    dates, L, G = load_att2()
    df3 = load_att3()

    # 0:00 时刻的负载/光伏预报（裕量按 Q2 的做法只依据 0:00 的预报构造）
    F_L0 = q3.build_load_forecast(L, L1, dates, q3.DECIDE_H,
                                  q3.LOAD_LAG, q3.ADAPT)[0]
    F_G0 = q3.build_pv_forecast(df3, dates, G, q3.DECIDE_H, q3.PV_MIX)[0]
    err_load = L - F_L0
    err_net = (L - G) - (F_L0 - F_G0)

    rule()
    log("问题三：安全裕量的误差口径对照（负载误差 vs 净负荷误差）")
    log(f"其余口径取 q3.py 现行默认：mix={q3.PV_MIX:g}、adapt={q3.ADAPT:g}、"
        f"回看窗口 {WIN} 天、决策时刻 {q3.DECIDE_H}")
    rule()

    rule("一、两种误差的统计特征")
    log(f"  {'口径':<12s}{'均值':>10s}{'标准差':>10s}{'q70':>10s}{'q75':>10s}{'q80':>10s}")
    for nm, e in (("负载误差", err_load), ("净负荷误差", err_net)):
        log(f"  {nm:<12s}{e.mean():>10.2f}{e.std():>10.2f}"
            f"{np.percentile(e, 70):>10.1f}{np.percentile(e, 75):>10.1f}"
            f"{np.percentile(e, 80):>10.1f}")
    log("")
    log(f"  净负荷误差的标准差是负载误差的 {err_net.std() / err_load.std():.2f} 倍"
        f"（光伏误差叠加进来），故其同分位数的裕量数值更大。")

    # 自检：用负载误差 + q=0.75 应当复现主程序现用的裕量矩阵
    X_load75 = q2.build_hedge(err_load, "quantile", q3.HEDGE_PARAM, q3.HEDGE_WIN)
    X_ref = q3.build_hedge_q3(L, L1, dates, q3.HEDGE_PARAM, q3.HEDGE_WIN,
                              q3.DECIDE_H, q3.LOAD_LAG, q3.ADAPT)
    same = np.allclose(X_load75, X_ref)
    log(f"  自检：负载误差 + q={q3.HEDGE_PARAM:g} 构造的裕量与 build_hedge_q3 一致 -> {same}")
    assert same, "裕量构造口径与主程序不一致，对照无意义"
    log(f"  现用裕量：日均 {X_ref.mean():.0f} kW、最大 {X_ref.max():.0f} kW")

    rule("二、全年总购电费对照（每个约 110 s）")
    cases = [
        (f"负载误差 q={q3.HEDGE_PARAM:g}（现行）", err_load, q3.HEDGE_PARAM),
        ("净负荷误差 q=0.75", err_net, 0.75),
        ("净负荷误差 q=0.80", err_net, 0.80),
        ("净负荷误差 q=0.70", err_net, 0.70),
    ]
    hdr = (f"  {'方案':<26s}{'日均裕量':>10s}{'总购电费':>15s}{'相对现行':>13s}"
           f"{'紧急购电费':>14s}{'弃光量':>14s}")
    log(hdr)
    log("  " + "-" * (len(hdr) + 2))
    base = None
    res = []
    for nm, err, q in cases:
        t1 = time.time()
        X = q2.build_hedge(err, "quantile", q, WIN)
        _, s = q3.run_year(df3, dates, L, G, pi, L1, None,
                           q3.DECIDE_H, q3.PV_MIX, X, q3.ADAPT, quiet=True)
        if base is None:
            base = s["cost_total"]
        res.append((nm, X.mean(), s))
        log(f"  {nm:<26s}{X.mean():>10.0f}{s['cost_total']:>15,.1f}"
            f"{s['cost_total'] - base:>+13,.1f}{s['cost_emg']:>14,.1f}"
            f"{s['q_curtail']:>14,.1f}   ({time.time() - t1:.0f}s)")
    log("")
    log(f"  自检：现行口径应复现主结果 13,589,362.4 元 -> 实得 {base:,.1f} 元，"
        f"差 {base - 13589362.4:+,.1f} 元")

    rule("三、结论")
    best = min(res, key=lambda r: r[2]["cost_total"])
    log(f"  最优方案：{best[0]}（{best[2]['cost_total']:,.1f} 元）")
    if best[0].startswith("负载误差"):
        log("  => 现用口径（负载误差 + 折中分位 0.75）确实是最优的，")
        log("     文档里给出的「避免同一误差被防两次」的理由得到实测支持。")
    else:
        log(f"  => 净负荷误差口径更优，比现行省 {base - best[2]['cost_total']:,.1f} 元"
            f"（{(base / best[2]['cost_total'] - 1):+.2%}）；")
        log("     若要改用，需要同步更新 q3.build_hedge_q3 与文档说明，并重跑 q4。")
    log("")
    log("  附：各方案完整排序")
    for nm, xm, s in sorted(res, key=lambda r: r[2]["cost_total"]):
        log(f"    {nm:<26s}{s['cost_total']:>15,.1f}  ({s['cost_total'] - base:+13,.1f})")

    write_report(TXT)
    log("")
    log("总耗时 %.1f 分钟" % ((time.time() - t0) / 60.0))


if __name__ == "__main__":
    main()
