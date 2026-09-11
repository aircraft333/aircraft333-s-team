# -*- coding: utf-8 -*-
"""PV_MIX 稳健性检验：0.30 相对 0.40 的 5,965 元优势是否稳定？

全年复检（q3_mix_recheck.py）显示最终口径下最优 mix 从 0.40 漂到 0.30，
但优势只有 5,965.5 元（0.044%）。这么小的差值必须做子区间检验：
若「0.30 更优」只在部分区间成立、在另一些区间反号，就说明它落在噪声里，
不应为了 0.044% 去改动主程序口径（否则等于对调参窗口过拟合）。

做法：把输出窗口等分为 3 段（每段约 111 天），逐段比较 mix=0.30 与 0.40。
每段都从同一初始储电量起跑，两个 mix 用完全相同的负载预报与裕量矩阵，
唯一变量就是光伏预报的混合权重，属严格对照。

输出：控制台 + q3_光伏权重稳健性.txt
"""
import time

import numpy as np
import pandas as pd

import q3
from config import *   # noqa: F401,F403

TXT = resolve("q3_光伏权重稳健性.txt")
MIXS = [0.30, 0.40]


def run_block(dates, L, G, pi, Lf, Gf, X, i0, i1):
    """在 [i0, i1) 上滚动调度（从 SOC0 起跑），返回分段汇总"""
    tot = dict(c_dev=0.0, c_emg=0.0, q_emg=0.0, q_curtail=0.0, n=0)
    E = SOC0
    for i in range(i0, i1):
        r = q3.run_day(pi, Lf[:, i], Gf[:, i], L[i], G[i], E, q3.DECIDE_H, X[i])
        E = r["E24"]
        tot["c_dev"] += r["cost_dev"]
        tot["c_emg"] += r["cost_emg"]
        tot["q_emg"] += float(r["e"].sum() * DT_H)
        tot["q_curtail"] += float(r["s"].sum() * DT_H)
        tot["n"] += 1
    tot["cost"] = tot["c_dev"] + tot["c_emg"]
    return tot


def main():
    t0 = time.time()
    A1 = att1_arrays(load_att1())
    pi, L1 = A1["price"], A1["load"]
    dates, L, G = load_att2()
    df3 = load_att3()

    # 公共：负载预报 + 分位裕量矩阵（只依赖 adapt，与 mix 无关）
    Lf = q3.build_load_forecast(L, L1, dates, q3.DECIDE_H, q3.LOAD_LAG, q3.ADAPT)
    X = q3.build_hedge_q3(L, L1, dates, q3.HEDGE_PARAM, q3.HEDGE_WIN,
                          q3.DECIDE_H, q3.LOAD_LAG, q3.ADAPT)

    rule()
    log("PV_MIX 稳健性检验：mix=0.30 vs 0.40（最终口径 adapt=%.1f + 分位裕量 q=%.2f/win=%d）"
        % (q3.ADAPT, q3.HEDGE_PARAM, q3.HEDGE_WIN))
    rule()

    acc = int(np.argmax(dates >= pd.Timestamp(q3.OUT_START)))
    D = len(dates)
    bounds = [acc] + [acc + round((D - acc) * k / 3) for k in (1, 2)] + [D]
    spans = [(bounds[k], bounds[k + 1]) for k in range(3)]

    GF = {mx: q3.build_pv_forecast(df3, dates, G, q3.DECIDE_H, mx) for mx in MIXS}

    hdr = ("  %-18s %10s %14s %14s %14s %10s"
           % ("区间", "天数", "mix=0.30", "mix=0.40", "差值(0.30-0.40)", "更优者"))
    log(hdr)
    log("  " + "-" * (len(hdr) + 2))

    wins = {mx: 0 for mx in MIXS}
    tot = {mx: 0.0 for mx in MIXS}
    diffs = []
    for (i0, i1) in spans:
        c = {}
        for mx in MIXS:
            r = run_block(dates, L, G, pi, Lf, GF[mx], X, i0, i1)
            c[mx] = r["cost"]
            tot[mx] += r["cost"]
        d = c[0.30] - c[0.40]
        diffs.append(d)
        w = 0.30 if d < 0 else 0.40
        wins[w] += 1
        log("  %-18s %10d %14.1f %14.1f %+14.1f %10s"
            % ("%s ~ %s" % (dates[i0].strftime("%m-%d"), dates[i1 - 1].strftime("%m-%d")),
               i1 - i0, c[0.30], c[0.40], d, "0.30" if d < 0 else "0.40"))

    log("  " + "-" * (len(hdr) + 2))
    log("  %-18s %10d %14.1f %14.1f %+14.1f"
        % ("三段合计", D - acc, tot[0.30], tot[0.40], tot[0.30] - tot[0.40]))
    log("")
    log("  分段胜负：mix=0.30 胜 %d 段，mix=0.40 胜 %d 段（共 3 段）" % (wins[0.30], wins[0.40]))
    log("  三段差值 = %s（元）；同号则说明优势稳定，反号则落在噪声内）"
        % ", ".join("%+.1f" % d for d in diffs))
    log("  最大单段差值 %.1f 元，跨段标准差 %.1f 元"
        % (max(abs(d) for d in diffs), float(np.std(diffs))))

    rule("结论")
    if wins[0.30] == 3:
        log("  mix=0.30 在三个子区间一致更优 → 优势稳定，可考虑改为 0.30。")
    elif wins[0.40] == 3:
        log("  mix=0.40 反而一致更优 → 全年 0.044% 的优势是窗口效应，保持 0.40。")
    else:
        log("  三段的优劣不一致（0.30 胜 %d 段 : 0.40 胜 %d 段）→ 「0.30 更优」落在噪声内，"
            % (wins[0.30], wins[0.40]))
        log("  全年的 0.044% 优势主要来自单一季节，属窗口效应，不构成改动依据。")
        log("  结论：保持 mix=0.40；文档按「0.2~0.6 为平台、参数不敏感」表述。")

    write_report(TXT)
    log("")
    log("总耗时 %.1f 分钟" % ((time.time() - t0) / 60.0))


if __name__ == "__main__":
    main()
