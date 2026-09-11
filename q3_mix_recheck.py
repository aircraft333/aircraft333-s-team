# -*- coding: utf-8 -*-
"""复检 光伏混合权重 PV_MIX：在最终口径下重扫

【为什么要复检】
  q3_参数寻优.txt 用的是「坐标轮换（one-factor-at-a-time）」搜索：
    第 1 步扫 mix      时基线是 hedge=0, adapt=0；
    第 2 步扫 hedge    时沿用了 mix=0.4；
    第 3 步扫 adapt    时沿用了 mix=0.4, hedge=200。
  也就是说 mix=0.4 是在「裕量=0、无自适应」的基线上定出来的，
  之后裕量换成报童分位规则（值远大于 200 kW）、adapt=0.2 打开，
  mix 一直没在新口径下复检过。坐标轮换不保证收敛到联合最优，
  本脚本就是补这一步：在最终口径（adapt=0.2 + 报童分位裕量）下重扫 mix。

【另一个目的】
  顺便在 adapt=0.0 上跑一个参照点（mix=0.4），直接量化 mix 与 adapt 的
  交互作用有多大。

用法：
    python q3_mix_recheck.py            # 全年复检（默认）
输出：控制台 + q3_光伏权重复检.txt
"""
import time

import numpy as np
import pandas as pd

import q3
from config import *   # noqa: F401,F403

TXT = resolve("q3_光伏权重复检.txt")

MIXS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0]   # 覆盖区间 + 最优点附近加密
REF_ADAPT0_MIX = 0.4                                # adapt=0 的参照点


def run_one(df3, dates, L, G, pi, L1, mix, adapt):
    """按指定口径跑全年，返回 (汇总, 耗时秒)"""
    t = time.time()
    _, s = q3.run_year(df3, dates, L, G, pi, L1, None,
                       q3.DECIDE_H, mix, None, adapt, quiet=True)
    return s, time.time() - t


def main():
    A1 = att1_arrays(load_att1())
    pi, L1 = A1["price"], A1["load"]
    dates, L, G = load_att2()
    df3 = load_att3()

    rule()
    log("问题三 光伏混合权重 PV_MIX 复检（最终口径：adapt=%.1f + 报童分位裕量 q=%.2f/win=%d）"
        % (q3.ADAPT, q3.HEDGE_PARAM, q3.HEDGE_WIN))
    log("输出窗口：%s 起，共 %d 天；决策时刻 %s；历史外推 %d 天"
        % (q3.OUT_START, int((dates >= pd.Timestamp(q3.OUT_START)).sum()),
           q3.DECIDE_H, q3.PV_HIST_N))
    rule()

    res = {}

    # ---------- 1. 最终口径下重扫 mix ----------
    rule("一、最终口径（adapt=0.2 + 分位裕量）下重扫 mix")
    for mx in MIXS:
        s, dt = run_one(df3, dates, L, G, pi, L1, mx, q3.ADAPT)
        res[(q3.ADAPT, mx)] = s
        log("  mix = %.2f   总费用 %12.1f 元   紧急购电费 %10.1f 元   "
            "紧急电量 %8.1f kWh   弃光 %9.1f kWh   (%2.0fs)"
            % (mx, s["cost_total"], s["cost_emg"], s["q_emg"],
               s["q_curtail"], dt))
    base = res[(q3.ADAPT, 0.4)]
    log("  基准自检：mix=0.4 应为 13,589,362.4 元 → 实得 %.1f 元，差 %.1f 元"
        % (base["cost_total"], base["cost_total"] - 13589362.4))

    best = min(MIXS, key=lambda m: res[(q3.ADAPT, m)]["cost_total"])
    bcost = res[(q3.ADAPT, best)]["cost_total"]
    log("  → 最终口径最优 mix = %.2f（总费用 %.1f 元）" % (best, bcost))
    log("  → 旧口径选定的 0.40 在最终口径下的代价：%.1f 元（%.3f%%）"
        % (bcost - base["cost_total"],
           100.0 * (bcost - base["cost_total"]) / base["cost_total"]))

    # ---------- 2. adapt=0 参照点，量化交互 ----------
    rule("二、交互作用量化（adapt=0.0 参照点）")
    s0, dt = run_one(df3, dates, L, G, pi, L1, REF_ADAPT0_MIX, 0.0)
    res[(0.0, REF_ADAPT0_MIX)] = s0
    log("  adapt=0.0  mix=0.40   总费用 %12.1f 元   紧急购电费 %10.1f 元   (%2.0fs)"
        % (s0["cost_total"], s0["cost_emg"], dt))
    log("  adapt=0.2  mix=0.40   总费用 %12.1f 元   紧急购电费 %10.1f 元"
        % (base["cost_total"], base["cost_emg"]))
    log("  → adapt 0.0→0.2 的净收益：%.1f 元（紧急购电降 %.1f 元，降幅 %.1f%%）"
        % (s0["cost_total"] - base["cost_total"],
           s0["cost_emg"] - base["cost_emg"],
           100.0 * (s0["cost_emg"] - base["cost_emg"]) / s0["cost_emg"]))

    # ---------- 3. 敏感性对比：mix 的边际价值 ----------
    rule("三、mix 的边际价值（相对 mix=0.4，正值=更贵）")
    log("  %-8s %14s %14s" % ("mix", "最终口径", "变化"))
    for mx in MIXS:
        d = res[(q3.ADAPT, mx)]["cost_total"] - base["cost_total"]
        flag = "  ← 最优" if mx == best else ""
        log("  %-8.2f %14.1f %+14.1f%s"
            % (mx, res[(q3.ADAPT, mx)]["cost_total"], d, flag))
    vals = [res[(q3.ADAPT, m)]["cost_total"] for m in MIXS]
    log("  全年极差（worst−best）：%.1f 元 = 总费用的 %.3f%%"
        % (max(vals) - min(vals), 100.0 * (max(vals) - min(vals)) / min(vals)))
    inner = [res[(q3.ADAPT, m)]["cost_total"] for m in MIXS if 0.2 <= m <= 0.6]
    log("  0.2~0.6 区间内极差：%.1f 元 = 总费用的 %.3f%%"
        % (max(inner) - min(inner), 100.0 * (max(inner) - min(inner)) / min(inner)))

    # ---------- 4. 结论 ----------
    rule("四、结论")
    if best == 0.4:
        log("  复检通过：最终口径下 mix=0.40 仍是最优，旧结论成立。")
    else:
        log("  复检未通过：最终口径下最优 mix=%.2f，与旧口径选定的 0.40 不一致。" % best)
        log("  0.40 改为 %.2f 可再省 %.1f 元（%.3f%%）。"
            % (best, base["cost_total"] - bcost,
               100.0 * (base["cost_total"] - bcost) / base["cost_total"]))
    log("  曲线平坦度：0.2~0.6 区间极差 %.1f 元 ≈ %.3f%%，属参数不敏感区，"
        % (max(inner) - min(inner), 100.0 * (max(inner) - min(inner)) / min(inner)))
    log("  故 mix 的取值本身对结果影响很小，但口径必须写清「在哪个基线上下扫出」。")

    write_report(TXT)


if __name__ == "__main__":
    main()
