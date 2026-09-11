# -*- coding: utf-8 -*-
"""问题三「口径递进」锚点核对：让 README / 文档里的每个数字都有可复现来源

背景：`README.md` 的口径递进表引用了「问题三 hedge=0 的全年 14,236,243.7 元，
见 q3_参数寻优.txt」，但该文件只覆盖区块 120~190，并不含全年 `hedge=0` 的结果
（应该是被后续运行覆盖掉了）。为消除这种「引用了不存在的来源」的问题，
本脚本用最小代价把三个锚点全部复算一遍并落盘：

    ①  hedge=0（无裕量，标量 0）      → 见下
    ②  hedge=300（固定 300 kW 均匀）  → 应与 q3_裕量寻优.txt 的 13,847,475.1 一致
    ③  报童分位 q=0.75/win=60         → 应与主结果 13,589,362.4 一致

其余参数一律取 q3.py 现行默认（mix=0.4, adapt=0.2, decide_h=(0,6,12,18)），
三个锚点共用同一套负载预报与决策时刻，唯一变量是安全裕量的取法。

输出：控制台 + q3_口径递进锚点.txt
"""
import time

import numpy as np
import pandas as pd

import q3
from config import *   # noqa: F401,F403

TXT = resolve("q3_口径递进锚点.txt")

# 文档中记载的期望值，用于自动比对
EXPECT = {
    "固定 300 kW": 13847475.1,
    "报童分位 q=0.75/win=60": 13589362.4,
    "无裕量（hedge=0）": None,      # README 记为 14,236,243.7，待核
}
README_HEDGE0 = 14236243.7


def main():
    t0 = time.time()
    A1 = att1_arrays(load_att1())
    pi, L1 = A1["price"], A1["load"]
    dates, L, G = load_att2()
    df3 = load_att3()

    rule()
    log("问题三 口径递进锚点核对（mix=%.2f, adapt=%.2f, 决策时刻 %s）"
        % (q3.PV_MIX, q3.ADAPT, q3.DECIDE_H))
    log("输出窗口：%s 起 334 天；唯一变量 = 安全裕量取法" % q3.OUT_START)
    rule()

    hdr = ("  %-24s %10s %14s %14s %14s %14s"
           % ("裕量取法", "日均裕量", "计划购电费", "调整净增", "紧急购电费", "全年合计"))
    log(hdr)
    log("  " + "-" * (len(hdr) + 2))

    cases = [("无裕量（hedge=0）", 0.0),
             ("固定 300 kW", 300.0),
             ("报童分位 q=0.75/win=60", None)]
    res = {}
    for tag, h in cases:
        t1 = time.time()
        recs, s = q3.run_year(df3, dates, L, G, pi, L1, None,
                              q3.DECIDE_H, q3.PV_MIX, h, q3.ADAPT, quiet=True)
        if h is None:
            hv = q3.build_hedge_q3(L, L1, dates, q3.HEDGE_PARAM, q3.HEDGE_WIN,
                                   q3.DECIDE_H, q3.LOAD_LAG, q3.ADAPT).mean()
        else:
            hv = h
        res[tag] = s
        log("  %-24s %10.0f %14.1f %14.1f %14.1f %14.1f   (%2.0fs)"
            % (tag, hv, s["cost_plan"], s["cost_dev"] - s["cost_plan"],
               s["cost_emg"], s["cost_total"], time.time() - t1))

    rule("与文档记载值比对")
    for tag, exp in EXPECT.items():
        if exp is None:
            continue
        got = res[tag]["cost_total"]
        d = got - exp
        log("  %-24s 文档 %14.1f   实算 %14.1f   差 %+.1f 元  %s"
            % (tag, exp, got, d, "一致" if abs(d) < 1.0 else "**不一致**"))
    got0 = res["无裕量（hedge=0）"]["cost_total"]
    d0 = got0 - README_HEDGE0
    log("  %-24s README %12.1f   实算 %14.1f   差 %+.1f 元  %s"
        % ("无裕量（hedge=0）", README_HEDGE0, got0, d0,
           "一致" if abs(d0) < 1.0 else "**README 数字需更正**"))

    rule("口径递进（与 README 表对应）")
    b = res["无裕量（hedge=0）"]["cost_total"]
    for tag in ("无裕量（hedge=0）", "固定 300 kW", "报童分位 q=0.75/win=60"):
        s = res[tag]
        log("  %-24s 计划 %11.1f  调整净增 %10.1f  紧急 %10.1f  合计 %11.1f  相对无裕量 %+11.1f (%+.2f%%)"
            % (tag, s["cost_plan"], s["cost_dev"] - s["cost_plan"], s["cost_emg"],
               s["cost_total"], s["cost_total"] - b,
               100.0 * (s["cost_total"] - b) / b))

    write_report(TXT)
    log("")
    log("总耗时 %.1f 分钟" % ((time.time() - t0) / 60.0))


if __name__ == "__main__":
    main()
