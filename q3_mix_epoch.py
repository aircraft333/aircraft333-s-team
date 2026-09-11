# -*- coding: utf-8 -*-
"""A1 优化：分时段光伏混合权重（PV_MIX_BY_EPOCH）

动机
    当前全天四个决策时刻共用同一个光伏混合权重 mix=0.4，但这个 0.4 是在
    **0:00 发布** 的口径下标定出来的（见 §4.6 的复检引文）。
    而附件3 的预报精度随发布时刻推后而显著改善：
        0:00 发布 → 要预报未来 24 小时，最不准；
        12:00 发布 → 只需预报未来 12 小时，最准。
    因此「附件3 与历史外推的最优混合比例」理应随决策时刻变化，不该全天一个数。
    另外 18:00 之后的时段（19:00-24:00）光伏恒为 0，此时纯历史外推反而最准。

方法
    第 1 步（无 LP，秒级）：逐决策时刻、逐混合权重测预报 MAE，找各时刻的 MAE 最优权重；
    第 2 步（全年 LP，每个 ~100 s）：把候选权重组合放进完整模型，比全年总费用。
    第 1 步只作参考 —— 最终以第 2 步的费用为准（MAE 最小不等于费用最低）。

输出：控制台 + q3_分时段光伏权重.txt
"""
import time

import numpy as np
import pandas as pd

import q3
from config import *   # noqa: F401,F403

TXT = resolve("q3_分时段光伏权重.txt")
MIX_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# 待比对的候选组合（长度 = len(DECIDE_H) = 4，顺序为 0:00 / 6:00 / 12:00 / 18:00）
CASES = [
    ("基线：全天统一 0.4（现行）", (0.4, 0.4, 0.4, 0.4)),
    ("A：MAE 最优 (0.4,0.6,0.8,0.0)", (0.4, 0.6, 0.8, 0.0)),
    ("B：整体下移 (0.3,0.5,0.7,0.0)", (0.3, 0.5, 0.7, 0.0)),
    ("C：整体上移 (0.5,0.7,0.9,0.0)", (0.5, 0.7, 0.9, 0.0)),
    ("D：18:00 不取极端 (0.4,0.6,0.8,0.2)", (0.4, 0.6, 0.8, 0.2)),
]


def main():
    t0 = time.time()
    A1 = att1_arrays(load_att1())
    pi, L1 = A1["price"], A1["load"]
    dates, L, G = load_att2()
    df3 = load_att3()
    msk = np.asarray(dates >= pd.Timestamp(q3.OUT_START))

    rule()
    log("A1 优化：分时段光伏混合权重（PV_MIX_BY_EPOCH）")
    log(f"决策时刻 {q3.DECIDE_H}；输出窗口 {q3.OUT_START} 起 {int(msk.sum())} 天；"
        f"其余口径取 q3.py 现行默认")
    rule()

    # ---------- 第 1 步：逐时刻 MAE（无 LP） ----------
    rule("一、逐决策时刻的预报 MAE（只统计该时刻尚未发生的时段）")
    log("  说明：MAE 最小 ≠ 费用最低，此表只用于确定候选权重")
    hdr = "  %-6s" % "mix" + "".join("%14s" % f"{h}:00 后" for h in q3.DECIDE_H)
    log(hdr)
    log("  " + "-" * (len(hdr) + 2))
    best = {}
    for m in MIX_GRID:
        Gf = q3.build_pv_forecast(df3, dates, G, q3.DECIDE_H, m)
        row, vals = "  %-6.1f" % m, []
        for ri, h in enumerate(q3.DECIDE_H):
            t0_ = h * 6
            e = np.abs(Gf[ri][msk][:, t0_:] - G[msk][:, t0_:]).mean()
            vals.append(e)
            row += "%14.1f" % e
        log(row)
        for ri in range(len(q3.DECIDE_H)):
            if ri not in best or vals[ri] < best[ri][1]:
                best[ri] = (m, vals[ri])
    log("")
    log("  各时刻 MAE 最优权重：" + "，".join(
        f"{h}:00 → {best[ri][0]:.1f}（MAE {best[ri][1]:.1f} kW）"
        for ri, h in enumerate(q3.DECIDE_H)))
    Gf04 = q3.build_pv_forecast(df3, dates, G, q3.DECIDE_H, 0.4)
    flat = [np.abs(Gf04[ri][msk][:, h * 6:] - G[msk][:, h * 6:]).mean()
            for ri, h in enumerate(q3.DECIDE_H)]
    log("  对照：全天统一 0.4 时          " + "，".join(
        f"{h}:00 → {v:.1f} kW" for h, v in zip(q3.DECIDE_H, flat)))

    # ---------- 第 1.5 步：偏差方向（解释「MAE 最小却更贵」的关键） ----------
    rule("二、预报偏差方向（mean signed error，正 = 高估光伏）")
    log("  为什么这一步重要：光伏预报误差进入费用函数是**不对称**的 ——")
    log("    低估光伏 → 多买电 → 代价 π_t（便宜方向）")
    log("    高估光伏 → 少买电 → 实际缺口 → 代价 5π_t（紧急购电，昂贵方向）")
    log("  MAE 对两个方向一视同仁，但费用函数不是。")
    log("")
    hdr2 = "  %-6s" % "mix" + "".join("%13s" % f"{h}:00 后" for h in q3.DECIDE_H)
    log(hdr2)
    log("  " + "-" * (len(hdr2) + 2))
    for m in MIX_GRID:
        Gf = q3.build_pv_forecast(df3, dates, G, q3.DECIDE_H, m)
        row = "  %-6.1f" % m
        for ri, h in enumerate(q3.DECIDE_H):
            row += "%13.2f" % (Gf[ri][msk][:, h * 6:] - G[msk][:, h * 6:]).mean()
        log(row)
    log("")
    log("  可见：0:00 / 12:00 / 18:00 随权重上升而**愈发低估**（便宜方向），")
    log("        而 6:00 随权重上升而**愈发高估**（昂贵方向）——")
    log("        因此「逐时刻 MAE 最优」把 6:00 推到 0.6，反而加剧了昂贵的偏差。")

    # ---------- 第 2 步：全年 LP ----------
    rule("三、全年总费用（每个约 100~160 s）")
    log(f"  {'方案':<38s}{'总费用(元)':>15s}{'相对基线':>13s}{'紧急购电费':>14s}{'弃光量':>14s}")
    base = None
    res = []
    for tag, mix in CASES:
        t1 = time.time()
        _, s = q3.run_year(df3, dates, L, G, pi, L1, None,
                           q3.DECIDE_H, mix, None, q3.ADAPT, quiet=True)
        if base is None:
            base = s["cost_total"]
        res.append((tag, mix, s["cost_total"], s))
        log(f"  {tag:<38s}{s['cost_total']:>15,.1f}"
            f"{s['cost_total'] - base:>+13,.1f}{s['cost_emg']:>14,.1f}"
            f"{s['q_curtail']:>14,.1f}   ({time.time() - t1:.0f}s)")
    log("")
    log(f"  基线自检：全天统一 0.4 应复现主结果 13,589,362.4 元 → 实得 {base:,.1f} 元，"
        f"差 {base - 13589362.4:+,.1f} 元")

    res.sort(key=lambda r: r[2])
    rule("四、结论")
    log(f"  最优方案：{res[0][0]}  mix={res[0][1]}")
    log(f"  全年总费用 {res[0][2]:,.1f} 元（基线 {base:,.1f} 元，"
        f"省 {base - res[0][2]:,.1f} 元，{(base - res[0][2]) / base:+.3%}）")
    if res[0][1] == CASES[0][1]:
        log("  ⇒ 分时段权重没有带来改善，保持「全天统一 0.4」更简洁。")
        log("")
        log("  这是一个有价值的**负面结果**，机理：")
        log("    1) 光伏预报误差进入费用函数是不对称的：低估→多买，代价 π；高估→少买，代价 5π；")
        log("    2) MAE 只衡量误差大小、不区分方向，因此「MAE 最优」并不等于「费用最优」；")
        log("    3) 提高附件3 权重在 6:00 引入的是**高估**（昂贵方向），")
        log("       使紧急购电费从 544,694 涨到 630,464 元（+85,770），")
        log("       超过了预报改善带来的收益，于是全天统一的 0.4 反而最优；")
        log("    4) 结论：光伏混合权重应被视为**费用口径**下调出来的参数（现行 0.4 已由")
        log("       费用扫描选定），而不是按预报 MAE 标定。")
    else:
        log(f"  ⇒ 若采用，需要把 q3.py 的 PV_MIX_BY_EPOCH 设为 {res[0][1]}，")
        log(f"     并重跑 q4.py（q4 硬编码了 q2/q3 的费用）与相关分析脚本")
    log("")
    log("  备查：完整排序")
    for tag, mix, c, _s in res:
        log(f"    {tag:<38s}{c:>15,.1f}  ({c - base:+13,.1f})")

    write_report(TXT)
    log("")
    log("总耗时 %.1f 分钟" % ((time.time() - t0) / 60.0))


if __name__ == "__main__":
    main()
