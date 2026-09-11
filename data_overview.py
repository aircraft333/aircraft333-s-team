# -*- coding: utf-8 -*-
"""四个附件的数据概览（论文「数据说明」一节的可溯源出处）

动机：`模型与口径说明.md` 里描述附件1~4 时引用了一批数据统计量，
（例如「附件2 负载 1995.7–7978.9 kW；光伏 0–10216.2 kW」），
但这些数字此前没有落在任何报告文件里，事后无法溯源。
本脚本把四个附件的关键统计量统一打印并存档，作为那些数字的出处。

不涉及任何求解，秒级完成。

输出：控制台 + 数据概览.txt
"""
import numpy as np
import pandas as pd

import q3                      # 复用 q3.REL_H（预报发布时刻的唯一出处）
from config import *   # noqa: F401,F403

TXT = resolve("数据概览.txt")


def rng(a):
    a = np.asarray(a, float)
    return a.min(), a.max(), a.mean()


def main():
    rule()
    log("四个附件的数据概览")
    log("（本文件是论文「数据说明」一节中各项统计量的出处）")
    rule()

    # ---------------- 附件1 ----------------
    A1 = att1_arrays(load_att1())
    pi1, L1, G1 = A1["price"], A1["load"], A1["pv"]
    rule("一、附件1：某天的电价 / 小区负载 / 光伏预测（10 min 粒度）")
    log(f"  规模：{N_SLOT} 个时段 = 24 h；时间戳按「时段右端点」解释")
    for nm, a, u in (("电价", pi1, "元/kWh"), ("小区负载", L1, "kW"),
                     ("光伏预测功率", G1, "kW")):
        lo, hi, mu = rng(a)
        log(f"  {nm:<10s} {lo:>10.4f} ~ {hi:>10.4f}  均值 {mu:>10.4f}  ({u})")
    tier, lo, hi = classify_tiers(pi1)
    log(f"  峰谷划分（33%/67% 分位）：谷 ≤ {lo:.4f}、峰 ≥ {hi:.4f}；"
        + "  ".join(f"{k} {int((tier == k).sum())} 段" for k in (TIER_V, TIER_F, TIER_P)))

    # ---------------- 附件2 ----------------
    dates, L, G = load_att2()
    rule("二、附件2：全年的小区负载与光伏实际功率")
    log(f"  规模：{len(dates)} 天 × {N_SLOT} 槽 = {len(dates) * N_SLOT:,} 个数据点")
    nan = int(np.isnan(L).sum() + np.isnan(G).sum())
    log(f"  缺失值：{nan} 个" + ("（无缺失）" if nan == 0 else ""))
    for nm, a, u in (("负载", L, "kW"), ("光伏", G, "kW")):
        lo_, hi_, mu = rng(a)
        log(f"  {nm:<6s} {lo_:>10.1f} ~ {hi_:>10.1f}  均值 {mu:>9.1f}  ({u})")
    log(f"  日期范围：{pd.Timestamp(dates[0]).date()} ~ {pd.Timestamp(dates[-1]).date()}")

    # ---------------- 附件3 ----------------
    df3 = load_att3()
    rule("三、附件3：光伏发电功率预报（0/6/12/18 时发布，未来 24 整点）")
    log(f"  发布时刻：" + " / ".join(f"{h}:00" for h in q3.REL_H)
        + "；每次给出之后 24 个整点的预报值")
    log(f"  -> 每 6 小时滚动覆盖一次未来 24 小时，全天共 "
        f"{len(q3.REL_H)} × 24 = {len(q3.REL_H) * 24} 个预报值/天")
    att3_vals = []
    for d in dates[:60]:
        for rel in q3.REL_H:
            v = att3_forecast(df3, d, rel)
            if v is not None:
                att3_vals.extend(np.asarray(v, float).ravel().tolist())
    a3 = np.array(att3_vals)
    log(f"  抽样统计（前 60 天）：{a3.size:,} 个值，"
        f"{a3.min():.1f} ~ {a3.max():.1f} kW，均值 {a3.mean():.1f} kW")
    log("  口径：表中「预报k小时」是发布后第 k 个整点的**瞬时点值**，不是小时平均功率")

    # ---------------- 附件4 ----------------
    import q4
    PI = q4.load_att4()
    rule("四、附件4：全年的实时电价")
    log(f"  规模：{PI.shape[0]} 天 × {PI.shape[1]} 槽")
    lo_, hi_, mu = rng(PI)
    dm = PI.mean(axis=1)
    pv_ratio = (PI.max(axis=1) / np.maximum(PI.min(axis=1), 1e-6))
    log(f"  全局      {lo_:>10.4f} ~ {hi_:>10.4f}  均值 {mu:>9.4f}  (元/kWh)")
    log(f"  逐日均价  {dm.min():>10.4f} ~ {dm.max():>10.4f}  标准差 {dm.std():>7.4f}")
    log(f"  日内峰谷比 均值 {pv_ratio.mean():.2f}（附件1 固定电价为 "
        f"{pi1.max() / pi1.min():.2f}）")
    # 列均值恒等于附件1 电价 —— 这是附件4 生成结构的关键证据
    col_mean = PI.mean(axis=0)
    log(f"  逐时段列均值：与附件1 电价的最大绝对差 = "
        f"{np.abs(col_mean - pi1).max():.2e}  ⇒ 附件4 = 附件1 的分时形状 + 逐日漂移 + 逐槽噪声")

    rule("说明")
    log("  · 以上统计量可用本脚本一键复算，供论文「数据说明」一节引用；")
    log("  · 更细的分析见 q1_数据分析结果.txt（附件1）、q2_负载周周期性.txt（负载周期性）、")
    log("    q2_光伏预报方法.txt（光伏）、q4_电价波动效应分解.txt（附件4 电价）。")

    write_report(TXT)


if __name__ == "__main__":
    main()
