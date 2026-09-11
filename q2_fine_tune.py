# -*- coding: utf-8 -*-
"""问题二裕量精细寻优：分位水平 q × 回看窗口 win
=====================================================================
背景：现行主口径是 q=0.75、win=60，但这两个数来自两次**不同**的搜索：
  · q=0.75 是粗网格（0.60~0.95，步长 0.05 附近）扫出来的
  · win=60 是更早（q=0.85 时代）用 q2_tune.py 搜出来的
两者没有一起重新确认过。本脚本在**全年 334 天**上做两轮精细搜索：

  第一轮：固定 win=60，细扫 q（0.01~0.02 步长）
  第二轮：固定 q=q*，扫 win ∈ {20, 30, 45, 90, 120}（60 已由第一轮给出）

每轮都是全年完整求解（含储能初值逐日连续），约 50 s / 个方案。

运行：python q2_fine_tune.py      （约 8~10 分钟）
"""
import time

import numpy as np

import q2
import q2_sensitivity as s
from config import *

QS = [0.70, 0.73, 0.74, 0.75, 0.76, 0.77, 0.80]   # 第一轮：细扫 q（win=60）
WINS = [20, 30, 45, 90, 120]                        # 第二轮：扫 win（q=q*）
QS2 = [0.72, 0.74, 0.75, 0.76, 0.78, 0.80]         # 第三轮：在最优 win 下重扫 q
WIN0 = 60
TXT = "q2_裕量精细寻优.txt"


def main():
    t0 = time.time()
    A1 = att1_arrays(load_att1())
    pi, L1, G1 = A1["price"], A1["load"], A1["pv"]
    dates, L, G = load_att2()
    F_L, F_G = q2.build_forecast(L, G, L1, G1, q2.FC_SPEC)
    msk = dates >= pd.Timestamp(q2.OUT_START)
    ACC = int(np.argmax(msk))
    err = (L - G) - (F_L - F_G)

    def run(q, win):
        X = q2.build_hedge(err, "quantile", q, win)
        r = s.run_block(pi, L, G, F_L, F_G, X, 5.0, 0, len(L), ACC)
        return float(X[msk].mean()), r

    rule("问题二裕量精细寻优：q × win（全年 334 天）")
    log(f"评价区间 {int(msk.sum())} 天；紧急电价 5π；其余口径同 q2.py 默认")

    # ---------------- 第一轮：细扫 q ----------------
    rule(f"【1】固定 win={WIN0}，细扫分位 q")
    hdr = (f"  {'q':>6s}{'日均裕量':>10s}{'计划购电费':>14s}{'紧急购电量':>12s}"
           f"{'紧急购电费':>12s}{'总购电费':>14s}{'相对最优':>11s}")
    log(hdr)
    log("  " + "-" * (len(hdr) + 2))
    R1 = {}
    for q in QS:
        t = time.time()
        xm, r = run(q, WIN0)
        R1[q] = dict(xm=xm, r=r)
        log(f"  {q:>6.2f}{xm:>10.0f}{r['c_plan']:>14,.1f}{r['q_emg']:>12,.1f}"
            f"{r['c_emg']:>12,.1f}{r['cost']:>14,.1f}{'':>11s}"
            f"   ({time.time() - t:.0f}s)")
    q_best = min(R1, key=lambda k: R1[k]["r"]["cost"])
    c_best = R1[q_best]["r"]["cost"]
    for q in QS:
        log(f"  {q:>6.2f}  相对最优 q={q_best:.2f}："
            f"{R1[q]['r']['cost'] - c_best:+,.1f} 元"
            f"（{(R1[q]['r']['cost'] - c_best) / c_best:+.3%}）")
    log(f"  ⇒ 第一轮最优 q = {q_best:.2f}（日均裕量 {R1[q_best]['xm']:.0f} kW）"
        f"→ {c_best:,.1f} 元")

    # ---------------- 第二轮：扫 win ----------------
    rule(f"【2】固定 q={q_best:.2f}，扫回看窗口 win")
    log(hdr.replace("q", "win", 1))
    log("  " + "-" * (len(hdr) + 2))
    R2 = {WIN0: dict(xm=R1[q_best]["xm"], r=R1[q_best]["r"])}
    for w in WINS:
        t = time.time()
        xm, r = run(q_best, w)
        R2[w] = dict(xm=xm, r=r)
        log(f"  {w:>6d}{xm:>10.0f}{r['c_plan']:>14,.1f}{r['q_emg']:>12,.1f}"
            f"{r['c_emg']:>12,.1f}{r['cost']:>14,.1f}{'':>11s}"
            f"   ({time.time() - t:.0f}s)")
    w_best = min(R2, key=lambda k: R2[k]["r"]["cost"])
    c2 = R2[w_best]["r"]["cost"]
    for w in sorted(R2):
        log(f"  {w:>6d}  相对最优 win={w_best}："
            f"{R2[w]['r']['cost'] - c2:+,.1f} 元"
            f"（{(R2[w]['r']['cost'] - c2) / c2:+.3%}）")
    log(f"  ⇒ 第二轮最优 win = {w_best} 天 → {c2:,.1f} 元")

    # ---------------- 第三轮：在最优 win 下重扫 q ----------------
    rule(f"【3】固定 win={w_best}，重扫 q（验证联立最优点）")
    log(hdr)
    log("  " + "-" * (len(hdr) + 2))
    R3 = {q_best: dict(xm=R2[w_best]["xm"], r=R2[w_best]["r"])}
    for q in QS2:
        if q in R3:
            continue
        t = time.time()
        xm, r = run(q, w_best)
        R3[q] = dict(xm=xm, r=r)
        log(f"  {q:>6.2f}{xm:>10.0f}{r['c_plan']:>14,.1f}{r['q_emg']:>12,.1f}"
            f"{r['c_emg']:>12,.1f}{r['cost']:>14,.1f}{'':>11s}"
            f"   ({time.time() - t:.0f}s)")
    q3b = min(R3, key=lambda k: R3[k]["r"]["cost"])
    c3 = R3[q3b]["r"]["cost"]
    for q in sorted(R3):
        log(f"  {q:>6.2f}  相对最优 q={q3b:.2f}：{R3[q]['r']['cost'] - c3:+,.1f} 元"
            f"（{(R3[q]['r']['cost'] - c3) / c3:+.3%}）")
    log(f"  ⇒ 联立最优点 q = {q3b:.2f}, win = {w_best} → {c3:,.1f} 元")

    # ---------------- 结论 ----------------
    rule("【结论】全年实测最优点")
    log(f"  联立最优：q = {q3b:.2f}，win = {w_best} 天，日均裕量 {R3[q3b]['xm']:.0f} kW")
    log(f"  全年总购电费 {c3:,.1f} 元（计划 {R3[q3b]['r']['c_plan']:,.1f} + "
        f"紧急 {R3[q3b]['r']['c_emg']:,.1f}）")
    log("")
    log(f"  对照：现行主口径 q={q2.HEDGE_PARAM:.2f}, win={q2.HEDGE_WIN} → "
        f"{R1[q2.HEDGE_PARAM]['r']['cost']:,.1f} 元")
    log(f"  报童解析值 q=0.80, win={WIN0} → {R1[0.80]['r']['cost']:,.1f} 元"
        f"（比最优点贵 {R1[0.80]['r']['cost'] - c3:,.1f} 元）")
    log(f"  固定 300 kW 均匀裕量 → 14,121,654.0 元"
        f"（比最优点贵 {14121654.0 - c3:,.1f} 元）")
    log("")
    log("  说明：一年只有 334 个样本，(q, win) 微调带来的差异若小于 0.1%（约 1.4 万元），")
    log("        应视为噪声、不要据此宣称「精确最优」，论文里写解析值 0.8 或实测 0.75 都站得住。")
    log(f"\n总耗时 {time.time() - t0:.1f} 秒")
    write_report(resolve(TXT))


if __name__ == "__main__":
    main()
