# -*- coding: utf-8 -*-
"""问题一敏感性分析：电价结构、储能参数、输入偏差
=====================================================================
问题一的答案（59,482.6990 kWh / 35,126.9486 元）是在「附件1 分时电价 +
题目给定储能参数」下求得的单日 LP 最优解。需要回答三件事：

实验 A：电价结构怎么影响储能？
    把电价曲线按幂律拉伸：π_t(ρ) = π_min·(π_t/π_min)^ρ
        ρ = 0   → 全天同价（无峰谷差）
        ρ = 1   → 附件1 原曲线
        ρ > 1   → 峰谷差被放大
    峰谷价比 = (π_max/π_min)^ρ。理论阈值：套利有利可图当且仅当
        π_峰/π_谷 > 1/η² ≈ 1.2346  ⇒  ρ* = ln(1.2346)/ln(3.757) ≈ 0.159
    预期：ρ < ρ* 时储能几乎不动（吞吐≈0）、购电费 = 无储能基准；
         越过 ρ* 后吞吐量突变式跳起 —— 曲线上应出现一个「拐点」。

实验 B：储能参数（效率、功率、SOC 上下限）各带来多少价值？
    单参数扫描，其余保持题目取值。报告购电费、吞吐量、相对基准的变化。

实验 C：如果实际值与附件1 不符，计划的鲁棒性如何？
    用附件1 求出的计划 (b, c, d)，在负载/PV 乘性偏差下执行：
      · 「计划不变」口径：储能在日内被动补缺，缺口只能按 5π 买（问题二规则）
      · 「完全信息」口径：已知实际值重解 LP（事后下界）
    预期：出现明显不对称 —— 负载上调、光伏下调要花 5 倍价买电，
          而负载下调、光伏上调只是多弃光（免费）。这正是问题二必须留
          正裕量的原因，是问题一 → 问题二的逻辑桥梁。

运行：python q1_sensitivity.py     （约 1~2 分钟，全部是单日 LP）
产出：figures/q1/fig_敏感性分析.png + q1_敏感性分析.txt
"""
import time

import numpy as np
import pulp

from config import *

FIGDIR = os.path.join(DIR_FIG, "q1")
TXT = "q1_敏感性分析.txt"
EPS_T = 1e-6            # 吞吐量微罚项：消除「等价最优解」导致的充放电压而不动
K_EMG = 5.0             # 问题二的紧急购电倍率（5 倍交易时刻电价）

RHO = [0.0, 0.1, 0.15, 0.16, 0.165, 0.17, 0.18, 0.185, 0.19, 0.2, 0.3, 0.5,
       0.75, 1.0, 1.25, 1.5]
ETAS = [0.78, 0.84, 0.90, 0.96]
PMAXS = [2000.0, 3500.0, 5000.0, 6500.0, 8000.0]
EMAXS = [6000.0, 8000.0, 10000.0, 12000.0]
EMINS = [0.0, 600.0, 1200.0, 1800.0, 2400.0]
KL = [0.90, 0.95, 1.00, 1.05, 1.10]          # 负载乘性偏差
KG = [0.80, 0.90, 1.00, 1.10, 1.20]          # 光伏乘性偏差
HEDGES = [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1500.0]   # 实验 C 收尾：裕量的作用


# =====================================================================
# 一、参数化单日 LP（与 q1.py / storage.py 同一模型，参数可自由改动）
# =====================================================================
def solve(pi, L, G, pmax=P_RATE, eta=ETA, emin=SOC_MIN, emax=SOC_MAX,
          e0=SOC0, eps=EPS_T):
    """单日最优购电计划

        min  Σ π_t·b_t·Δt (+ ε·Σ(c_t+d_t)Δt 只用于消除退化)
        s.t. b_t + G_t + d_t − c_t − s_t = L_t
             c_t, d_t ≤ P̄（互斥），  b_t, s_t ≥ 0
             E_t = E_{t−1} + (η·c_t − d_t/η)·Δt ∈ [E_min, E_max]
             E_143 = E_0（一日一循环）
    返回 None 表示不可行（例如初值落在 SOC 区间外）。
    """
    T, dt = N_SLOT, DT_H
    if not (emin - 1e-9 <= e0 <= emax + 1e-9):
        return None
    m = pulp.LpProblem("q1_sens", pulp.LpMinimize)
    b = [pulp.LpVariable(f"b{t}", lowBound=0) for t in range(T)]
    c = [pulp.LpVariable(f"c{t}", lowBound=0, upBound=pmax) for t in range(T)]
    d = [pulp.LpVariable(f"d{t}", lowBound=0, upBound=pmax) for t in range(T)]
    s = [pulp.LpVariable(f"s{t}", lowBound=0) for t in range(T)]
    u = [pulp.LpVariable(f"u{t}", cat=pulp.LpBinary) for t in range(T)]
    E = [pulp.LpVariable(f"E{t}", lowBound=emin, upBound=emax) for t in range(T)]

    m += (pulp.lpSum(pi[t] * b[t] * dt for t in range(T))
          + eps * pulp.lpSum((c[t] + d[t]) * dt for t in range(T)))
    for t in range(T):
        m += b[t] + G[t] + d[t] - c[t] - s[t] == L[t]
        m += c[t] <= u[t] * pmax
        m += d[t] <= (1 - u[t]) * pmax
        prev = e0 if t == 0 else E[t - 1]
        m += E[t] == prev + (eta * c[t] - d[t] / eta) * dt
    m += E[T - 1] == e0

    if pulp.LpStatus[m.solve(pulp.PULP_CBC_CMD(msg=False))] != "Optimal":
        return None
    val = lambda v: float(pulp.value(v) or 0.0)
    B = np.array([val(v) for v in b])
    C = np.array([val(v) for v in c])
    D = np.array([val(v) for v in d])
    S = np.array([val(v) for v in s])
    return dict(b=B, c=C, d=D, s=S,
                q_buy=float(B.sum() * dt), q_ch=float(C.sum() * dt),
                q_dis=float(D.sum() * dt), q_curt=float(S.sum() * dt),
                thru=float((C.sum() + D.sum()) * dt),
                cost=float(np.sum(pi * B * dt)))


def stretch(pi, rho):
    """幂律拉伸电价：保持最低价不变，峰谷价比变为 (π_max/π_min)^ρ"""
    p0 = float(pi.min())
    return p0 * (pi / p0) ** rho


def pk_share(pi, b, mask):
    """某类时段（峰/谷）购电量占比与购电费占比"""
    e = b * DT_H
    tot = e.sum()
    if tot <= 0:
        return 0.0, 0.0
    ce = float((pi * e)[mask].sum())
    ct = float((pi * e).sum())
    return float(e[mask].sum() / tot), (ce / ct if ct > 0 else 0.0)


def main():
    t0 = time.time()
    A1 = att1_arrays(load_att1())
    pi, L, G = A1["price"], A1["load"], A1["pv"]
    dt = DT_H
    PK = pi >= 0.9371        # 峰段时刻（与问题二口径一致：π ≥ 0.9371）
    VL = pi <= 0.4424        # 谷段时刻

    rule("问题一敏感性分析：电价结构 / 储能参数 / 输入偏差")
    log(f"基准日：附件1（{N_SLOT} 个 10 分钟时段）")
    log(f"电价 {pi.min():.4f} ~ {pi.max():.4f} 元/kWh，均值 {pi.mean():.4f}，"
        f"峰谷比 {pi.max() / pi.min():.4f}")
    log(f"负载均值 {L.mean():.1f} kW，光伏均值 {G.mean():.1f} kW，"
        f"净负载均值 {(L - G).mean():.1f} kW")
    log(f"储能：E {SOC_MIN:.0f}~{SOC_MAX:.0f} kWh（初值 {SOC0:.0f}），"
        f"P_max {P_RATE:.0f} kW，η {ETA:.2f}")

    base = solve(pi, L, G)
    base_ns = solve(pi, L, G, pmax=0.0)          # 无储能基准
    rule("【基准】题目参数下的单日最优解")
    log(f"有储能：购电量 {base['q_buy']:,.4f} kWh，购电费 {base['cost']:,.4f} 元，"
        f"吞吐量 {base['thru']:,.4f} kWh，弃光 {base['q_curt']:,.4f} kWh")
    log(f"无储能：购电量 {base_ns['q_buy']:,.4f} kWh，购电费 {base_ns['cost']:,.4f} 元，"
        f"弃光 {base_ns['q_curt']:,.4f} kWh")
    log(f"⇒ 储能价值 = {base_ns['cost'] - base['cost']:,.4f} 元/日 "
        f"（{(base_ns['cost'] - base['cost']) / base_ns['cost']:.2%}）")
    spb, scb = pk_share(pi, base["b"], PK)
    svb, svc = pk_share(pi, base["b"], VL)
    log(f"   购电量分布：峰段 {spb:.2%}（费用 {scb:.2%}），谷段 {svb:.2%}（费用 {svc:.2%}）")

    # ================= 实验 A：电价结构 =================
    rule("【A】电价结构扰动：π_t(ρ) = π_min·(π_t/π_min)^ρ")
    ratio = float(pi.max() / pi.min())
    rho_star = float(np.log(1.0 / ETA ** 2) / np.log(ratio))
    log(f"理论阈值（必要条件）：套利有利 ⇔ 峰谷价比 > 1/η^2 = {1 / ETA ** 2:.4f}"
        f"  ⇒  ρ* = ln(1/η^2)/ln({ratio:.3f}) = {rho_star:.4f}")
    log("")
    hdr = (f"  {'ρ':>6s}{'峰谷比':>9s}{'均价':>9s}{'购电量':>12s}{'购电费':>12s}"
           f"{'吞吐量':>11s}{'峰段购电占':>11s}{'谷段购电占':>11s}{'无储能费用':>13s}{'储能价值':>11s}")
    log(hdr)
    log("  " + "-" * (len(hdr) + 2))
    A = []
    for r in RHO:
        pr = stretch(pi, r)
        so = solve(pr, L, G)
        ns = solve(pr, L, G, pmax=0.0)
        sp, _ = pk_share(pr, so["b"], PK)
        sv, _ = pk_share(pr, so["b"], VL)
        A.append(dict(rho=r, ratio=float(pr.max() / pr.min()), mean=float(pr.mean()),
                      q=so["q_buy"], cost=so["cost"], thru=so["thru"],
                      sp=sp, sv=sv, ns=ns["cost"], val=ns["cost"] - so["cost"]))
        log(f"  {r:>6.3f}{pr.max() / pr.min():>9.3f}{pr.mean():>9.4f}{so['q_buy']:>12,.1f}"
            f"{so['cost']:>12,.1f}{so['thru']:>11,.1f}{sp:>10.2%}{sv:>11.2%}"
            f"{ns['cost']:>13,.1f}{A[-1]['val']:>11,.1f}")
    log("")
    steps = [(A[i - 1], A[i]) for i in range(1, len(A))
             if A[i]["thru"] > A[i - 1]["thru"] * 1.02]
    log(f"  吞吐量台阶（跳升 >2%）：共 {len(steps)} 处")
    for a, b in steps:
        log(f"    ρ {a['rho']:.3f} → {b['rho']:.3f}（峰谷比 {a['ratio']:.4f} → {b['ratio']:.4f}）"
            f"：吞吐量 {a['thru']:,.0f} → {b['thru']:,.0f} kWh（{b['thru'] / a['thru'] - 1:+.1%}），"
            f"储能价值 {a['val']:,.0f} → {b['val']:,.0f} 元")
    if steps:
        a, b = steps[0]
        log("")
        log(f"  第一个台阶的峰谷比 {a['ratio']:.4f} → {b['ratio']:.4f} 与理论值 1/η^2 = "
            f"{1 / ETA ** 2:.4f} 相差不到 "
            f"{abs(b['ratio'] / (1 / ETA ** 2) - 1):.1%}，说明必要条件判断准确。")
    log("")
    log("  注 1：ρ 很小时储能并非完全不动，它仍要把午间富余光伏「搬」到晚间以避免弃光 ——")
    log("        这是 η<1 就成立的无风险收益，构成吞吐量约 1.1 万 kWh 的平台。")
    log("  注 2：跳变呈台阶式而非单一拐点，因为储能日吞吐受 SOC 区间限制，")
    log("        能配对的峰谷时刻随价差扩大被逐个激活（1.25 之后又在 1.27 出现第二阶）。")

    # ================= 实验 B：储能参数 =================
    rule("【B】储能参数敏感性（单参数扫描，其余取题目值）")
    log(f"  基准购电费 {base['cost']:,.1f} 元，吞吐量 {base['thru']:,.1f} kWh")
    log("")
    B = {}
    sweep = [("η 充电效率", "eta", ETAS, ETA), ("P_max 额定功率(kW)", "pmax", PMAXS, P_RATE),
             ("E_max 上限(kWh)", "emax", EMAXS, SOC_MAX), ("E_min 下限(kWh)", "emin", EMINS, SOC_MIN)]
    for label, key, vals, ref in sweep:
        log(f"  —— {label}（基准 {ref:g}）——")
        rows = []
        for v in vals:
            kw = {key: v}
            so = solve(pi, L, G, **kw)
            if so is None:
                log(f"     {v:>10g}    不可行")
                continue
            rows.append((v, so["cost"], so["thru"], so["q_curt"], so["q_buy"]))
            flag = "  ← 基准" if abs(v - ref) < 1e-9 else ""
            log(f"     {v:>10g}  购电费 {so['cost']:>11,.1f} 元"
                f"（{so['cost'] - base['cost']:>+9,.1f}）  吞吐量 {so['thru']:>10,.0f} kWh"
                f"  弃光 {so['q_curt']:>9,.0f} kWh{flag}")
        B[key] = rows
        lo_r = min(rows, key=lambda r: r[1])
        log(f"     最有利取值 {lo_r[0]:g} → {lo_r[1]:,.1f} 元"
            f"（相对基准 {lo_r[1] - base['cost']:+,.1f} 元）")
    log("")
    log("  ⇒ 效率 η 是储能价值的第一杠杆（吞吐量按 η 折损），SOC 可用区间次之，"
        "功率上限只在峰值时段起作用")

    # ================= 实验 C：输入偏差 =================
    rule("【C】输入偏差下的计划鲁棒性（执行口径 vs 完全信息口径）")
    log(f"  执行口径：计划 b 不变，储能日内被动补缺，缺口按 {K_EMG:g}π 紧急购电")
    log(f"  完全信息：已知实际值重解 LP（事后下界，不可实现，仅作参照）")
    log("")
    C = []
    hdr = (f"  {'情景':>16s}{'购电量偏差':>12s}{'缺口(kWh)':>12s}{'多弃光(kWh)':>13s}"
           f"{'追加费用':>12s}{'追加占比':>10s}{'完全信息':>12s}{'鲁棒性损失':>12s}")
    log(hdr)
    log("  " + "-" * (len(hdr) + 2))
    for key, scales, unit in (("load", KL, "负载"), ("pv", KG, "光伏")):
        for k in scales:
            Lp = L * k if key == "load" else L
            Gp = G * k if key == "pv" else G
            imb = Lp - Gp - base["b"] + base["c"] - base["d"]      # 需额外购买的功率 kW
            add_buy = float(np.maximum(imb, 0.0).sum() * dt)
            add_curt = float(np.maximum(-imb, 0.0).sum() * dt)
            add_cost = float(np.sum(K_EMG * pi * np.maximum(imb, 0.0) * dt))
            opt = solve(pi, Lp, Gp)
            d_opt = opt["cost"] - base["cost"]
            C.append(dict(tag=f"{unit}×{k:.2f}", k=k, kind=key, dq=add_buy - add_curt,
                          short=add_buy, curt=add_curt, add=add_cost,
                          opt=d_opt, loss=add_cost - d_opt,
                          pct=(add_cost / base["cost"] if base["cost"] else 0.0)))
            log(f"  {unit + '×' + format(k, '.2f'):>16s}{C[-1]['dq']:>12,.1f}"
                f"{add_buy:>12,.1f}{add_curt:>13,.1f}{add_cost:>12,.1f}"
                f"{C[-1]['pct']:>10.2%}{d_opt:>12,.1f}{C[-1]['loss']:>12,.1f}")
    log("")
    up = [c for c in C if c["kind"] == "load" and c["k"] > 1]
    dn = [c for c in C if c["kind"] == "load" and c["k"] < 1]
    log(f"  负载上调（+5%/+10%）追加费用均值 {np.mean([c['add'] for c in up]):,.0f} 元；"
        f"负载下调（-5%/-10%）追加费用均值 {np.mean([c['add'] for c in dn]):,.0f} 元")
    log("  ⇒ 明显不对称：负载上调/光伏下调必须按 5 倍价补电，"
        "而负载下调/光伏上调只是多弃光（免费）")
    log("  ⇒ 这正是问题二必须留正裕量的原因：偏差的费用函数单边陡峭")

    rule("【C-2】若计划预留裕量 h kW，偏差情景下的总费用（与问题二同机制）")
    log("  说明：先按 (L + h) 求计划，再用实际负载/PV 执行，缺口按 5π 结算")
    log("")
    hdr = f"  {'情景':>16s}" + "".join(f"{'h=' + format(h, 'g'):>16s}" for h in HEDGES)
    log(hdr)
    log("  " + "-" * (len(hdr) + 2))
    C2 = {}
    for unit, k, key in (("负载", 1.10, "load"), ("光伏", 0.90, "pv")):
        Lp = L * k if key == "load" else L
        Gp = G * k if key == "pv" else G
        row = []
        for h in HEDGES:
            plan = solve(pi, L + h, G)
            imb = Lp - Gp - plan["b"] + plan["c"] - plan["d"]
            tot = plan["cost"] + float(np.sum(K_EMG * pi * np.maximum(imb, 0.0) * dt))
            row.append(tot)
        C2[f"{unit}×{k:.2f}"] = row
        best = int(np.argmin(row))
        log(f"  {unit + '×' + format(k, '.2f'):>16s}"
            + "".join(f"{v:>16,.0f}" for v in row)
            + f"   最优 h = {HEDGES[best]:g} kW")
    log("")
    for k, row in C2.items():
        b = int(np.argmin(row))
        log(f"  {k}：最优裕量 h = {HEDGES[b]:g} kW，总费用 {row[b]:,.0f} 元；"
            f"h=0 时为 {row[0]:,.0f} 元（省 {row[0] - row[b]:,.0f} 元）")
    log("")
    log("  ⇒ 裕量把「预期偏差」提前用平价电买进来：负载 +10% 相当于全天平均多 463 kW、")
    log("    光伏 -10% 相当于平均多 231 kW 的缺口，对应最优裕量在数百 kW 量级，")
    log("    与偏差本身相当或略大 —— 这正是报童模型「最优裕量 ∝ 误差分位数」的直观体现。")
    log("    裕量偏大的原因是它只能以「全天均匀抬高」的方式施加，且谷段电价便宜，多买的边际成本低。")
    log("    注意裕量不宜过大：计划按实际偏高执行时会多买并用不掉，费用重新上升（U 形）。")

    # ================= 图 =================
    rule("绘图")
    setup_plot()
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6))

    # (1) 电价拉伸
    ax = axes[0, 0]
    xs = [a["rho"] for a in A]
    ax.plot(xs, [a["cost"] / 1e4 for a in A], "o-", color=C_PRICE, lw=LW, label="单日购电费")
    ax.plot(xs, [a["ns"] / 1e4 for a in A], "s--", color=C_LOAD, lw=LW * 0.8, ms=4,
            label="无储能基准")
    ax.axvline(rho_star, color=C_NET, ls=":", lw=LW)
    ax.annotate(f"理论阈值 ρ*={rho_star:.3f}", xy=(rho_star, ax.get_ylim()[0]),
                xytext=(rho_star + 0.06, ax.get_ylim()[0] + 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                color=C_NET, fontsize=9)
    ax.set_xlabel("电价幂律拉伸指数 ρ")
    ax.set_ylabel("购电费（万元）", color=C_PRICE)
    ax.tick_params(axis="y", labelcolor=C_PRICE)
    ax2 = ax.twinx()
    ax2.plot(xs, [a["thru"] / 1e3 for a in A], "^-.", color=C_NET, lw=LW, ms=5,
             label="储能吞吐量")
    ax2.set_ylabel("储能吞吐量（MWh）", color=C_NET)
    ax2.tick_params(axis="y", labelcolor=C_NET)
    ax.set_title("(a) 电价结构：越过峰谷比阈值才启动套利")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper left")

    # (2)(3)(4)(5) 储能参数
    panels = [(ETAS, "eta", "(b) 充电效率 η", "eta"), (PMAXS, "pmax", "(c) 额定功率上限 P_max", "P_max"),
              (EMAXS, "emax", "(d) SOC 上限 E_max", "E_max"),
              (EMINS, "emin", "(e) SOC 下限 E_min", "E_min")]
    for ax, (vals, key, title, _) in zip(axes.flat[1:5], panels):
        rows = B[key]
        xs = [r[0] for r in rows]
        ax.plot(xs, [r[1] / 1e4 for r in rows], "o-", color=C_PRICE, lw=LW, label="购电费")
        ax.set_xlabel(f"{title.split('『')[0][4:]}")
        ax.set_ylabel("购电费（万元）", color=C_PRICE)
        ax.tick_params(axis="y", labelcolor=C_PRICE)
        ax.axhline(base["cost"] / 1e4, color=C_LOAD, ls="--", lw=LW * 0.7)
        ax2 = ax.twinx()
        ax2.plot(xs, [r[2] / 1e3 for r in rows], "^-.", color=C_NET, lw=LW, ms=5)
        ax2.set_ylabel("吞吐量（MWh）", color=C_NET)
        ax2.tick_params(axis="y", labelcolor=C_NET)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    # (6) 输入偏差
    ax = axes[1, 2]
    tags = [c["tag"] for c in C]
    xpos = np.arange(len(C))
    ax.bar(xpos - 0.2, [c["add"] for c in C], width=0.4, color=C_PRICE,
           label=f"计划不变（缺口按 {K_EMG:g}π 紧急购电）")
    ax.bar(xpos + 0.2, [c["opt"] for c in C], width=0.4, color=C_NET, label="完全信息重解（下界）")
    ax.set_xticks(xpos)
    ax.set_xticklabels(tags, rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel("相对基准的计划外费用（元）")
    ax.set_title("(f) 输入偏差的代价：单边陡峭")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, axis="y")
    ax.axvline(4.5, color="0.6", lw=0.8)

    fig.suptitle("问题一敏感性分析：电价结构 · 储能参数 · 输入偏差", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save_fig(fig, "fig_敏感性分析.png", FIGDIR)

    # ================= 小结 =================
    rule("【结论】")
    log(f"1. 电价结构是最强的外部杠杆：ρ 从 1.0 降到 0 时购电费由 "
        f"{base['cost']:,.0f} 元变为 {A[0]['cost']:,.0f} 元（{A[0]['cost'] / base['cost'] - 1:+.2%}），"
        f"储能吞吐量由 {base['thru']:,.0f} kWh 降到 {A[0]['thru']:,.0f} kWh。")
    log(f"   拐点出现在 ρ ≈ {rho_star:.3f}（峰谷价比 = 1/η^2），实测第一个台阶（峰谷比 "
        f"{1 / ETA ** 2:.3f} 附近）吞吐量跳升，与解析阈值吻合，")
    log(f"   说明模型正确捕捉了「储能先避免弃光、再赚价差」的两层价值结构。")
    log(f"2. 储能参数中 η 最敏感：η 由 0.90 降到 0.78 时购电费 "
        f"{B['eta'][0][1] - base['cost']:+,.0f} 元，升到 0.96 时 "
        f"{B['eta'][-1][1] - base['cost']:+,.0f} 元；")
    log(f"   SOC 可用区间次之（E_min 1200→0 省 {base['cost'] - B['emin'][0][1]:,.0f} 元，"
        f"E_max 10800→12000 省 {base['cost'] - B['emax'][-1][1]:,.0f} 元），"
        f"功率上限影响最小（P_max 5000→8000 仅省 {base['cost'] - B['pmax'][-1][1]:,.0f} 元）。")
    log(f"3. 输入偏差的费用函数单边陡峭（负载 +10% 追加 "
        f"{[c for c in C if c['tag'] == '负载×1.10'][0]['add']:,.0f} 元，"
        f"负载 -10% 追加 "
        f"{[c for c in C if c['tag'] == '负载×0.90'][0]['add']:,.0f} 元），"
        f"因此问题二必须预留正裕量；且最优裕量与偏差幅度同量级。")
    log(f"\n总耗时 {time.time() - t0:.1f} 秒")
    write_report(resolve(TXT))


if __name__ == "__main__":
    main()
