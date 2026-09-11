# -*- coding: utf-8 -*-
"""
电价对比折线图：附件1 基准电价 vs 附件4 随机抽取两日的波动电价
=================================================================
用法：
    python price_compare.py             # 每次随机抽两天（打印所用种子）
    python price_compare.py 42          # 指定随机种子 42，保证可复现
    python price_compare.py 7 "2025-04-26,2025-01-06"    # 直接指定两天日期

输出：
    电价对比_附件1_vs_随机两日.pdf (矢量图格式)
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================ 配置 ============================
N_SLOT = 144  # 每天 144 个 10 分钟时段
ATT1 = "附件1.xlsx"
ATT4 = "附件4.xlsx"
OUT_PDF = "电价对比_附件1_vs_随机两日.pdf"  # 保存为 PDF 矢量格式

# ============================ 数据读取 ============================
def load_att1_price():
    """附件1：电价在第 2 列（0:10 ~ 0:00+1 共 144 个时段）"""
    df = pd.read_excel(ATT1)
    price = df.iloc[:, 1].to_numpy(float)
    assert price.shape[0] == N_SLOT, f"附件1 电价长度异常：{price.shape}"
    return price

def load_att4():
    """附件4：返回 dates (365,) 与电价矩阵 (365, 144)"""
    df = pd.read_excel(ATT4)
    dates = pd.to_datetime(df.iloc[:, 0])
    P = df.iloc[:, 1 : 1 + N_SLOT].to_numpy(float)
    assert P.shape[1] == N_SLOT, f"附件4 时段数异常：{P.shape}"
    return dates, P

# ============================ 抽两天 ============================
def pick_two_days(dates, P):
    """解析命令行：种子 / 指定日期；否则真随机并打印种子"""
    args = sys.argv[1:]
    if len(args) >= 2:  # 手动指定日期
        wanted = [d.strip() for d in args[1].split(",")]
        idx = []
        for w in wanted:
            hit = np.where(dates == pd.Timestamp(w))[0]
            if len(hit) == 0:
                raise ValueError(f"附件4 中找不到日期 {w}")
            idx.append(int(hit[0]))
        return idx, f"手动指定 {wanted}"
    if len(args) == 1:  # 指定随机种子
        seed = int(args[0])
    else:  # 真随机
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(dates), size=2, replace=False).tolist())
    return idx, f"随机种子 = {seed}"

# ============================ 主程序 ============================
def main():
    p1 = load_att1_price()
    dates, P = load_att4()
    idx, how = pick_two_days(dates, P)

    print(f"抽样方式：{how}")
    print(
        f"{'日期':<12s} {'均价(元/kWh)':>12s} {'最低':>10s} {'最高':>10s} {'峰谷比':>8s}"
    )

    # 时段右端点 → 小时坐标（0:10 对应 1/6 小时，24:00 对应 24）
    hrs = (np.arange(N_SLOT) + 1) / 6.0

    # 设置支持中文的字体与数学符号负号
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    # 保证导出 PDF 时文本以 TrueType 字体嵌入
    plt.rcParams["pdf.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(14, 6.8))

    # ---- 附件1 基准电价（深蓝加粗，最醒目） ----
    ax.plot(
        hrs,
        p1,
        color="#0D3B66",
        lw=2.6,
        alpha=1.0,
        zorder=5,
        label=f"附件1 基准电价（均价 {p1.mean():.4f} 元/kWh，峰谷比 {p1.max() / p1.min():.2f}）",
    )

    # ---- 随机两日波动电价 ----
    palette = ["#C62828", "#1B5E20"]  # 深红 / 墨绿
    styles = [("--", 2.0), ("-.", 2.0)]
    for k, i in enumerate(idx):
        pi = P[i]
        d = dates.iloc[i]
        ax.plot(
            hrs,
            pi,
            color=palette[k],
            lw=styles[k][1],
            linestyle=styles[k][0],
            alpha=0.95,
            label=f"{d.strftime('%Y-%m-%d')} 波动电价（均价 {pi.mean():.4f} 元/kWh，峰谷比 {pi.max() / max(pi.min(), 1e-6):.2f}）",
        )
        print(
            f"{d.strftime('%Y-%m-%d'):<12s} {pi.mean():>12.4f} {pi.min():>10.4f}"
            f" {pi.max():>10.4f} {pi.max() / max(pi.min(), 1e-6):>8.2f}"
        )

    print(
        f"{'附件1(基准)':<12s} {p1.mean():>12.4f} {p1.min():>10.4f}"
        f" {p1.max():>10.4f} {p1.max() / p1.min():>8.2f}"
    )

    # ---- 坐标轴与装饰 ----
    ax.set_xlabel("时刻（每日 144 个 10 分钟时段）", fontsize=11)
    ax.set_ylabel("电价（元/kWh）", fontsize=11)
    ax.set_title(
        "附件1 基准电价 与 附件4 随机两日波动电价 对比",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 1))
    ax.set_xticklabels([f"{h}:00" for h in np.arange(0, 25)], fontsize=9)

    # ---- 留出顶部安全区，确保内部图例不遮挡曲线波峰 ----
    y_min = min(p1.min(), *[P[i].min() for i in idx])
    y_max = max(p1.max(), *[P[i].max() for i in idx])
    # 顶部多留 35% 空间，刚好容纳图例
    ax.set_ylim(max(0.0, y_min - 0.06), y_max * 1.35)
    ax.grid(True, linestyle="--", alpha=0.45)

    # ---- 图例放置在图表内部正上方 (upper center) ----
    ax.legend(
        loc="upper center",
        ncol=1,  # 纵向排列或单列清晰明了
        frameon=True,
        framealpha=0.92,
        edgecolor="#D0D0D0",
        facecolor="#FFFFFF",
        fontsize=10.5,
    )

    plt.tight_layout()
    # 保存为矢量 PDF 格式
    plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    print(f"\n>>> 矢量 PDF 图已成功保存至：{OUT_PDF}")
    plt.show()

if __name__ == "__main__":
    main()