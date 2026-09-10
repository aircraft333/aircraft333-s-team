# -*- coding: utf-8 -*-
"""
2026 高教社杯 C 题 · 全局配置与公共函数
=====================================================================
所有常量（路径 / 物理参数 / 绘图风格）与工具函数集中在
本文件，各问脚本只需：

    from config import *

---------------------------------------------------------------------
时间口径（重要，务必与 result1.xlsx 模板保持一致）：
    附件1 的时间戳按「时段右端点」解释，第 k 行时间 T_k 对应区间
    [T_k - 10min, T_k]：
        第 1   行 00:10    -> [00:00, 00:10]
        最后 1 行 0:00+1   -> [23:50, 24:00]   （0:00+1 表示当天 24:00）
    这样 144 段恰好铺满 [00:00, 24:00]，与题目「储能 0:00 与 24:00
    储电量相同」对得上。
"""
import datetime as dt
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sps

# =====================================================================
# 一、路径配置（相对路径统一相对「项目根目录」= 本文件所在目录）
# =====================================================================
ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve(*parts):
    """把相对路径解析成相对项目根的绝对路径

    这样脚本无论从哪个工作目录运行、被挪到哪，都能找到附件与模板。
    """
    return os.path.join(ROOT, *parts)


ATT1 = "附件1.xlsx"              # 某天：电价、小区负载、光伏预测功率（10 min 粒度）
ATT2 = "附件2.xlsx"              # 全年：小区负载 + 光伏发电实际功率
ATT3 = "附件3.xlsx"              # 全年：光伏发电功率预报（0/6/12/18 时发布）
ATT4 = "附件4.xlsx"              # 全年：电价

DIR_RESULT = "result"            # 官方结果模板目录
TPL1 = os.path.join(DIR_RESULT, "result1.xlsx")       # 问题一模板
TPL2 = os.path.join(DIR_RESULT, "result2.xlsx")       # 问题二模板
TPL3 = os.path.join(DIR_RESULT, "result3.xlsx")       # 问题三模板
TPL4_2 = os.path.join(DIR_RESULT, "result4-2.xlsx")   # 问题四(对应问题2)模板
TPL4_3 = os.path.join(DIR_RESULT, "result4-3.xlsx")   # 问题四(对应问题3)模板

OUT1 = "result1.xlsx"            # 结果输出（写到项目根，不动 result/ 里的模板）
OUT2 = "result2.xlsx"
OUT3 = "result3.xlsx"
OUT4_2 = "result4-2.xlsx"
OUT4_3 = "result4-3.xlsx"

DIR_FIG = "figures"              # 图片根目录
FIG_Q1 = os.path.join(DIR_FIG, "q1")
TXT_Q1 = "q1_数据分析结果.txt"

# =====================================================================
# 二、系统参数（题目附录1）
# =====================================================================
N_SLOT = 144                     # 单天时段数（每段 10 min）
DT_H = 1 / 6                     # 单个时段时长（小时）
HORIZON_H = N_SLOT * DT_H        # 调度周期 = 24 h

ETA = 0.90                       # 储能充放电（单程）效率
ETA2 = ETA ** 2                  # 往返效率（充一次、放一次）
E_CAP = 12000.0                  # 储能设备最大容量 (kWh)
SOC_MIN = 1200.0                 # 储电量下限 (kWh) —— 避免过度放电
SOC_MAX = 10800.0                # 储电量上限 (kWh) —— 避免过度充电
SOC0 = 6000.0                    # 2025-01-01 0:00 的初始储电量 (kWh)
P_RATE = 5000.0                  # 最大充/放电功率 (kW)
SLOT_MAX = P_RATE * DT_H         # 单时段最大充/放电量 (kWh) = 833.33

# =====================================================================
# 三、绘图风格
# =====================================================================
FONT = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]   # 依次回退，避免个别字形缺失
DPI = 120
LW = 1.6                         # 常规线宽
C_PRICE = "#d62728"              # 电价：红
C_LOAD = "#1f77b4"               # 负载：蓝
C_PV = "#ff7f0e"                 # 光伏：橙
C_NET = "#2ca02c"                # 净负荷 / 谷时段：绿
A_TIER = 0.08                    # 峰谷背景色带透明度


def setup_plot():
    """统一绘图风格（中文字体、负号、分辨率）。脚本开头调用一次即可。"""
    plt.rcParams["font.family"] = FONT
    plt.rcParams["axes.unicode_minus"] = False       # 负号正常显示
    plt.rcParams["figure.dpi"] = DPI


def save_fig(fig, name, figdir=FIG_Q1):
    """保存图片到 figdir（自动建目录），返回完整路径"""
    os.makedirs(figdir, exist_ok=True)
    path = os.path.join(figdir, name)
    fig.savefig(path, bbox_inches="tight")
    log(f"    → 已保存 {path}")
    return path


# =====================================================================
# 四、报告输出（把控制台的统计事实同时落盘）
# =====================================================================
REPORT = []


def log(*args):
    """打印并记入报告缓冲区

    注意：Windows 下若 stdout 被重定向为管道/文件，Python 会按 ANSI
    代码页（cp936）编码，遇到 '²'、'η' 这类字符会抛 UnicodeEncodeError。
    这里做一次兜底，保证 `python x.py > out.txt` 场景下也不中断。
    """
    s = " ".join(str(a) for a in args)
    try:
        print(s)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(s.encode(enc, "replace").decode(enc, "replace"))
    REPORT.append(s)
    return s


def rule(title=None, width=78):
    """打印分隔线；给了 title 就打印居中的小节标题"""
    log("=" * width)
    if title:
        log(title)
        log("=" * width)


def write_report(path):
    """把报告缓冲区写到文本文件"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))
    print(f"\n分析结果已写入 {path}")


def stat_line(name, arr, unit=""):
    """打印一行基本统计量（min / max / mean / std / 极差比）"""
    arr = np.asarray(arr, dtype=float)
    log(f"{name:<10s} min={arr.min():10.4f}  max={arr.max():10.4f}  "
        f"mean={arr.mean():9.4f}  std={arr.std():8.4f}  "
        f"极差/均值={np.ptp(arr) / arr.mean():6.2f}   ({unit})".rstrip())


# =====================================================================
# 五、时间处理
# =====================================================================
def parse_time_to_min(x):
    """把 '00:10:00' / '23:40' / '0:00+1' / datetime.time 统一成
    「距当天 0:00 的分钟数」(0 ~ 1440)

    附件1 的『时间』列是混合类型：部分单元格被 Excel 存成时间格式
    （读出来是 datetime.time），其余是文本，最后一行还是带 '+1' 的
    '0:00+1'，所以不能直接用 pd.to_timedelta。
    """
    if isinstance(x, dt.time):                       # Excel 时间格式
        return x.hour * 60 + x.minute
    s = str(x).strip()
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(?:\+(\d+))?$", s)
    if m is None:
        raise ValueError(f"无法解析时间: {s!r}")
    mins = int(m.group(1)) * 60 + int(m.group(2))
    if m.group(4):                                   # '0:00+1' -> 当天 24:00
        mins += 24 * 60 * int(m.group(4))
    return mins


def to_min(x):
    """parse_time_to_min 的简写别名"""
    return parse_time_to_min(x)


def fmt(minutes):
    """分钟数 -> 'HH:MM'（1440 显示为 24:00）"""
    h, m = divmod(int(round(minutes)), 60)
    return f"{h:02d}:{m:02d}"


# =====================================================================
# 六、附件读取
# =====================================================================
def load_att1(path=ATT1):
    """读取附件1（某天，10 min 粒度），解析时间列后返回 DataFrame

    新增列：
        分钟   : 时段右端点（距 0:00 的分钟数，0 ~ 1440）
        小时   : 时段右端点（小时）
        时段起 : 时段左端点（小时）

    原始列：时间 / 电价(元/kWh) / 小区负载(kW) / 光伏发电预测功率(kW)
    """
    df = pd.read_excel(path, sheet_name="Sheet1")
    df["分钟"] = df["时间"].map(parse_time_to_min)
    df["小时"] = df["分钟"] / 60.0
    df["时段起"] = (df["分钟"] - 10) / 60.0
    assert df["分钟"].is_monotonic_increasing, "时间列不是单调递增，请检查原始数据"
    assert len(df) == N_SLOT and df["分钟"].iloc[-1] == 24 * 60, \
        f"附件1 应为 {N_SLOT} 段、覆盖 [0:00, 24:00]"
    return df


def att1_arrays(df):
    """从 load_att1 的结果里取出常用数组，返回字典

    键：minutes / hours / price / load / pv / net
        （net = 负载 - 光伏，即净负荷，>0 表示需外网补充）
    """
    load = df["小区负载"].to_numpy(float)
    pv = df["光伏发电预测功率"].to_numpy(float)
    return {
        "minutes": df["分钟"].to_numpy(float),
        "hours": df["小时"].to_numpy(float),
        "price": df["电价"].to_numpy(float),
        "load": load,
        "pv": pv,
        "net": load - pv,
    }


# =====================================================================
# 七、峰谷平时段划分
# =====================================================================
TIER_V, TIER_F, TIER_P = "谷", "平", "峰"


def classify_tiers(price, q_lo=0.33, q_hi=0.67):
    """按电价分位数把各时段划成 谷 / 平 / 峰

    返回 (tier 数组, 谷价阈值, 峰价阈值)
    """
    price = np.asarray(price, dtype=float)
    lo, hi = np.quantile(price, [q_lo, q_hi])
    tier = np.where(price <= lo, TIER_V, np.where(price >= hi, TIER_P, TIER_F))
    return tier, float(lo), float(hi)


def label_segments(minutes, labels):
    """把逐时段标签合并成连续区间

    返回 [(起始分钟, 结束分钟, 标签), ...]，
    其中区间按 [右端点 - 10, 右端点] 解释，相邻同标签段首尾相接。
    """
    minutes = np.asarray(minutes, dtype=float)
    segs, start, cur = [], minutes[0], labels[0]
    for m, lab in zip(minutes[1:], labels[1:]):
        if lab != cur:
            segs.append((start, m, cur))
            start, cur = m, lab
    segs.append((start, minutes[-1], cur))
    return [(s - 10, e, lab) for s, e, lab in segs]


def shade_tiers(ax, segs, alpha=A_TIER):
    """在坐标轴上用背景色带标出峰 / 谷时段（segs 来自 label_segments）"""
    for s, e, lab in segs:
        if lab == TIER_P:
            ax.axvspan(s, e, color=C_PRICE, alpha=alpha, lw=0)
        elif lab == TIER_V:
            ax.axvspan(s, e, color=C_NET, alpha=alpha, lw=0)


def tier_legend(ax, alpha=0.2):
    """给坐标轴补上图例用的峰 / 谷色块（不画数据）"""
    ax.plot([], [], color=C_PRICE, alpha=alpha, lw=8, label="峰时段")
    ax.plot([], [], color=C_NET, alpha=alpha, lw=8, label="谷时段")


# =====================================================================
# 八、导出清单（from config import * 只会带出这里列出的名字）
# =====================================================================
__all__ = [
    # 路径
    "ROOT", "resolve",
    "ATT1", "ATT2", "ATT3", "ATT4", "DIR_RESULT", "DIR_FIG", "FIG_Q1", "TXT_Q1",
    "TPL1", "TPL2", "TPL3", "TPL4_2", "TPL4_3",
    "OUT1", "OUT2", "OUT3", "OUT4_2", "OUT4_3",
    # 系统参数
    "N_SLOT", "DT_H", "HORIZON_H", "ETA", "ETA2", "E_CAP",
    "SOC_MIN", "SOC_MAX", "SOC0", "P_RATE", "SLOT_MAX",
    # 绘图
    "FONT", "DPI", "LW", "C_PRICE", "C_LOAD", "C_PV", "C_NET", "A_TIER",
    "setup_plot", "save_fig",
    # 报告
    "REPORT", "log", "rule", "write_report", "stat_line",
    # 时间
    "parse_time_to_min", "to_min", "fmt",
    # 数据
    "load_att1", "att1_arrays",
    # 峰谷平
    "TIER_V", "TIER_F", "TIER_P", "classify_tiers",
    "label_segments", "shade_tiers", "tier_legend",
    # 依赖（供脚本直接使用，免去重复 import）
    # 注意：不导出 dt / re / sys，避免与脚本里的局部变量名（如 dt = DT_H）冲突
    "np", "pd", "plt", "sps", "os",
]
