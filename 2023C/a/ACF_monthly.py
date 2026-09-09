# -*- coding: utf-8 -*-
"""
脚本功能：
1. 读取《每月所有蔬菜总销量合并表.xlsx》
2. 计算月度销量序列的全滞后阶数 ACF (Lag 0 到 N-1，涵盖全部 36 个月)
3. 导出包含置信带、下边缘周期标注的学术级 SVG 矢量图及完整 Excel 明细
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ================= 1. 环境与 SVG 配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
os.chdir(current_dir)

# 确保导出的 SVG 文字为矢量轮廓
plt.rcParams['svg.fonttype'] = 'none'

# 配置中文字体
font_candidates = [
    "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf", # 黑体
    "C:/Windows/Fonts/simsun.ttc", # 宋体
    "/System/Library/Fonts/PingFang.ttc"
]
font_path = next((p for p in font_candidates if os.path.exists(p)), None)
if font_path:
    fm.fontManager.addfont(font_path)
    zh_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = zh_font.get_name()
else:
    zh_font = fm.FontProperties()
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 读取月度合并表 =================
input_file = "每月所有蔬菜总销量合并表.xlsx"
if not os.path.exists(input_file):
    alt_files = [f for f in os.listdir('.') if '月' in f and '销量' in f and f.endswith('.xlsx')]
    if alt_files:
        input_file = alt_files[0]
    else:
        raise FileNotFoundError(f"❌ 未找到《{input_file}》，请先运行按月合并脚本！")

df = pd.read_excel(input_file)

if '日均销量(千克)' in df.columns:
    series = df['日均销量(千克)'].values
    series_title = "月度日均销量"
elif '总销量(千克)' in df.columns:
    series = df['总销量(千克)'].values
    series_title = "月度总销量"
else:
    val_col = [c for c in df.columns if '销量' in str(c)][0]
    series = df[val_col].values
    series_title = val_col

N = len(series)
print(f"数据总月数 N = {N}，将计算全部 Lag 0 到 Lag {N-1} 的自相关系数...")

# ================= 3. 计算全部滞后阶数的 ACF =================
nlags = N - 1  # 包含所有月份的滞后阶数 (0 ~ 35)

def compute_acf(x, nlags):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = np.mean(x)
    var = np.var(x)
    if var == 0:
        return np.ones(nlags + 1), np.zeros(nlags + 1)
    
    acf_vals = [1.0]
    for k in range(1, nlags + 1):
        c_k = np.sum((x[:-k] - mean) * (x[k:] - mean)) / n
        acf_vals.append(c_k / var)
    
    acf_vals = np.array(acf_vals)
    ci = 1.96 / np.sqrt(n)
    return acf_vals, ci

acf_values, ci_val = compute_acf(series, nlags)
lags = np.arange(nlags + 1)

# 导出完整的全月份 ACF 表格
df_acf_all = pd.DataFrame({
    '滞后阶数(Lag/月)': lags,
    'ACF自相关系数': np.round(acf_values, 4),
    '95%置信上限': np.round(ci_val, 4),
    '95%置信下限': np.round(-ci_val, 4),
    '显著性判断': ['基准点' if k == 0 else ('显著相关' if abs(v) > ci_val else '不显著') for k, v in zip(lags, acf_values)]
})
df_acf_all.to_excel("全月份销量ACF计算结果.xlsx", index=False)
print("✅ 完整全月份 ACF 数值已保存至: 全月份销量ACF计算结果.xlsx")

# ================= 4. 绘制全月份学术级 SVG 矢量图 =================
# 适当加宽画布尺寸（14 x 5.2），确保 36 个柱子不拥挤
fig, ax = plt.subplots(figsize=(14, 5.2), dpi=300)

# 1. 绘制 95% 置信带
ax.axhline(0, color='#333333', linewidth=1.0, linestyle='-')
ax.axhline(ci_val, color='#1f77b4', linestyle='--', linewidth=1.2, alpha=0.85, label=f'95% 置信区间 (±{ci_val:.3f})')
ax.axhline(-ci_val, color='#1f77b4', linestyle='--', linewidth=1.2, alpha=0.85)
ax.fill_between(lags, ci_val, -ci_val, color='#1f77b4', alpha=0.12)

# 2. 区分颜色绘制 Stem 图（突出 Lag 12 和 Lag 24 两个年度周期点）
colors = []
for k in lags:
    if k == 0:
        colors.append('#7f7f7f')
    elif k in [12, 24]:
        colors.append('#d62728')  # 红色突出年度周期 (Lag 12, Lag 24)
    elif abs(acf_values[k]) > ci_val:
        colors.append('#2ca02c')  # 绿色突出显著阶数
    else:
        colors.append('#1f77b4')  # 普通蓝色

for k, val, col in zip(lags, acf_values, colors):
    ax.plot([k, k], [0, val], color=col, linewidth=2.0)
    ax.scatter(k, val, color=col, s=40, zorder=3, edgecolors='white', linewidth=0.8)

# 3. 标注周期点：统一放置在靠近下边缘区域 (y = -0.58)
target_bottom_y = -0.58

# (1) 标注 Lag 12 (第1个年周期)
if nlags >= 12:
    val_12 = acf_values[12]
    ax.annotate(f'第1年周期点\nLag 12 = {val_12:.3f}',
                xy=(12, val_12),
                xytext=(12, target_bottom_y),
                ha='center', va='top',
                fontsize=9.0, fontproperties=zh_font, fontweight='bold', color='#d62728',
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2, shrinkA=3, shrinkB=4))

# (2) 标注 Lag 24 (第2个年周期)
if nlags >= 24:
    val_24 = acf_values[24]
    ax.annotate(f'第2年周期点\nLag 24 = {val_24:.3f}',
                xy=(24, val_24),
                xytext=(24, target_bottom_y),
                ha='center', va='top',
                fontsize=9.0, fontproperties=zh_font, fontweight='bold', color='#d62728',
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2, shrinkA=3, shrinkB=4))

# 4. 图表坐标与细节美化
ax.set_title(f"商超蔬菜{series_title}全月份自相关函数 (ACF) 分析 (Lag 0 - {nlags})",
             fontproperties=zh_font, fontsize=13, fontweight='bold', pad=14)
ax.set_xlabel("滞后阶数 Lag (月)", fontproperties=zh_font, fontsize=11)
ax.set_ylabel("自相关系数 (ACF)", fontproperties=zh_font, fontsize=11)

# X轴刻度完整显示全部滞后阶数
ax.set_xticks(lags)
ax.set_xticklabels(lags, fontsize=8)
ax.set_xlim(-0.8, nlags + 0.8)
ax.set_ylim([-1.05, 1.15])
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(prop=zh_font, loc='upper right')

plt.tight_layout()

# 导出全月份 SVG
output_svg = "月度蔬菜总销量ACF图.svg"
plt.savefig(output_svg, format='svg', bbox_inches='tight')
plt.close()

print(f"✅ 已生成包含所有 {nlags+1} 个月份滞后阶数的 SVG 矢量图: {os.path.abspath(output_svg)}")