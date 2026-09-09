# -*- coding: utf-8 -*-
"""
脚本功能：
1. 读取刚生成的《历年所有蔬菜总销量合并表.xlsx》
2. 计算年度销售序列的自相关系数 (ACF) 及其 95% 置信带
3. 导出包含置信区间的学术级 SVG 矢量图及 ACF 数据明细 Excel
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ================= 1. 环境与 SVG 矢量图配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
os.chdir(current_dir)

# 确保导出的 SVG 文字为矢量曲线，导入 Word/PPT 绝不乱码
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

# ================= 2. 读取刚生成的年度合并表 =================
input_file = "历年所有蔬菜总销量合并表.xlsx"
if not os.path.exists(input_file):
    # 容错：若未在当前目录，寻找同类表
    alt_files = [f for f in os.listdir('.') if '年' in f and '销量' in f and f.endswith('.xlsx')]
    if alt_files:
        input_file = alt_files[0]
    else:
        raise FileNotFoundError(f"❌ 未找到《{input_file}》，请先运行上一步的合并脚本！")

print(f"正在读取文件: {input_file} ...")
df = pd.read_excel(input_file)

# 优先选用标准化后的【日均销售量(千克)】以消除半年度样本偏差；同时兼容总销量
if '日均销售量(千克)' in df.columns:
    series = df['日均销售量(千克)'].values
    series_name = "日均年化销量"
elif '总销售量(吨)' in df.columns:
    series = df['总销售量(吨)'].values
    series_name = "年度总销量"
else:
    val_col = [c for c in df.columns if '销量' in str(c)][0]
    series = df[val_col].values
    series_name = val_col

N = len(series)
print(f"参与 ACF 分析的数据长度 N = {N} (年份: {list(df['年份'])})")

# ================= 3. 计算 ACF (自相关函数) =================
# 对于 N=4，最大合理滞后阶数设为 N-2 = 2
nlags = min(2, N - 1)

def compute_acf(x, nlags):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = np.mean(x)
    var = np.var(x)
    if var == 0:
        return np.ones(nlags + 1), np.zeros(nlags + 1), np.zeros(nlags + 1)
    
    acf_vals = [1.0]
    for k in range(1, nlags + 1):
        c_k = np.sum((x[:-k] - mean) * (x[k:] - mean)) / n
        acf_vals.append(c_k / var)
    
    acf_vals = np.array(acf_vals)
    # Bartlett 95% 置信区间: ±1.96 / sqrt(N)
    ci = 1.96 / np.sqrt(n)
    return acf_vals, ci

acf_values, ci_val = compute_acf(series, nlags)
lags = np.arange(nlags + 1)

# 保存 ACF 数值表格
df_acf = pd.DataFrame({
    '滞后阶数(Lag/年)': lags,
    'ACF自相关系数': np.round(acf_values, 4),
    '95%置信上限': np.round(ci_val, 4),
    '95%置信下限': np.round(-ci_val, 4)
})
acf_out_xlsx = "年度销量ACF值.xlsx"
df_acf.to_excel(acf_out_xlsx, index=False)
print(f"✅ ACF 数值明细已导出至: {acf_out_xlsx}")

# ================= 4. 绘制并导出学术级 SVG 矢量图 =================
fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)

# 1. 绘制 95% 置信带（浅蓝半透明阴影）
ax.axhline(0, color='gray', linewidth=1.0, linestyle='-')
ax.axhline(ci_val, color='#1f77b4', linestyle='--', linewidth=1.2, alpha=0.8, label='95% 置信区间')
ax.axhline(-ci_val, color='#1f77b4', linestyle='--', linewidth=1.2, alpha=0.8)
ax.fill_between(lags, ci_val, -ci_val, color='#1f77b4', alpha=0.12)

# 2. 绘制针状图 (Stem Plot)
markerline, stemlines, baseline = ax.stem(lags, acf_values, linefmt='#d95f02', markerfmt='o', basefmt=' ')
plt.setp(stemlines, 'color', '#d95f02', 'linewidth', 2.2)
plt.setp(markerline, 'color', '#d95f02', 'markersize', 8, 'markeredgewidth', 1.5, 'markeredgecolor', 'white')

# 3. 添加数值标注
for i in range(len(lags)):
    offset = 0.08 if acf_values[i] >= 0 else -0.12
    ax.annotate(f"{acf_values[i]:.3f}",
                xy=(lags[i], acf_values[i]),
                xytext=(lags[i], acf_values[i] + offset),
                ha='center', fontsize=10, fontproperties=zh_font, fontweight='bold', color='#222222')

# 4. 图表细节美化
ax.set_title(f"商超蔬菜{series_name}年度自相关函数 (ACF) 分析", fontproperties=zh_font, fontsize=12, fontweight='bold', pad=14)
ax.set_xlabel("滞后阶数 Lag (年)", fontproperties=zh_font, fontsize=11)
ax.set_ylabel("自相关系数 (ACF)", fontproperties=zh_font, fontsize=11)
ax.set_xticks(lags)
ax.set_xticklabels([f"Lag {k}" for k in lags], fontproperties=zh_font)
ax.set_ylim([-1.15, 1.25])
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(prop=zh_font, loc='upper right')

plt.tight_layout()

# 导出 SVG
svg_path = "历年蔬菜总销量ACF图.svg"
plt.savefig(svg_path, format='svg', bbox_inches='tight')
plt.close()

print(f"✅ SVG 矢量图生成成功: {os.path.abspath(svg_path)}")
print("\n" + "=" * 50)
print("【年度销量 ACF 分析结果速览】")
print(df_acf.to_string(index=False))
print("=" * 50)