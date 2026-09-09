# -*- coding: utf-8 -*-
"""
脚本功能：
1. 读取《全天24小时分时平均销量分布.xlsx》（或直接从附件2提取小时序列）
2. 计算 0 ~ 12 阶滞后（Lag）的自相关函数（ACF）及其 95% 置信区间
3. 生成无损 SVG 矢量图及《全天分时销量ACF值.xlsx》
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ================= 1. 环境与 SVG 矢量图配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
os.chdir(current_dir)

# 确保 SVG 字体转为矢量路径，Word/PPT 导入不乱码
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

# ================= 2. 获取 24 小时分时平均销量数据 =================
data_file = "全天24小时分时平均销量分布.xlsx"

if os.path.exists(data_file):
    print(f"正在读取文件: {data_file} ...")
    df_dist = pd.read_excel(data_file)
    # 识别销量列
    val_col = [c for c in df_dist.columns if '销量' in str(c) or 'val' in str(c).lower()][0]
    hourly_series = df_dist[val_col].values
else:
    # 容错：如果尚未生成该文件，直接读取附件2计算
    print("未找到《全天24小时分时平均销量分布.xlsx》，尝试从《附件2.xlsx》直接计算...")
    f2 = "附件2.xlsx" if os.path.exists("附件2.xlsx") else "附件2.csv"
    df2 = pd.read_excel(f2) if f2.endswith('.xlsx') else pd.read_csv(f2)
    df2['小时'] = df2['扫码销售时间'].apply(lambda x: int(str(x).strip().split(':')[0]) if pd.notna(x) else np.nan)
    df2 = df2.dropna(subset=['小时', '销量(千克)'])
    df2['小时'] = df2['小时'].astype(int)
    
    unique_days = df2['销售日期'].nunique()
    h_sum = df2.groupby('小时')['销量(千克)'].sum()
    all_h = pd.Series(0.0, index=range(24))
    all_h.update(h_sum / unique_days)
    hourly_series = all_h.values

# 确保序列长度为 24
if len(hourly_series) != 24:
    hourly_series = np.resize(hourly_series, 24)

# ================= 3. 计算 ACF (自相关函数) =================
# 计算最大滞后阶数（取总长度的一半左右，即 lag=12）
nlags = 12
N = len(hourly_series)

try:
    from statsmodels.tsa.stattools import acf
    acf_values, confint = acf(hourly_series, nlags=nlags, alpha=0.05)
    lower_bound = confint[:, 0] - acf_values
    upper_bound = confint[:, 1] - acf_values
except ImportError:
    # 自定义标准 ACF 计算公式与 Bartlett 置信区间
    mean_val = np.mean(hourly_series)
    c0 = np.sum((hourly_series - mean_val) ** 2) / N
    acf_values = [1.0]
    for k in range(1, nlags + 1):
        ck = np.sum((hourly_series[:-k] - mean_val) * (hourly_series[k:] - mean_val)) / N
        acf_values.append(ck / c0 if c0 != 0 else 0)
    acf_values = np.array(acf_values)
    # 95% 置信区间 (Bartlett 公式: ±1.96 / sqrt(N))
    z_val = 1.96 / np.sqrt(N)
    upper_bound = np.full(nlags + 1, z_val)
    lower_bound = np.full(nlags + 1, -z_val)

lags = np.arange(nlags + 1)

# 保存 ACF 数据表格
df_acf = pd.DataFrame({
    '滞后阶数(Lag/小时)': lags,
    'ACF自相关系数值': np.round(acf_values, 4),
    '95%置信上限': np.round(upper_bound, 4),
    '95%置信下限': np.round(lower_bound, 4)
})
df_acf.to_excel("全天分时销量ACF值.xlsx", index=False)
print("✅ ACF 数值结果已导出至: 全天分时销量ACF值.xlsx")

# ================= 4. 绘制并保存学术级 SVG 矢量图 =================
fig, ax = plt.subplots(figsize=(9, 5))

# 绘制 95% 置信带 (阴影区域)
ax.axhline(0, color='black', linewidth=1, linestyle='-')
ax.fill_between(lags, upper_bound, lower_bound, color='#1f77b4', alpha=0.15, label='95% 置信区间')
ax.plot(lags, upper_bound, color='#1f77b4', linestyle='--', linewidth=1, alpha=0.7)
ax.plot(lags, lower_bound, color='#1f77b4', linestyle='--', linewidth=1, alpha=0.7)

# 绘制针状图 (Stem plot)
markerline, stemlines, baseline = ax.stem(lags, acf_values, linefmt='b-', markerfmt='bo', basefmt=' ')
plt.setp(stemlines, 'color', '#1f77b4', 'linewidth', 1.8)
plt.setp(markerline, 'color', '#1f77b4', 'markersize', 6)

# 图表装饰与标注
ax.set_title("全天24小时蔬菜平均总销量自相关函数 (ACF) 图像", fontproperties=zh_font, fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("滞后阶数 Lag (小时)", fontproperties=zh_font, fontsize=11)
ax.set_ylabel("自相关系数 (ACF)", fontproperties=zh_font, fontsize=11)
ax.set_xticks(lags)
ax.set_ylim([-0.8, 1.15])
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(prop=zh_font, loc='upper right')

# 在关键点添加数值标注 (以 Lag 1 和 Lag 2 为例)
for i in range(1, 4):
    ax.annotate(f"{acf_values[i]:.2f}",
                xy=(i, acf_values[i]),
                xytext=(i, acf_values[i] + (0.08 if acf_values[i] >= 0 else -0.12)),
                ha='center', fontsize=9, fontproperties=zh_font, color='#333333')

plt.tight_layout()

# 保存为 SVG 矢量图
svg_out_path = "全天24小时分时销量ACF图.svg"
plt.savefig(svg_out_path, format='svg', bbox_inches='tight')
plt.close()

print("="*50)
print(f"✅ SVG 矢量图生成成功: {os.path.abspath(svg_out_path)}")
print("="*50)
print("\n【ACF 计算结果速览】")
print(df_acf.to_string(index=False))