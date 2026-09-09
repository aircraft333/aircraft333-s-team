# -*- coding: utf-8 -*-
"""
脚本功能：
根据《附件2.xlsx》，按【月份（YYYY-MM）】合并汇总所有蔬菜的总销售量与销售额，
并生成统计 Excel 表格和高清月度趋势 SVG 矢量图。
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ================= 1. 环境与 SVG 矢量图配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
os.chdir(current_dir)

# 确保导出的 SVG 文字为矢量轮廓，Word 插入不乱码
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

# ================= 2. 自动读取附件2数据 =================
def find_file(filename):
    for p in [filename, os.path.join("data", filename), os.path.join("..", filename)]:
        if os.path.exists(p):
            return p
    return None

file_path = find_file("附件2.xlsx") or find_file("附件2.csv")
if not file_path:
    raise FileNotFoundError("❌ 未找到《附件2.xlsx》，请确保文件在当前目录或 data/ 目录下！")

print(f"正在读取数据: {file_path} ...")
df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

# ================= 3. 数据清洗与按月聚合 =================
print("正在提取月份并汇总月度总销量...")

# 转换日期并提取年月
df['销售日期'] = pd.to_datetime(df['销售日期'])
df['年月'] = df['销售日期'].dt.to_period('M').astype(str)  # 格式如 '2020-07'
df['年份'] = df['销售日期'].dt.year
df['月份'] = df['销售日期'].dt.month

# 过滤缺失数据
df = df.dropna(subset=['年月', '销量(千克)'])

# 计算销售额（如果存在单价）
has_price = '销售单价(元/千克)' in df.columns
if has_price:
    df['销售额'] = df['销量(千克)'] * df['销售单价(元/千克)']

# 核心聚合：按【年月】分组汇总
agg_dict = {
    '总销量_千克': ('销量(千克)', 'sum'),
    '营业天数': ('销售日期', 'nunique'),
    '订单笔数': ('销量(千克)', 'count')
}
if has_price:
    agg_dict['总销售额_元'] = ('销售额', 'sum')

monthly_sales = df.groupby(['年月', '年份', '月份']).agg(**agg_dict).reset_index()

# 计算衍生指标
monthly_sales['总销量(千克)'] = monthly_sales['总销量_千克'].round(2)
monthly_sales['总销量(吨)'] = (monthly_sales['总销量_千克'] / 1000.0).round(3)
monthly_sales['日均销量(千克)'] = (monthly_sales['总销量_千克'] / monthly_sales['营业天数']).round(2)

if has_price:
    monthly_sales['总销售额(元)'] = monthly_sales['总销售额_元'].round(2)
    monthly_sales['均价(元/千克)'] = (monthly_sales['总销售额(元)'] / monthly_sales['总销量(千克)']).round(2)

# 整理输出列
output_cols = ['年月', '年份', '月份', '总销量(千克)', '总销量(吨)', '日均销量(千克)', '营业天数', '订单笔数']
if has_price:
    output_cols.insert(6, '总销售额(元)')
    output_cols.insert(7, '均价(元/千克)')

result_df = monthly_sales[output_cols].sort_values(by='年月').reset_index(drop=True)

# 导出 Excel
output_excel = "每月所有蔬菜总销量合并表.xlsx"
result_df.to_excel(output_excel, index=False)
print(f"✅ 月度合并表已导出至: {output_excel}")

# ================= 4. 生成月度走势 SVG 矢量图 =================
fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

x_indices = range(len(result_df))
x_labels = result_df['年月'].tolist()
y_values = result_df['总销量(吨)'].values

# 绘制折线与数据散点
ax.plot(x_indices, y_values, marker='o', markersize=4, color='#1f77b4', linewidth=2.0, label='月度总销量 (吨)')

# 标注峰值与谷值
max_idx = y_values.argmax()
min_idx = y_values.argmin()
ax.scatter([max_idx, min_idx], [y_values[max_idx], y_values[min_idx]], color='#d62728', s=60, zorder=5)
ax.annotate(f"最高: {y_values[max_idx]:.2f} 吨\n({x_labels[max_idx]})",
            xy=(max_idx, y_values[max_idx]), xytext=(0, 8), textcoords='offset points',
            ha='center', fontsize=9, fontproperties=zh_font, fontweight='bold', color='#d62728')
ax.annotate(f"最低: {y_values[min_idx]:.2f} 吨\n({x_labels[min_idx]})",
            xy=(min_idx, y_values[min_idx]), xytext=(0, -20), textcoords='offset points',
            ha='center', fontsize=9, fontproperties=zh_font, fontweight='bold', color='#1a7832')

# 坐标轴美化
ax.set_title("商超蔬菜月度总销量走势 (2020.07 - 2023.06)", fontproperties=zh_font, fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("年月 (YYYY-MM)", fontproperties=zh_font, fontsize=11)
ax.set_ylabel("月度总销量 (吨)", fontproperties=zh_font, fontsize=11)

# X轴刻度每隔2个月显示一个，避免重叠
ax.set_xticks(x_indices[::2])
ax.set_xticklabels(x_labels[::2], rotation=45, ha='right', fontproperties=zh_font, fontsize=9)
ax.set_xlim(-0.5, len(x_labels) - 0.5)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(prop=zh_font, loc='upper right')

plt.tight_layout()

svg_path = "历年蔬菜月度销量走势图.svg"
plt.savefig(svg_path, format='svg', bbox_inches='tight')
plt.close()

print(f"✅ 月度走势 SVG 矢量图已生成: {svg_path}")

# ================= 5. 输出数据前瞻 =================
print("\n" + "="*60)
print("【月度销量合并表前 5 行预览】")
print(result_df.head().to_string(index=False))
print("\n【月度销量合并表后 5 行预览】")
print(result_df.tail().to_string(index=False))
print("="*60)