# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
脚本功能：
1. 从《附件2.xlsx》汇总同一小时时段内所有蔬菜的销量，导出《每日各小时蔬菜总销量.xlsx》。
2. 绘制全天 24 小时平均销量分布图，并仅保存为【无损 SVG 矢量图】。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ================= 1. 环境与 SVG 矢量参数设置 =================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
os.chdir(current_dir)

# 确保 SVG 字体转为矢量路径/标准文本，Word 导入不乱码
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

# ================= 2. 读取附件2数据 =================
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

# ================= 3. 提取小时并汇总销量 =================
print("正在按小时聚合所有蔬菜销量...")
df['销售日期'] = pd.to_datetime(df['销售日期']).dt.date

def extract_hour(time_val):
    if pd.isna(time_val):
        return np.nan
    try:
        return int(str(time_val).strip().split(':')[0])
    except:
        return np.nan

df['小时'] = df['扫码销售时间'].apply(extract_hour)
df = df.dropna(subset=['小时', '销量(千克)'])
df['小时'] = df['小时'].astype(int)
df['小时时段'] = df['小时'].apply(lambda h: f"{h:02d}:00-{h:02d}:59")

# 核心聚合：按日期与小时段将所有蔬菜求和
hourly_sales = df.groupby(['销售日期', '小时时段', '小时'])['销量(千克)'].sum().reset_index()
hourly_sales.sort_values(by=['销售日期', '小时'], inplace=True)
hourly_sales['时段总销量(千克)'] = hourly_sales['销量(千克)'].round(3)

# 导出 Excel
output_xlsx = "每日各小时蔬菜总销量.xlsx"
hourly_sales[['销售日期', '小时时段', '时段总销量(千克)']].to_excel(output_xlsx, index=False)
print(f"✅ Excel 汇总已保存: {output_xlsx}")

# ================= 4. 生成 24 小时走势图并保存为 SVG 矢量图 =================
hour_avg = df.groupby('小时')['销量(千克)'].sum() / df['销售日期'].nunique()
all_hours = pd.Series(0.0, index=range(24))
all_hours.update(hour_avg)

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(all_hours.index, all_hours.values, marker='o', markersize=5, color='#1f77b4', linewidth=2, label='各时段平均总销量')
ax.fill_between(all_hours.index, all_hours.values, color='#1f77b4', alpha=0.15)

# 标注早市最高峰
max_hour = all_hours.idxmax()
max_val = all_hours[max_hour]
ax.annotate(f'早市峰值: {max_val:.2f} kg',
            xy=(max_hour, max_val),
            xytext=(max_hour + 0.8, max_val),
            arrowprops=dict(facecolor='#d9534f', shrink=0.08, width=1, headwidth=6),
            fontproperties=zh_font, fontsize=10, color='#d9534f', fontweight='bold')

ax.set_title("商超各时段蔬菜平均销量分布（早晚高峰时序特征）", fontproperties=zh_font, fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("营业时段 (小时)", fontproperties=zh_font, fontsize=11)
ax.set_ylabel("单日平均销量 (千克)", fontproperties=zh_font, fontsize=11)
ax.set_xticks(range(0, 24))
ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], fontproperties=zh_font, fontsize=8.5, rotation=35)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(prop=zh_font, loc='upper right')

plt.tight_layout()

# 核心保存为 SVG 矢量图
svg_path = "全天24小时分时销量走势图.svg"
plt.savefig(svg_path, format='svg', bbox_inches='tight')
plt.close()

print("\n" + "="*50)
print(f"✅ SVG 矢量图已生成: {os.path.abspath(svg_path)}")
print("💡 提示：在 Word 中直接点击 [插入] -> [图片] 选择此 .svg 文件即可，放大任意倍数均保持极致清晰。")
print("="*50)