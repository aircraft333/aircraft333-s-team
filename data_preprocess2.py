# -*- coding: utf-8 -*-
"""
2023C 数据预处理：找出应剔除的单品（无销量 / 销售天数过少 / 销量占比过低）

流程：
    第一步：找附件1 里有、但附件2 中没有任何销售记录的单品（无销量）
    第二步：销售天数 <= threshold_1(=10) 的单品（销售天数过少）并保存直方图
    第三步：累计销量占比 < threshold_2(=0.00003) 的单品（销量过低）并保存直方图
    第四步：取第二步与第三步的交集
    第五步：挑一个交集单品绘制并保存其销量随时间的变化折线图
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.cm import ScalarMappable

# ================= 1. 强制加载系统物理中文字体（防止乱码） =================
font_candidates = [
    "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf", # 黑体
    "C:/Windows/Fonts/simsun.ttc", # 宋体
    "/System/Library/Fonts/PingFang.ttc" # Mac
]

font_path = None
for p in font_candidates:
    if os.path.exists(p):
        font_path = p
        break

if font_path:
    fm.fontManager.addfont(font_path)
    zh_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = zh_font.get_name()
else:
    zh_font = fm.FontProperties()

plt.rcParams['axes.unicode_minus'] = False

# 输出图片的目录
output_dir = "preprocess_plots"
os.makedirs(output_dir, exist_ok=True)

# ================= 2. 数据读入与预处理 =================
csv_file1 = 'data/附件1.xlsx'
csv_file2 = 'data/附件2.xlsx'

# 兼容当前目录或上一级目录查找
if not os.path.exists(csv_file1):
    csv_file1 = '附件1.xlsx'
    csv_file2 = '附件2.xlsx'

print("正在读取附件1与附件2...")
df_1 = pd.read_excel(csv_file1)   # 单品信息
df = pd.read_excel(csv_file2)     # 销售流水

# 转换时间
df['销售日期'] = pd.to_datetime(df['销售日期'])
# 计算销售金额
df['销售金额'] = df['销量(千克)'] * df['销售单价(元/千克)']
# 映射品类
mapping_dict = df_1.set_index('单品编码')['分类名称'].to_dict()
df['品类'] = df['单品编码'].map(mapping_dict)

# ================= 第一步：处理没有销量的数据 =================
unique_values_df = df['单品编码'].unique()
unique_values_df_1 = df_1['单品编码'].unique()
values_only_in_df_1 = set(unique_values_df_1) - set(unique_values_df)
count_values_only_in_df_1 = len(values_only_in_df_1)

print("\n--- 第一步：无销量单品分析 ---")
print("df列'单品编码'的唯一值个数：", len(unique_values_df))
print("df_1列'单品编码'的唯一值个数：", len(unique_values_df_1))
print("df_1中有但是df中没有的值：", values_only_in_df_1)
print("这些值的个数：", count_values_only_in_df_1)

# ================= 第二步：销售天数少（阈值1）的单品 =================
threshold_1 = 10

result = df.groupby('单品编码')['销售日期'].nunique().reset_index()
result.rename(columns={'销售日期': '销售天数'}, inplace=True)

hist, bins = np.histogram(result['销售天数'], bins=10)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
cmap = plt.cm.coolwarm
norm = plt.Normalize(vmin=min(hist), vmax=max(hist))
colors = cmap(norm(hist))

plt.figure(figsize=(9, 6))
bars = plt.bar(bin_centers, hist, width=bins[1] - bins[0], color=colors, edgecolor='k', alpha=0.7)
for i, count in enumerate(hist):
    plt.text(bin_centers[i], count + 5, str(count), ha='center', va='bottom', fontproperties=zh_font)

plt.xlabel('销售天数', fontproperties=zh_font, fontsize=12)
plt.ylabel('单品数', fontproperties=zh_font, fontsize=12)
plt.title('单品销售天数分布直方图', fontproperties=zh_font, fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical')
cbar.set_label('计数', fontproperties=zh_font, rotation=90, labelpad=15)

plt.tight_layout()
save_path1 = os.path.join(output_dir, "1_销售天数分布直方图.png")
plt.savefig(save_path1, dpi=300)
print(f"-> 销售天数直方图已保存至: {save_path1}")
plt.close()

filtered_result = result[result['销售天数'] <= threshold_1]
list_1 = filtered_result['单品编码'].tolist()
print(f'\n--- 第二步：销售天数 <= {threshold_1} 的单品数量: {len(list_1)} ---')

# ================= 第三步：销量低（阈值 2）的单品 =================
threshold_2 = 0.00003

grouped = df.groupby('单品编码')['销量(千克)'].sum().reset_index()
total_sales = grouped['销量(千克)'].sum()
grouped['销量占比'] = grouped['销量(千克)'] / total_sales
low_percentage_groups = grouped[grouped['销量占比'] < threshold_2]['单品编码']
list_2 = low_percentage_groups.tolist()

print(f"\n--- 第三步：销量占比 < {threshold_2} 的单品数量: {len(list_2)} ---")
print("所有单品销量总和(kg):", total_sales)

data_g = [i for i in grouped['销量占比'] if i <= threshold_2]

plt.figure(figsize=(9, 6))
n, bins, patches = plt.hist(data_g, bins=10, edgecolor='k', color='skyblue', alpha=0.8)
plt.xlabel('销量占比', fontproperties=zh_font, fontsize=12)
plt.ylabel('频数', fontproperties=zh_font, fontsize=12)
plt.title('低销量单品销量占比分布直方图', fontproperties=zh_font, fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)

for rect in patches:
    height = rect.get_height()
    if height > 0:
        plt.annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', va='bottom', fontproperties=zh_font)

plt.tight_layout()
save_path2 = os.path.join(output_dir, "2_低销量单品占比直方图.png")
plt.savefig(save_path2, dpi=300)
print(f"-> 销量占比直方图已保存至: {save_path2}")
plt.close()

# ================= 第四步：取交集 =================
intersection = list(set(list_1) & set(list_2))
print(f"\n--- 第四步：双重筛选交集数量: {len(intersection)} ---")
print(intersection)

# ================= 第五步：查看交集单品的时间序列销量 =================
if len(intersection) >= 2:
    target_item = intersection[1]
elif len(intersection) == 1:
    target_item = intersection[0]
else:
    target_item = None

if target_item is not None:
    grouped_item = df.groupby(['单品编码', '销售日期'])['销量(千克)'].sum().reset_index()
    filtered_df = grouped_item[grouped_item['单品编码'] == target_item].sort_values(by='销售日期')
    
    # 查找单品名称（如果有）
    item_name_match = df_1[df_1['单品编码'] == target_item]['单品名称'].values
    item_name = item_name_match[0] if len(item_name_match) > 0 else "未知单品"

    plt.figure(figsize=(10, 5))
    plt.plot(filtered_df['销售日期'], filtered_df['销量(千克)'], marker='o', linestyle='-', color='#d9534f', linewidth=2)
    plt.title(f'剔除单品样例展示：单品编码 {target_item} ({item_name}) 历史销量走势', 
              fontproperties=zh_font, fontsize=13, fontweight='bold')
    plt.xlabel('销售日期', fontproperties=zh_font, fontsize=11)
    plt.ylabel('销量 (千克)', fontproperties=zh_font, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=30)
    
    plt.tight_layout()
    save_path3 = os.path.join(output_dir, f"3_交集单品_{target_item}_销量时序走势.png")
    plt.savefig(save_path3, dpi=300)
    print(f"-> 单品样例时序图已保存至: {save_path3}")
    plt.close()
else:
    print("交集为空，未绘制单品折线图。")

print(f"\n 全部完成！请查看本地生成的文件目录：`{output_dir}/`")