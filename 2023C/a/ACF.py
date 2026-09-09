import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# 1. 寻找文件
FILE_PATH = "各品类每日总销量.xlsx"

if not os.path.exists(FILE_PATH):
    print(f"找不到文件: {FILE_PATH}")
    exit()

print(f"成功找到文件: {FILE_PATH}，正在读取并进行 ACF 分析...")

# 2. 读取数据与中文显示配置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC',
    'WenQuanYi Zen Hei', 'KaiTi', 'Arial Unicode MS'
]
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

df = pd.read_excel(FILE_PATH)
df['销售日期'] = pd.to_datetime(df['销售日期'])

# 获取所有蔬菜品类
categories = df['品类名称'].dropna().unique()
n_cats = len(categories)
print(f"共检测到 {n_cats} 个蔬菜品类: {list(categories)}")

# 3. 循环绘制每个品类的 ACF 自相关图
cols = 3
rows = (n_cats + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
axes = np.array(axes).reshape(-1)

for i, cat in enumerate(categories):
    ax = axes[i]
    # 过滤出当前品类
    cat_df = df[df['品类名称'] == cat].sort_values('销售日期')
    
    # 修复点：只对销量这一列（Series）建立时间索引并重采样补 0
    series = cat_df.set_index('销售日期')['日总销量(千克)']
    
    # 补全连续日历天，如果某天没卖则销量补 0
    series = series.asfreq('D', fill_value=0)
    
    # 绘制 ACF 自相关图（滞后30天）
    plot_acf(series, lags=30, ax=ax, title=f'{cat} - 销量自相关图 (ACF)', color='darkblue')
    ax.set_xlabel('滞后阶数 (天)')
    ax.set_ylabel('自相关系数')
    ax.grid(True, linestyle='--', alpha=0.5)

# 隐藏多余的空白子图
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

# 保存论文级高清图片
output_img = "各品类ACF自相关分析图.png"
plt.savefig(output_img, dpi=300)
print(f"ACF 图表已成功生成并保存为: {output_img}")
plt.show()