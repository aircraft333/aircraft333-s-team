# -*- coding: utf-8 -*-
"""2023C 数据预处理：找出应剔除的单品（无销量 / 销售天数过少 / 销量占比过低）

数据来源：
    data/附件1.xlsx —— 单品信息（单品编码、分类名称等）
    data/附件2.xlsx —— 销售流水（销售日期、扫码销售时间、单品编码、销量(千克)、销售单价(元/千克)）

流程：
    第一步：找附件1 里有、但附件2 中没有任何销售记录的单品（无销量）
    第二步：销售天数 <= threshold_1(=10) 的单品（销售天数过少）
    第三步：累计销量占比 < threshold_2(=0.00003) 的单品（销量过低）
    第四步：取第二步与第三步的交集
    第五步：挑一个交集单品查看其销量随时间的变化
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

plt.rcParams['font.sans-serif'] = [u'simHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据读入
csv_file = 'data/附件1.xlsx'
df_1 = pd.read_excel(csv_file)   # 单品信息
csv_file = 'data/附件2.xlsx'
df = pd.read_excel(csv_file)     # 销售流水

# 将指定列转换为时间序列
df['销售日期'] = pd.to_datetime(df['销售日期'])
df['扫码销售时间'] = pd.to_datetime(
    df['销售日期'].dt.strftime('%Y-%m-%d') + ' ' + df['扫码销售时间'].astype(str),
    errors='coerce'
)
# 计算销售金额
df['销售金额'] = df['销量(千克)'] * df['销售单价(元/千克)']
# 分品类
mapping_dict = df_1.set_index('单品编码')['分类名称'].to_dict()
df['品类'] = df['单品编码'].map(mapping_dict)
print(df.head(5))

####### 第一步：处理没有销量的数据 ######
unique_values_df = df['单品编码'].unique()
unique_values_df_1 = df_1['单品编码'].unique()
values_only_in_df_1 = set(unique_values_df_1) - set(unique_values_df)
count_values_only_in_df_1 = len(values_only_in_df_1)

print("df列'单品编码'的唯一值个数：", len(unique_values_df))
print("df_1列'单品编码'的唯一值个数：", len(unique_values_df_1))
print("df_1中有但是df中没有的值：", values_only_in_df_1)
print("这些值的个数：", count_values_only_in_df_1)  # 5

####### 第二步：销售天数少（阈值1）的单品找出来 ########
threshold_1 = 10

result = df.groupby('单品编码')['销售日期'].nunique().reset_index()
result.rename(columns={'销售日期': '销售天数'}, inplace=True)

hist, bins = np.histogram(result['销售天数'], bins=10)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
cmap = plt.cm.coolwarm
norm = plt.Normalize(vmin=min(hist), vmax=max(hist))
colors = cmap(norm(hist))
plt.figure(figsize=(8, 6))
bars = plt.bar(bin_centers, hist, width=bins[1] - bins[0], color=colors, edgecolor='k',
               alpha=0.7)
for i, count in enumerate(hist):
    plt.text(bin_centers[i], count + 5, str(count), ha='center', va='bottom')
plt.xlabel('销售天数')
plt.ylabel('单品数')
plt.title('销售天数分布直方图')
plt.grid(True)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical')
cbar.set_label('计数', rotation=90, labelpad=15)
plt.show()

filtered_result = result[result['销售天数'] <= threshold_1]
count = filtered_result.shape[0]
# print(f"销售天数小于等于 {threshold_1} 的单品编码和数量:")
list_1 = []
for index, row in filtered_result.iterrows():
    if row['销售天数'] <= threshold_1:
        list_1.append(row['单品编码'])
print(f'\n阈值为{threshold_1}时被筛除的单品数量: {count}')
print("分别是:")
print(list_1)

threshold_2 = 0.00003
######### 第三步：销量低（阈值 2）的单品找出来 #########
grouped = df.groupby('单品编码')['销量(千克)'].sum().reset_index()
print(len(grouped))
total_sales = grouped['销量(千克)'].sum()
grouped['销量占比'] = grouped['销量(千克)'] / total_sales
low_percentage_groups = grouped[grouped['销量占比'] < threshold_2]['单品编码']
list_2 = low_percentage_groups.to_list()
print("\n所有组的销量总和:", total_sales)
print(f"销量占比低于{threshold_2}的组的单品编码:")
print(list_2)
print(f"总数是：{len(low_percentage_groups)}")

data_g = []
for i in grouped['销量占比']:
    if i <= threshold_2:
        data_g.append(i)
bins = 10
n, bins, patches = plt.hist(data_g, bins=bins, edgecolor='k')
plt.xlabel('销量占比')
plt.ylabel('频数')
plt.title('销量占比直方图')
plt.grid()
for i, rect in enumerate(patches):
    height = rect.get_height()
    plt.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 5), textcoords='offset points',
                 ha='center', va='bottom')
plt.show()

########### 第四步：取交集 ###########
intersection = list(set(list_1) & set(list_2))
print(f"\n交集数量为：{len(intersection)}")
print(intersection)

############ 第五步：查看 ###########
if len(intersection) >= 2:
    target_item = intersection[1]
else:
    target_item = intersection[0] if intersection else None

if target_item is not None:
    # 选择特定的单品编码
    grouped = df.groupby(['单品编码', '销售日期'])['销量(千克)'].sum().reset_index()
    filtered_df = grouped[grouped['单品编码'] == target_item]
    print(filtered_df)
    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_df['销售日期'], filtered_df['销量(千克)'], marker='o', linestyle='-')
    plt.title(f'单品编码 {target_item} 的销售日期和销量折线图')
    plt.xlabel('销售日期')
    plt.ylabel('销售金额')
    plt.grid(True)
    plt.show()
else:
    print("交集为空，没有可查看的目标单品。")

####### 第六步：统计各品类的最大值，最小值，平均值，中位数，标准差，方差，偏度，峰度等指标 ########
# 口径：以"每个销售日某品类的总销量(千克)"为样本，统计六大品类的分布特征。
# 若想改成"品类内各单品总销量"的分布，只需把下面 groupby 里的 '销售日期' 换成 '单品编码'。
daily_cat = df.groupby(['品类', '销售日期'])['销量(千克)'].sum().reset_index()


def _skew(s):
    return s.skew()


def _kurt(s):
    return s.kurt()


cat_stats = daily_cat.groupby('品类').agg(
    销售天数=('销量(千克)', 'count'),   # 有销售的天数
    总销量kg=('销量(千克)', 'sum'),
    最大值=('销量(千克)', 'max'),
    最小值=('销量(千克)', 'min'),
    平均值=('销量(千克)', 'mean'),
    中位数=('销量(千克)', 'median'),
    标准差=('销量(千克)', 'std'),
    方差=('销量(千克)', 'var'),
    偏度=('销量(千克)', _skew),
    峰度=('销量(千克)', _kurt),
)
cat_stats = cat_stats.sort_values('总销量kg', ascending=False)
# 附加指标：变异系数（衡量相对离散程度）与极差
cat_stats['变异系数'] = cat_stats['标准差'] / cat_stats['平均值']
cat_stats['极差'] = cat_stats['最大值'] - cat_stats['最小值']

print("\n####### 第六步：各品类日销量(kg/天)分布统计 #######")
print(cat_stats.round(4).to_string())

# 箱线图：直观展示各品类日销量的分布形态
order = cat_stats.index.tolist()
box_data = [daily_cat.loc[daily_cat['品类'] == c, '销量(千克)'].values for c in order]
plt.figure(figsize=(10, 6))
plt.boxplot(box_data, tick_labels=order, showmeans=True)
plt.yscale('log')  # 品类间量级差异大，取对数便于观察
plt.xlabel('品类')
plt.ylabel('日销量(千克)')
plt.title('六大品类日销量分布箱线图（对数轴，▲=均值）')
plt.grid(axis='y', alpha=0.3)
plt.show()

