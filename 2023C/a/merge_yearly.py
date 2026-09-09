# -*- coding: utf-8 -*-
"""
脚本功能：
根据《附件2.xlsx》，按【年份】合并汇总所有蔬菜的总销售量（千克），并导出结果。
"""
import os
import pandas as pd

# 1. 自动定位文件路径
def find_file(filename):
    for p in [filename, os.path.join("data", filename), os.path.join("..", filename)]:
        if os.path.exists(p):
            return p
    return None

file_path = find_file("附件2.xlsx") or find_file("附件2.csv")
if not file_path:
    raise FileNotFoundError("❌ 未找到《附件2.xlsx》，请确保文件在当前脚本目录或 data/ 目录下！")

print(f"正在读取数据: {file_path} ...")
# 读取销售流水
df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

# 2. 提取年份并合并销量
# 转换日期格式提取年份
df['销售日期'] = pd.to_datetime(df['销售日期'])
df['年份'] = df['销售日期'].dt.year

# 过滤缺失数据
df = df.dropna(subset=['年份', '销量(千克)'])
df['年份'] = df['年份'].astype(int)

# 核心合并：按年份对所有蔬菜销量求和
yearly_sales = df.groupby('年份').agg(
    总销售量_千克=('销量(千克)', 'sum'),
    统计天数=('销售日期', 'nunique')
).reset_index()

# 计算日均销量与吨数换算（方便论文描述）
yearly_sales['总销售量(千克)'] = yearly_sales['总销售量_千克'].round(2)
yearly_sales['总销售量(吨)'] = (yearly_sales['总销售量_千克'] / 1000.0).round(3)
yearly_sales['日均销售量(千克)'] = (yearly_sales['总销售量_千克'] / yearly_sales['统计天数']).round(2)

# 整理输出列
result = yearly_sales[['年份', '总销售量(千克)', '总销售量(吨)', '日均销售量(千克)', '统计天数']]

# 3. 导出并展示结果
output_file = "历年所有蔬菜总销量合并表.xlsx"
result.to_excel(output_file, index=False)

print("\n" + "=" * 55)
print(f"✅ 合并完成！结果已保存至: {os.path.abspath(output_file)}")
print("=" * 55)
print("【合并结果预览】\n")
print(result.to_string(index=False))
print("=" * 55)