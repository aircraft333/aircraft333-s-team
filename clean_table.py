import pandas as pd

# 1. 读取 Excel 文件，将第一列作为索引（index_col=0），第一行默认作为表头
df_load = pd.read_excel('小区用电量.xlsx', index_col=0)
df_pv = pd.read_excel('光伏.xlsx', index_col=0)

# 2. 矩阵对应位置直接相减：净用电量 = 小区用电量 - 光伏出力
df_net = df_load - df_pv

# 3. （可选业务处理）若净用电量小于0表示光伏倒送网，若不允许倒送需截断为0，可取消下行注释：
# df_net = df_net.clip(lower=0)

# 4. 保存为新的 Excel 文件，保留原有的行索引和列标签
df_net.to_excel('净用电量.xlsx')

print("计算完成，已生成 净用电量.xlsx！")