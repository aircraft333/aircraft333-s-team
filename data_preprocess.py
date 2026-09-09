import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
PATH = "\data\附件1.xlsx"
def preprocess_data(PATH):
    # 读取Excel文件
    df = pd.read_excel(PATH, sheet_name=None)

    # 获取所有工作表的名称
    sheet_names = df.keys()

    # 创建一个空的DataFrame来存储合并后的数据
    merged_df = pd.DataFrame()

    # 遍历每个工作表并合并数据
    for sheet_name in sheet_names:
        sheet_df = df[sheet_name]
        merged_df = pd.concat([merged_df, sheet_df], ignore_index=True)

    # 数据清洗和预处理
    merged_df.dropna(inplace=True)  # 删除缺失值
    merged_df.drop_duplicates(inplace=True)  # 删除重复值

    # 数据标准化（示例：将数值列标准化到0-1范围）
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    merged_df[numeric_cols] = (merged_df[numeric_cols] - merged_df[numeric_cols].min()) / (merged_df[numeric_cols].max() - merged_df[numeric_cols].min())

    return merged_df