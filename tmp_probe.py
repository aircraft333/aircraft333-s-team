# -*- coding: utf-8 -*-
"""探测数据质量：行数/取值/缺失/关联，为 Q1 分析脚本做准备"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

BASE = r"c:\Users\dell\Downloads\56RoZe4zc@c3A\C题\data"

def clean_cols(df):
    df.columns = [str(c).replace(" ", "").replace("\u3000", "") for c in df.columns]
    return df

# 附件1
a1 = clean_cols(pd.read_excel(os.path.join(BASE, "附件1.xlsx")))
print("附件1:", a1.shape)
print(" 分类计数:\n", a1["分类名称"].value_counts().to_string())
print(" 是否有重复单品编码:", a1["单品编码"].duplicated().any())

# 附件2
a2 = clean_cols(pd.read_excel(os.path.join(BASE, "附件2.xlsx"),
                              parse_dates=["销售日期"]))
print("\n附件2:", a2.shape)
print(" 日期范围:", a2["销售日期"].min(), "~", a2["销售日期"].max())
print(" 销售类型取值:\n", a2["销售类型"].value_counts().to_string())
print(" 是否打折销售取值:\n", a2["是否打折销售"].value_counts().to_string())
print(" 销量<=0 行数:", (a2["销量(千克)"] <= 0).sum())
print(" 单价<=0 行数:", (a2["销售单价(元/千克)"] <= 0).sum())
print(" 空值统计:\n", a2.isna().sum().to_string())
print(" 单品数(附件2):", a2["单品编码"].nunique())

# 关联检查
codes_a1 = set(a1["单品编码"])
codes_a2 = set(a2["单品编码"])
print("\n附件1中有但附件2从未销售的单品数:", len(codes_a1 - codes_a2))
print("附件2中有但附件1缺失(无分类)的单品数:", len(codes_a2 - codes_a1))
miss = list(codes_a2 - codes_a1)[:5]
print("  例子:", miss)

# 附件3 / 附件4 简览
a3 = clean_cols(pd.read_excel(os.path.join(BASE, "附件3.xlsx"), parse_dates=["日期"]))
print("\n附件3:", a3.shape, "日期", a3["日期"].min(), "~", a3["日期"].max(),
      "批发价缺失:", a3["批发价格(元/千克)"].isna().sum())

xl4 = pd.ExcelFile(os.path.join(BASE, "附件4.xlsx"))
sh = xl4.sheet_names[0]
a4a = clean_cols(xl4.parse(sh))
print("\n附件4[sheet1 平均损耗率]:", a4a.shape)
print(a4a.to_string())
a4b = clean_cols(xl4.parse("Sheet1"))
print("附件4[sheet2 单品损耗率]:", a4b.shape, "损耗缺失:", a4b["损耗率(%)"].isna().sum())
