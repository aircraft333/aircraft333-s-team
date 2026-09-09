# -*- coding: utf-8 -*-
"""共用：读取附件、解析孕周、中文显示设置（Q1~Q4 复用）"""
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = "附件.xlsx"


def parse_weeks(s):
    """'11w+6'/'13w' -> 连续孕周(周数+天数/7)"""
    if pd.isna(s):
        return np.nan
    m = re.fullmatch(r"(\d+)\s*w(?:\+(\d+))?", str(s).strip().lower())
    if not m:
        return np.nan
    return int(m.group(1)) + (int(m.group(2)) if m.group(2) else 0) / 7.0


def setup_chinese():
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "WenQuanYi Zen Hei", "KaiTi", "Arial Unicode MS"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_male():
    """男胎表：英文列名 + 连续孕周"""
    df = pd.read_excel(PATH, sheet_name="男胎检测数据")
    sub = df[[
        "孕妇代码", "年龄", "身高", "体重", "检测孕周", "孕妇BMI",
        "Y染色体浓度", "染色体的非整倍体", "胎儿是否健康",
    ]].rename(columns={
        "孕妇代码": "pid", "年龄": "age", "身高": "height", "体重": "weight",
        "检测孕周": "gw_raw", "孕妇BMI": "bmi",
        "Y染色体浓度": "yconc", "染色体的非整倍体": "ab",
        "胎儿是否健康": "health",
    })
    sub["gw"] = sub["gw_raw"].map(parse_weeks)
    sub = sub.dropna(subset=["gw", "bmi", "yconc"])
    sub = sub[(sub["gw"] >= 10) & (sub["gw"] <= 30)]
    return sub.reset_index(drop=True)


def load_female():
    """女胎表：英文列名 + 连续孕周 + AB标签（先去掉表头空格再映射）"""
    df = pd.read_excel(PATH, sheet_name="女胎检测数据")
    df.columns = [str(c).replace(" ", "") if "Unnamed" not in str(c)
                  else str(c) for c in df.columns]
    cols = {
        "孕妇代码": "pid", "年龄": "age", "身高": "height",
        "体重": "weight", "检测孕周": "gw_raw", "孕妇BMI": "bmi",
        "原始读段数": "n_raw", "在参考基因组上比对的比例": "map_ratio",
        "重复读段的比例": "dup_ratio", "唯一比对的读段数": "n_uniq",
        "GC含量": "gc", "13号染色体的Z值": "z13", "18号染色体的Z值": "z18",
        "21号染色体的Z值": "z21", "X染色体的Z值": "zx",
        "X染色体浓度": "xc", "13号染色体的GC含量": "gc13",
        "18号染色体的GC含量": "gc18", "21号染色体的GC含量": "gc21",
        "被过滤掉读段数的比例": "filter_ratio",
        "染色体的非整倍体": "ab", "怀孕次数": "gravida",
        "生产次数": "para", "胎儿是否健康": "health",
    }
    sub = df[[c for c in cols if c in df.columns]].rename(
        columns={k: v for k, v in cols.items() if k in df.columns})
    sub["gw"] = sub["gw_raw"].map(parse_weeks)
    sub["label"] = sub["ab"].notna().astype(int)
    return sub.reset_index(drop=True)
