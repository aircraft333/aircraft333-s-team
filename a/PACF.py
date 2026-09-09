import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import pacf
import matplotlib
import matplotlib.font_manager as fm

# 强制重建字体缓存并指定微软雅黑
matplotlib.font_manager._load_fontmanager(try_read_cache=False)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# ----------------- 1. 图表样式与字体设置 (支持中文显示) -----------------
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.family'] = 'sans-serif' #增强字体渲染清晰度（可选）
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 创建输出目录
output_dir = "pacf_results"
os.makedirs(output_dir, exist_ok=True)

# ----------------- 2. 偏相关矩阵计算函数 (Partial Correlation) -----------------
def partial_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    通过协方差矩阵的逆（精度矩阵 Precision Matrix）计算多变量之间的偏相关系数矩阵。
    偏相关系数衡量的是：在剔除其他所有品类线性影响后，品类 i 与品类 j 之间的真实直接相关性。
    """
    # 丢弃含有 NaN 的行或填充
    clean_df = df.dropna()
    cov_matrix = clean_df.cov().values
    
    # 计算精度矩阵 (逆协方差矩阵)
    try:
        inv_cov = np.linalg.pinv(cov_matrix)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6)
        
    d = np.sqrt(np.diag(inv_cov))
    pcorr = -inv_cov / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)
    
    return pd.DataFrame(pcorr, index=df.columns, columns=df.columns)

# ----------------- 3. 主分析流程 -----------------
def run_analysis(file_path: str, date_col: str = None):
    """
    参数:
        file_path: 数据集路径 (支持 .xlsx 或 .csv)
        date_col: 日期列名称 (如果为 None，则假设第一列是日期)
    """
    # 1. 读取数据
    print("正在读取数据...")
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
        
    # 如果指定了日期列，将其设为索引并转为时间序列
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
        df.set_index(date_col, inplace=True)
    elif pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]) or '日期' in str(df.columns[0]) or 'date' in str(df.columns[0]).lower():
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.sort_values(by=df.columns[0])
        df.set_index(df.columns[0], inplace=True)

    # 仅保留数值列（各品类销量）
    numeric_df = df.select_dtypes(include=[np.number])
    # 填充缺失值 (前向填充 + 均值填充)
    numeric_df = numeric_df.ffill().bfill().fillna(0)
    
    categories = numeric_df.columns.tolist()
    print(f"检测到 {len(categories)} 个品类: {categories}")

    # ----------------- A. 各品类自身的 PACF 分析 -----------------
    max_lags = min(30, len(numeric_df) // 2 - 1)  # 设定最大滞后期数
    pacf_results = {}

    for cat in categories:
        # 计算偏自相关系数 (使用 ywm 方法)
        pacf_vals, confint = pacf(numeric_df[cat], nlags=max_lags, method='ywm', alpha=0.05)
        pacf_results[cat] = pacf_vals

    df_pacf = pd.DataFrame(pacf_results, index=[f"Lag_{i}" for i in range(max_lags + 1)])
    df_pacf.to_excel(os.path.join(output_dir, "各品类自身偏自相关系数_PACF.xlsx"))
    print("-> 各品类自身 PACF 计算完成，已保存为 Excel。")

    # 绘制各品类 PACF 对比热力图
    plt.figure(figsize=(12, max(6, len(categories) * 0.4)))
    sns.heatmap(df_pacf.T, cmap="coolwarm", center=0, annot=False, cbar_kws={'label': 'PACF 值'})
    plt.title(f"各品类不同滞后期偏自相关系数 (PACF) 热力图 (Lag 0 ~ {max_lags})", fontsize=14, fontweight='bold')
    plt.xlabel("滞后期数 (Lags)", fontsize=12)
    plt.ylabel("品类名称", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "PACF_各品类滞后热力图.png"), dpi=300)
    plt.close()

    # ----------------- B. 品类与品类之间的【偏相关系数矩阵】 -----------------
    # 普通 Pearson 相关 vs 偏相关 (Partial Correlation)
    pearson_corr = numeric_df.corr()
    partial_corr = partial_correlation_matrix(numeric_df)

    # 保存偏相关矩阵
    partial_corr.to_excel(os.path.join(output_dir, "品类间偏相关系数矩阵_Partial_Correlation.xlsx"))
    print("-> 品类间偏相关矩阵计算完成，已保存为 Excel。")

    # 绘制对比热力图 (普通相关 vs 偏相关)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Pearson
    sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=axes[0],
                cbar_kws={'label': 'Pearson Correlation'})
    axes[0].set_title("品类间普通相关系数 (含间接影响与趋势叠加)", fontsize=13, fontweight='bold')

    # Partial
    sns.heatmap(partial_corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=axes[1],
                cbar_kws={'label': 'Partial Correlation'})
    axes[1].set_title("品类间偏相关系数 (已控制剔除其他品类的协同影响)", fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "品类间相关性对比_Pearson_vs_Partial.png"), dpi=300)
    plt.close()
    print(f"-> 分析完成！所有图表和表格均已导出至 `{output_dir}/` 文件夹。")

# ----------------- 4. 运行入口 -----------------
if __name__ == "__main__":
    # 请将下面的文件名替换为你实际的文件路径
    DATA_FILE = "a/各品类每日总销量.xlsx"  # 或者 "各品类每日总销量.xlsx"
    
    if os.path.exists(DATA_FILE):
        run_analysis(DATA_FILE)
    else:
        # 如果未找到文件，生成一份模拟数据供测试验证
        print(f"未在路径 `{DATA_FILE}` 找到文件，正在生成模拟品类销量数据进行演示...")
        dates = pd.date_range("2024-01-01", periods=180, freq="D")
        np.random.seed(42)
        
        # 构造几个具有内在关联与自相关特性的品类
        base_trend = np.sin(np.linspace(0, 10, 180)) * 20
        cat_A = 100 + base_trend + np.random.normal(0, 5, 180)
        cat_B = 80 + 0.6 * cat_A + np.random.normal(0, 3, 180)  # B 受 A 直接影响
        cat_C = 50 + 0.4 * cat_B + np.random.normal(0, 4, 180)  # C 受 B 影响，但受 A 间接影响
        cat_D = 120 - 0.3 * cat_A + np.random.normal(0, 6, 180)
        
        dummy_df = pd.DataFrame({
            "日期": dates,
            "品类_叶菜类": cat_A,
            "品类_根茎类": cat_B,
            "品类_水生根茎类": cat_C,
            "品类_茄类": cat_D
        })
        dummy_path = "mock_sales_data.xlsx"
        dummy_df.to_excel(dummy_path, index=False)
        
        run_analysis(dummy_path, date_col="日期")