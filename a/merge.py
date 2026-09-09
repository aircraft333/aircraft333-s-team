import os
import pandas as pd

def merge_and_aggregate_sales(item_info_path, sales_flow_path, output_path="各品类每日总销量.xlsx"):
    # 检查文件是否存在
    if not os.path.exists(item_info_path):
        print(f"❌ 找不到表1文件: {item_info_path}，请检查文件名和路径！")
        return None
    if not os.path.exists(sales_flow_path):
        print(f"❌ 找不到表2文件: {sales_flow_path}，请检查文件名和路径！")
        return None

    print(">>> 1. 正在读取数据...")
    def read_file(path):
        if path.endswith('.csv'):
            return pd.read_csv(path)
        return pd.read_excel(path)

    df_info = read_file(item_info_path)
    df_sales = read_file(sales_flow_path)

    # ------------------------------------------------------------------
    # 自动识别常见列名（无需手动改列名，只要包含相关字眼就能认出来）
    # ------------------------------------------------------------------
    def find_col(df, candidates):
        for c in df.columns:
            if any(k in str(c) for k in candidates):
                return c
        return None

    info_code_col = find_col(df_info, ['单品编码', '编码', '编号', 'code'])
    info_cat_col  = find_col(df_info, ['分类名称', '品类', '大类', '类别', 'category'])
    
    sales_date_col = find_col(df_sales, ['销售日期', '日期', 'date', '时间'])
    sales_code_col = find_col(df_sales, ['单品编码', '编码', '编号', 'code'])
    sales_qty_col  = find_col(df_sales, ['销量', '销售数量', '数量', 'qty', 'weight'])

    print(f"    表1识别到: 编码列[{info_code_col}], 品类列[{info_cat_col}]")
    print(f"    表2识别到: 日期列[{sales_date_col}], 编码列[{sales_code_col}], 销量列[{sales_qty_col}]")

    # 格式统一清洗
    df_info[info_code_col] = df_info[info_code_col].astype(str).str.strip()
    df_sales[sales_code_col] = df_sales[sales_code_col].astype(str).str.strip()
    df_sales[sales_date_col] = pd.to_datetime(df_sales[sales_date_col]).dt.date
    df_sales[sales_qty_col] = pd.to_numeric(df_sales[sales_qty_col], errors='coerce').fillna(0)

    print(">>> 2. 正在通过单品编号合并两表...")
    merged_df = pd.merge(
        df_sales[[sales_date_col, sales_code_col, sales_qty_col]],
        df_info[[info_code_col, info_cat_col]],
        left_on=sales_code_col,
        right_on=info_code_col,
        how='left'
    )

    print(">>> 3. 正在按【日期 + 品类】聚合求和...")
    daily_sales = merged_df.groupby([sales_date_col, info_cat_col], as_index=False)[sales_qty_col].sum()
    daily_sales.columns = ['销售日期', '品类名称', '日总销量(千克)']
    daily_sales.sort_values(by=['销售日期', '品类名称'], inplace=True)

    # 导出文件
    daily_sales.to_excel(output_path, index=False)
    print(f"\n🎉 运行成功！合并后的文件已保存到: {output_path}\n")
    print("前 10 行预览：")
    print(daily_sales.head(10))
    return daily_sales

# ==========================================
# ⚠️ 这里只需要确认两张表的名字即可：
# ==========================================
if __name__ == '__main__':
    # 请确保这两个名字与你 a 文件夹下的表格文件名一致（带上 .xlsx）
    TABLE_1 = "附件1.xlsx"  # 单品与品类对应的表
    TABLE_2 = "附件2.xlsx"  # 销售流水表

    merge_and_aggregate_sales(TABLE_1, TABLE_2, "各品类每日总销量.xlsx")