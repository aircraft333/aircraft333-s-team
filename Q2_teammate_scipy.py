import numpy as np
import pandas as pd
from scipy.optimize import linprog
import os
import time

# ==========================================
# 1. 物理参数配置
# ==========================================
DELTA_T = 10 / 60.0  # 10 分钟 = 1/6 小时
N_SLOTS = 144        # 每天 144 个时段

BAT_CAP_MAX = 12000.0   # 额定容量 (kWh)
BAT_SOC_MIN = 1200.0    # 最小储电量 (kWh)
BAT_SOC_MAX = 10800.0   # 最大储电量 (kWh)
BAT_P_MAX   = 5000.0    # 最大充放电功率 (kW)
BAT_EFF     = 0.90      # 充放电效率
INITIAL_SOC = 6000.0    # 1月1日 初始 SOC

# ==========================================
# 2. 从《附件1.xlsx》加载真实 144 时段电价
# ==========================================
def load_price_from_excel(price_file_path='附件1.xlsx'):
    if not os.path.exists(price_file_path):
        raise FileNotFoundError(f"未找到电价文件: {price_file_path}，请确认文件路径是否正确。")
    
    df_price = pd.read_excel(price_file_path)
    price_array = df_price.iloc[:N_SLOTS, 1].to_numpy(dtype=float)
    return price_array

# ==========================================
# 3. 电价自适应情景生成器 (日前预测用)
# ==========================================
def generate_scenarios(history_df, target_date, price_vector, hist_window=14):
    hist_dates = history_df.index
    target_dow = target_date.weekday()
    
    valid_dates = [d for d in hist_dates if (target_date - d).days > 0][-hist_window:]
    
    days_data = []
    weights = []
    for d in valid_dates:
        days_diff = (target_date - d).days
        w = np.exp(-days_diff / 4.5)
        if d.weekday() == target_dow:
            w *= 1.8
        days_data.append(history_df.loc[d].values[:N_SLOTS])
        weights.append(w)
        
    days_data = np.array(days_data)
    weights = np.array(weights)
    weights /= np.sum(weights)
    
    p50_baseline = np.sum(days_data * weights[:, None], axis=0)
    if len(days_data) >= 3:
        recent_3d = np.mean(days_data[-3:], axis=0)
        p50 = 0.65 * p50_baseline + 0.35 * recent_3d
    else:
        p50 = p50_baseline
        
    residuals = days_data - p50
    
    p_norm = (price_vector - price_vector.min()) / (price_vector.max() - price_vector.min() + 1e-6)
    q_high_vec = 75 + 15 * p_norm
    q_ext_vec  = 90 + 7 * p_norm
    
    high_diff = np.zeros(N_SLOTS)
    ext_diff = np.zeros(N_SLOTS)
    for t in range(N_SLOTS):
        high_diff[t] = np.percentile(residuals[:, t], q_high_vec[t])
        ext_diff[t] = np.percentile(residuals[:, t], q_ext_vec[t])
        
    s_low  = p50 + np.percentile(residuals, 20, axis=0)
    s_mid  = p50
    s_high = p50 + high_diff
    s_ext  = p50 + ext_diff
    
    scenarios = [s_low, s_mid, s_high, s_ext]
    probabilities = [0.15, 0.50, 0.25, 0.10]
    
    return scenarios, probabilities

# ==========================================
# 4. 日前两阶段随机规划求解器 (确定 b_t 计划)
# ==========================================
def solve_two_stage_stochastic_schedule(scenarios, probabilities, start_soc, target_month, price_vector):
    T = N_SLOTS
    S = len(scenarios)
    
    if target_month in [5, 6, 7, 8, 9]:
        target_end_soc = 4500.0
    else:
        target_end_soc = 5800.0
        
    total_vars = T + S * (4 * T + 1)
    c = np.zeros(total_vars)
    
    c[0:T] = price_vector * DELTA_T  # 日前购电成本
    
    avg_price = np.mean(price_vector)
    for s in range(S):
        prob = probabilities[s]
        base_idx = T + s * (4 * T + 1)
        
        c[base_idx : base_idx + T] = prob * 1e-4
        c[base_idx + T : base_idx + 2*T] = prob * 1e-4
        c[base_idx + 2*T : base_idx + 3*T] = prob * (5.0 * price_vector * DELTA_T)
        c[base_idx + 3*T : base_idx + 4*T] = prob * 1e-3
        c[base_idx + 4*T] = prob * (avg_price * 1.1)

    A_eq = np.zeros((S * T, total_vars))
    b_eq = np.zeros(S * T)
    
    for s in range(S):
        L_s = scenarios[s]
        base_idx = T + s * (4 * T + 1)
        for t in range(T):
            row = s * T + t
            A_eq[row, t] = 1.0
            A_eq[row, base_idx + t] = -1.0
            A_eq[row, base_idx + T + t] = 1.0
            A_eq[row, base_idx + 2*T + t] = 1.0
            A_eq[row, base_idx + 3*T + t] = -1.0
            b_eq[row] = L_s[t]

    A_ub = np.zeros((S * (2 * T + 1), total_vars))
    b_ub = np.zeros(S * (2 * T + 1))
    
    for s in range(S):
        base_idx = T + s * (4 * T + 1)
        ub_row_base = s * (2 * T + 1)
        
        for t in range(T):
            A_ub[ub_row_base + t, base_idx : base_idx + t + 1] = BAT_EFF * DELTA_T
            A_ub[ub_row_base + t, base_idx + T : base_idx + T + t + 1] = -(1.0 / BAT_EFF) * DELTA_T
            b_ub[ub_row_base + t] = BAT_SOC_MAX - start_soc
            
            A_ub[ub_row_base + T + t, base_idx : base_idx + t + 1] = -BAT_EFF * DELTA_T
            A_ub[ub_row_base + T + t, base_idx + T : base_idx + T + t + 1] = (1.0 / BAT_EFF) * DELTA_T
            b_ub[ub_row_base + T + t] = start_soc - BAT_SOC_MIN

        A_ub[ub_row_base + 2*T, base_idx : base_idx + T] = -BAT_EFF * DELTA_T
        A_ub[ub_row_base + 2*T, base_idx + T : base_idx + 2*T] = (1.0 / BAT_EFF) * DELTA_T
        A_ub[ub_row_base + 2*T, base_idx + 4*T] = -1.0
        b_ub[ub_row_base + 2*T] = start_soc - target_end_soc

    bounds = [(0, None)] * T
    for s in range(S):
        bounds += [(0, BAT_P_MAX)] * T
        bounds += [(0, BAT_P_MAX)] * T
        bounds += [(0, None)] * T
        bounds += [(0, None)] * T
        bounds += [(0, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success and res.x is not None:
        b_opt_power = res.x[0:T]
    else:
        b_opt_power = np.maximum(0.0, scenarios[1])

    b_plan_energy = b_opt_power * DELTA_T
    return b_plan_energy, b_opt_power

# ==========================================
# 5. 【核心升级】日内最优分配求解器 (Intra-day Optimal Dispatch LP)
# ==========================================
def solve_intraday_optimal_operation(actual_net_load, b_plan_energy, current_soc, price_vector):
    """
    日内已知真实净负荷 actual_net_load 与日前申报 b_plan_energy 后，
    通过日内二次 LP 求解全局最优充放电与最小紧急买电
    """
    T = N_SLOTS
    b_plan_power = b_plan_energy / DELTA_T
    
    # 决策变量: [pch(0..T-1), pdis(0..T-1), e(0..T-1), pcurt(0..T-1)] 共 4*T 个
    n_vars = 4 * T
    c = np.zeros(n_vars)
    
    # 目标: 最小化 5 倍紧急买电罚金 + 微小充放电磨损
    c[0 : T]       = 1e-5                               # pch 损耗惩罚
    c[T : 2*T]     = 1e-5                               # pdis 损耗惩罚
    c[2*T : 3*T]   = 5.0 * price_vector * DELTA_T       # 紧急买电 5 倍罚金
    c[3*T : 4*T]   = 1e-4                               # 弃光惩罚
    
    # 等式约束: 功率平衡 pdis - pch + e - pcurt = actual_net_load - b_plan_power
    A_eq = np.zeros((T, n_vars))
    b_eq = np.zeros(T)
    for t in range(T):
        A_eq[t, t]       = -1.0   # -pch
        A_eq[t, T + t]   = 1.0    # +pdis
        A_eq[t, 2*T + t] = 1.0    # +e
        A_eq[t, 3*T + t] = -1.0   # -pcurt
        b_eq[t]          = actual_net_load[t] - b_plan_power[t]
        
    # 不等式约束: SOC 上下限约束 (共 2*T 个约束)
    A_ub = np.zeros((2 * T, n_vars))
    b_ub = np.zeros(2 * T)
    for t in range(T):
        # E_t <= 10800
        A_ub[t, 0 : t + 1]       = BAT_EFF * DELTA_T
        A_ub[t, T : T + t + 1]   = -(1.0 / BAT_EFF) * DELTA_T
        b_ub[t]                  = BAT_SOC_MAX - current_soc
        
        # E_t >= 1200
        A_ub[T + t, 0 : t + 1]     = -BAT_EFF * DELTA_T
        A_ub[T + t, T : T + t + 1] = (1.0 / BAT_EFF) * DELTA_T
        b_ub[T + t]                = current_soc - BAT_SOC_MIN

    bounds = [(0, BAT_P_MAX)] * T + [(0, BAT_P_MAX)] * T + [(0, None)] * T + [(0, None)] * T
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success and res.x is not None:
        opt_pch   = res.x[0 : T]
        opt_pdis  = res.x[T : 2*T]
        opt_e     = res.x[2*T : 3*T]
        opt_pcurt = res.x[3*T : 4*T]
    else:
        # 若极罕见数值异常，兜底为基础差值
        opt_pch = np.zeros(T)
        opt_pdis = np.zeros(T)
        opt_e = np.maximum(0.0, actual_net_load - b_plan_power)
        opt_pcurt = np.maximum(0.0, b_plan_power - actual_net_load)

    # 计算精确 SOC 轨迹
    soc_record = np.zeros(T + 1)
    soc_record[0] = current_soc
    for t in range(T):
        delta_e = (opt_pch[t] * BAT_EFF - opt_pdis[t] / BAT_EFF) * DELTA_T
        soc_record[t+1] = np.clip(soc_record[t] + delta_e, BAT_SOC_MIN, BAT_SOC_MAX)
        
    emergency_energy = opt_e * DELTA_T
    actual_ch_energy = opt_pch * DELTA_T
    actual_dis_energy = opt_pdis * DELTA_T
    
    return emergency_energy, actual_ch_energy, actual_dis_energy, soc_record

# ==========================================
# 6. 全年滚动优化主程序
# ==========================================
def run_master_optimization(net_load_file_path='净用电量.xlsx', price_file_path='附件1.xlsx', output_excel_path='result2.xlsx'):
    print("=" * 80)
    print("【微网顶层决策系统】启动：日前两阶段随机规划 + 日内全局最优分配二次调度")
    print("=" * 80)
    
    PRICE = load_price_from_excel(price_file_path)
    print(f"✓ 成功加载附件1电价 (波动范围: {PRICE.min():.4f} ~ {PRICE.max():.4f} 元/kWh)")
    
    df_raw = pd.read_excel(net_load_file_path)
    df_raw.rename(columns={df_raw.columns[0]: 'Date'}, inplace=True)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_raw.set_index('Date', inplace=True)
    all_dates = df_raw.index.sort_values()
    
    plan_buy_records = {}
    charge_dis_table2 = []
    emergency_records = []
    
    total_plan_cost = 0.0
    total_emerg_cost = 0.0
    total_emerg_kwh = 0.0
    
    current_soc = INITIAL_SOC
    
    # 1月预热
    print("正在执行 1月 状态预热与环境辨识...")
    jan_dates = [d for d in all_dates if d.month == 1]
    for d in jan_dates:
        p_act = df_raw.loc[d].values[:N_SLOTS]
        b_e, _ = solve_two_stage_stochastic_schedule([p_act], [1.0], current_soc, 1, PRICE)
        _, _, _, soc_rec = solve_intraday_optimal_operation(p_act, b_e, current_soc, PRICE)
        current_soc = soc_rec[-1]
    
    print(f"1月预热完毕，2月1日 0:00 初始 SOC: {current_soc:.2f} kWh")
    print("开始执行 2025.2.1 - 2025.12.31 全年滚动求解 (日内最优二次分配)...")
    
    sim_dates = [d for d in all_dates if d >= pd.Timestamp('2025-02-01')]
    period_indices = [
        (0, 24, "0:00-4:00"), (24, 48, "4:00-8:00"), (48, 72, "8:00-12:00"),
        (72, 96, "12:00-16:00"), (96, 120, "16:00-20:00"), (120, 144, "20:00-24:00")
    ]
    
    for d in sim_dates:
        d_str = d.strftime('%Y/%m/%d')
        history_df = df_raw.loc[df_raw.index < d]
        
        # 1. 日前情景生成与两阶段随机规划
        scenarios, probs = generate_scenarios(history_df, d, PRICE, hist_window=14)
        soc_0 = current_soc
        b_plan_energy, _ = solve_two_stage_stochastic_schedule(scenarios, probs, soc_0, d.month, PRICE)
        plan_buy_records[d_str] = b_plan_energy
        
        # 2. 日内全局最优分配 (二次 LP 优化调度)
        actual_net = df_raw.loc[d].values[:N_SLOTS]
        em_e, ch_e, dis_e, soc_rec = solve_intraday_optimal_operation(actual_net, b_plan_energy, soc_0, PRICE)
        soc_24 = soc_rec[-1]
        current_soc = soc_24
        
        # 3. 费用累计
        day_plan_cost = np.sum(b_plan_energy * PRICE)
        day_emerg_cost = np.sum(em_e * PRICE * 5.0)
        total_plan_cost += day_plan_cost
        total_emerg_cost += day_emerg_cost
        total_emerg_kwh += np.sum(em_e)
        
        # 4. 表2记录
        row_tab2 = {'日期': d_str, '0:00 储电量': round(soc_0, 2), '24:00 储电量': round(soc_24, 2)}
        for p_start, p_end, p_name in period_indices:
            row_tab2[f'{p_name} 充电量'] = round(np.sum(ch_e[p_start:p_end]), 2)
            row_tab2[f'{p_name} 放电量'] = round(np.sum(dis_e[p_start:p_end]), 2)
        charge_dis_table2.append(row_tab2)
        
        # 5. 表3记录 (合并连续时段)
        t_idx = 0
        while t_idx < N_SLOTS:
            if em_e[t_idx] > 1e-4:
                start_slot = t_idx
                end_slot = t_idx
                while end_slot + 1 < N_SLOTS and em_e[end_slot + 1] > 1e-4:
                    end_slot += 1
                st_h, st_m = divmod(start_slot * 10, 60)
                et_h, et_m = divmod((end_slot + 1) * 10, 60)
                time_range_str = f"{st_h:02d}:{st_m:02d}-{et_h:02d}:{et_m:02d}"
                emergency_records.append({
                    '日期': d_str,
                    '紧急购电时间段': time_range_str,
                    '紧急购电量': round(np.sum(em_e[start_slot:end_slot+1]), 2)
                })
                t_idx = end_slot + 1
            else:
                t_idx += 1

    # 导出 Excel 格式
    slot_cols = [f"{h:02d}:{m:02d}" if h < 24 else "24:00" for h, m in [divmod(t * 10, 60) for t in range(1, N_SLOTS + 1)]]
    df_plan_out = pd.DataFrame(plan_buy_records).T
    df_plan_out.columns = slot_cols
    df_plan_out.index.name = '日期'

    df_tab2_out = pd.DataFrame(charge_dis_table2)
    df_emerg_out = pd.DataFrame(emergency_records)
    if df_emerg_out.empty:
        df_emerg_out = pd.DataFrame(columns=['日期', '紧急购电时间段', '紧急购电量'])

    # 保存并防占用
    final_output_path = output_excel_path
    save_done = False
    for attempt in range(2):
        try:
            with pd.ExcelWriter(final_output_path, engine='openpyxl') as writer:
                df_plan_out.round(2).to_excel(writer, sheet_name='计划购电量')
                df_tab2_out.to_excel(writer, sheet_name='充放电量', index=False)
                df_emerg_out.to_excel(writer, sheet_name='紧急购电量', index=False)
            save_done = True
            break
        except PermissionError:
            final_output_path = f"result2_{int(time.time())}.xlsx"
            print(f"⚠️ 提示: 检测到原文件被打开占用，已自动另存为: {final_output_path}")

    total_actual_cost = total_plan_cost + total_emerg_cost
    print("\n" + "=" * 80)
    print("【优化完成！系统综合费用清单】")
    print("=" * 80)
    print(f"◆ 1. 日前计划购电费用 (常规电价) : {total_plan_cost:15,.2f} 元")
    print(f"◆ 2. 实时紧急购电总量             : {total_emerg_kwh:15,.2f} kWh")
    print(f"◆ 3. 实时紧急购电罚金 (5倍惩罚)  : {total_emerg_cost:15,.2f} 元")
    print("-" * 75)
    print(f"★ 4. 【实际全年总买电费用】       : {total_actual_cost:15,.2f} 元")
    print("=" * 80)
    if save_done:
        print(f"✓ 规范结果已成功生成并保存至: {final_output_path}")

if __name__ == '__main__':
    net_load_file = '净用电量.xlsx'
    price_file = '附件1.xlsx'
    
    if os.path.exists(net_load_file) and os.path.exists(price_file):
        run_master_optimization(net_load_file, price_file, 'result2.xlsx')
    else:
        print(f"请确保当前目录下存在 '{net_load_file}' 和 '{price_file}'！")