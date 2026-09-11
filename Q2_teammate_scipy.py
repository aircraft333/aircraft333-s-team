import sys
import openpyxl
import pulp

from config import *

FC_SPEC = "wday:7|ma:3"
HEDGE_MODE = "scalar"         
HEDGE_PARAM = 300.0           
HEDGE_WIN = 60                                               
OUT_START = "2025-02-01"         
TXT_Q2 = "q2_结果汇总.txt"
KEY_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]   
IDX4H = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)] 
LAB4H = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"]
IDX_REP = [60, 72, 84, 96, 108, 120]                                   
LAB_REP = ["10:00-10:10", "12:00-12:10", "14:00-14:10", "16:00-16:10", "18:00-18:10", "20:00-20:10"]

def solve_day(pi, L_plan, G_plan, E_start):
    T = N_SLOT
    m = pulp.LpProblem("q2_day", pulp.LpMinimize)
    b = [pulp.LpVariable(f"b{t}", lowBound=0) for t in range(T)]
    c = [pulp.LpVariable(f"c{t}", lowBound=0, upBound=P_RATE) for t in range(T)]
    d = [pulp.LpVariable(f"d{t}", lowBound=0, upBound=P_RATE) for t in range(T)]
    s = [pulp.LpVariable(f"s{t}", lowBound=0) for t in range(T)]
    E = [pulp.LpVariable(f"E{t}", lowBound=SOC_MIN, upBound=SOC_MAX) for t in range(T)]

    m += pulp.lpSum(pi[t] * b[t] * DT_H for t in range(T))
    for t in range(T):
        m += b[t] + G_plan[t] + d[t] - c[t] - s[t] == L_plan[t]
        prev = E_start if t == 0 else E[t - 1]
        m += E[t] == prev + (ETA * c[t] - d[t] / ETA) * DT_H

    m.solve(pulp.PULP_CBC_CMD(msg=False))
    return {k: np.array([v.value() for v in arr])
            for k, arr in (("b", b), ("c", c), ("d", d), ("s", s), ("E", E))}

def build_hedge(err, mode="quantile", param=0.85, win=60):
    D, T = err.shape
    if mode != "quantile":
        return np.full((D, T), float(param))
    X = np.zeros((D, T))
    for i in range(D):
        s = 0 if win == 0 else max(0, i - win)
        h = err[s:i]
        if h.shape[0]:
            X[i] = np.quantile(h, param, axis=0)
    return X

def solve_rt(pi, L_real, G_real, b_fixed, E_start):
    T = N_SLOT
    m = pulp.LpProblem("q2_rt", pulp.LpMinimize)
    c = [pulp.LpVariable(f"rc{t}", lowBound=0, upBound=P_RATE) for t in range(T)]
    d = [pulp.LpVariable(f"rd{t}", lowBound=0, upBound=P_RATE) for t in range(T)]
    e = [pulp.LpVariable(f"re{t}", lowBound=0) for t in range(T)]
    E = [pulp.LpVariable(f"rE{t}", lowBound=SOC_MIN, upBound=SOC_MAX) for t in range(T)]
    m += pulp.lpSum(5.0 * pi[t] * e[t] * DT_H for t in range(T))
    for t in range(T):
        m += e[t] >= L_real[t] - G_real[t] - b_fixed[t] - d[t] + c[t]
        prev = E_start if t == 0 else E[t - 1]
        m += E[t] == prev + (ETA * c[t] - d[t] / ETA) * DT_H
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    return {k: np.array([v.value() for v in arr])
            for k, arr in (("e", e), ("c", c), ("d", d), ("E", E))}

def seg_range(mask):
    segs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start * 10, i * 10))
            start = None
    if start is not None:
        segs.append((start * 10, len(mask) * 10))
    return segs

def fmt_span(a, b):
    return f"{fmt(a)}-{fmt(b)}"

def _forecast_one(X, X1, spec):
    meth, _, par = spec.partition(":")
    meth = meth.strip()
    par = par.strip()
    if meth == "att1":           
        return np.tile(X1, (X.shape[0], 1))
    if meth == "ewma":
        return fc_ewma(X, float(par) if par else 0.3, X1)
    if meth == "wday":
        return fc_wday(X, int(float(par)) if par else 7, X1)
    return fc_ma(X, max(1, int(float(par)) if par else 1), X1)     

def build_forecast(L, G, L1, G1, spec="ma:1"):
    s_l, _, s_g = spec.partition("|")
    s_l, s_g = s_l.strip(), (s_g or s_l).strip()
    if s_l == "perfect":
        return L, G
    return _forecast_one(L, L1, s_l), _forecast_one(G, G1, s_g)

def main():
    global FC_SPEC, HEDGE_MODE, HEDGE_PARAM, HEDGE_WIN, TXT_Q2
    is_default = (len(sys.argv) == 1)            
    if len(sys.argv) > 1:
        FC_SPEC = sys.argv[1]
    if len(sys.argv) > 2:              
        HEDGE_MODE, HEDGE_PARAM = "scalar", float(sys.argv[2])
    if len(sys.argv) > 3:              
        HEDGE_MODE, HEDGE_PARAM, HEDGE_WIN = "quantile", float(sys.argv[2]), int(sys.argv[3])
    tag = FC_SPEC.replace(":", "").replace("|", "-")
    tag += (f"_h{HEDGE_PARAM:g}" if HEDGE_MODE == "scalar"
            else f"_q{HEDGE_PARAM:g}w{HEDGE_WIN}")
    tag += "_rt"
    TXT_Q2 = f"q2_结果汇总_{tag}.txt"

    A1 = att1_arrays(load_att1())
    pi, L1, G1 = A1["price"], A1["load"], A1["pv"]     
    dates, L, G = load_att2()                            
    F_L, F_G = build_forecast(L, G, L1, G1, FC_SPEC)
    err = (L - G) - (F_L - F_G)                        
    X = build_hedge(err, HEDGE_MODE, HEDGE_PARAM, HEDGE_WIN)

    E_start = SOC0
    recs = []
    for i, day in enumerate(dates):
        L_plan = F_L[i] + X[i]           # 负载预测 + 逐时段安全裕量
        G_plan = F_G[i]
        plan = solve_day(pi, L_plan, G_plan, E_start)

        sim = simulate_dispatch(L[i], G[i], plan["b"], E_start)
        deficit = sim["e"]
        E_next = sim["E"][-1]
        q_ch, q_dis, E_traj = sim["c"] * DT_H, sim["d"] * DT_H, sim["E"]
      
        rec = {
            "date": day, "E0": E_start, "E24": E_next,
            "q_buy": plan["b"] * DT_H, "q_ch": q_ch, "q_dis": q_dis,
            "E": E_traj, "q_emg": deficit * DT_H,
            "cost": float(np.sum(pi * plan["b"] * DT_H)),
            "cost_emg": float(np.sum(5.0 * pi * deficit * DT_H)),
        }
        recs.append(rec)
        E_start = E_next

    out = [r for r in recs if r["date"] >= pd.Timestamp(OUT_START)]
    tot_buy = sum(r["q_buy"].sum() for r in out)
    tot_cost = sum(r["cost"] for r in out)
    tot_emg = sum(r["q_emg"].sum() for r in out)
    tot_cost_emg = sum(r["cost_emg"] for r in out)
    n_emg_days = sum(1 for r in out if r["q_emg"].sum() > 1e-6)

    for ds in KEY_DATES:
        r = next((x for x in recs if x["date"] == pd.Timestamp(ds)), None)
        if r is None:
            continue
        for j in range(0, 6, 2):
            a = f"{LAB_REP[j]:<14s} {r['q_buy'][IDX_REP[j]]:>12.2f}"
            b = f"{LAB_REP[j + 1]:<14s} {r['q_buy'][IDX_REP[j + 1]]:>12.2f}"

    write_report(TXT_Q2)

    result_path = resolve(OUT2 if is_default else f"result2_{tag}.xlsx")
    wb = openpyxl.load_workbook(resolve(TPL2))

    ws = wb["计划购电量"]
    for i, r in enumerate(out):
        row = i + 2
        for t in range(N_SLOT):
            ws.cell(row=row, column=2 + t, value=round(float(r["q_buy"][t]), 4))
        ws.cell(row=row, column=146, value=round(float(r["q_buy"].sum()), 4))
        ws.cell(row=row, column=147, value=round(float(r["cost"]), 4))

    ws = wb["充放电量"]
    for i, r in enumerate(out):
        for k, (s0, s1) in enumerate(IDX4H):
            row = 2 + i * 6 + k
            if k == 0:
                ws.cell(row=row, column=1, value=r["date"].to_pydatetime())
            ws.cell(row=row, column=2, value=LAB4H[k])
            ws.cell(row=row, column=3, value=round(float(r["q_ch"][s0:s1].sum()), 4))
            ws.cell(row=row, column=4, value=round(float(r["q_dis"][s0:s1].sum()), 4))
            if k < 2:
                ws.cell(row=row, column=5, value="00:00" if k == 0 else "24:00")
                ws.cell(row=row, column=6,
                        value=round(float(r["E0"] if k == 0 else r["E24"]), 4))

    ws = wb["紧急购电量"]
    row = 2
    for r in out:
        segs = seg_range(r["q_emg"] > 1e-6)
        if not segs:                   
            ws.cell(row=row, column=1, value=r["date"].to_pydatetime())
            row += 1
            continue
        for j, (a, b) in enumerate(segs):
            if j == 0:
                ws.cell(row=row, column=1, value=r["date"].to_pydatetime())
            ws.cell(row=row, column=2, value=fmt_span(a, b))
            ws.cell(row=row, column=3, value=round(float(r["q_emg"][a // 10:b // 10].sum()), 4))
            row += 1

    wb.save(result_path)

if __name__ == "__main__":
    main()
