# 2026 高教社杯 C 题 · 微网与外部电网电力调控策略

## 运行环境

- **Python 3.13**：`C:\Users\dell\AppData\Local\Programs\Python\Python313\python.exe`
- 依赖：`pulp`（COIN-OR CBC 求解器）、`pandas`、`numpy`、`openpyxl`、`matplotlib`、`scipy`
- 所有脚本开头都是 `from config import *`；**相对路径一律相对项目根目录**（即 `config.py` 所在目录），
  从任意工作目录运行都能找到附件与结果模板。

## 快速开始

```powershell
python data_analysis.py   # 问题一数据探索（输出 6 张图 + 文字报告）
python q1.py              # 问题一  → result1.xlsx
python q2.py              # 问题二  → result2.xlsx
python q3.py              # 问题三  → result3.xlsx
python q2_diag.py > q2_口径诊断.txt    # 储能执行口径对比诊断
```

各主程序都支持命令行参数来切口径（详见文件头 docstring），不带参数时写官方文件名。

## 文件地图

| 文件 | 作用 | 产出 |
|---|---|---|
| `config.py` | **公共库**：路径/常量/绘图风格/时间处理/附件读取/峰谷划分/日前预测/`simulate_dispatch` 逐槽执行 | — |
| `data_analysis.py` | 问题一数据探索 | `figures/q1/fig1~6`、`q1_数据分析结果.txt` |
| `q1.py` / `storage.py` | 问题一 MILP（两份实现，结果完全一致） | `result1.xlsx`、`storage.xlsx` |
| `test_model1.py` | 问题一对照实验（变电价 / 变负载 / 用基准日光伏） | `figures/q1/figA~C` |
| **`q2.py`** | **问题二主程序**（全年 0:00 计划制定） | `result2.xlsx`、`q2_结果汇总_*.txt` |
| `q2_tune.py` | 预测规格 × 安全裕量 两阶段寻优 | `q2_参数寻优.txt` |
| `q2_hedge.py` | 逐时段滚动分位数裕量 | `q2_裕量优化.txt` |
| `q2_multiday.py` | 逐日 LP vs 多日联合 LP | `q2_多日LP对比.txt` |
| `q2_rt.py` | 储能执行口径对比（按计划 / 实时再调度） | `q2_实时再调度对比.txt` |
| `q2_peer.py` | 与队友方案的 2×2 对标 | `q2_对标队友.txt` |
| `q2_diag.py` | 逐槽被动平衡 vs 完全信息最优 诊断 | `q2_口径诊断.txt` |
| **`q3.py`** | **问题三主程序**（0:00 / 6:00 / 12:00 / 18:00 四阶段滚动调整） | `result3.xlsx`、`q3_结果汇总.txt` |
| `q3.py ablate` | 各时刻预报的边际价值分析 | `q3_预报时刻边际价值.txt` |
| `q3_tune.py` | 混合权重 / 裕量 / 负载自适应 / 决策时刻 寻优 | `q3_参数寻优.txt` |
| `Q2_teammate_scipy.py` | 队友的另一份第二问实现（`scipy.linprog`），**仅作参考，不参与主流程** | — |
| `clean_table.py`、`光伏.xlsx`、`小区用电量.xlsx`、`净用电量.xlsx` | 论文数据表整理 | — |

> `result/` 目录里是官方给的**模板**，脚本只读不写；输出一律写到项目根目录。

## 结果汇总

| 问 | 结果文件 | 关键数字 |
|---|---|---|
| 一 | `result1.xlsx` | 购电量 **59,482.6990 kWh**；购电费 **35,126.9486 元**；0:00 = 24:00 = 6,000 kWh |
| 二 | `result2.xlsx` | 计划 20,070,215.3 kWh / 1,219.6 万元；紧急 523,825.4 kWh / 303.3 万元；**合计 1,522.8 万元** |
| 三 | `result3.xlsx` | 计划 20,116,321.3 kWh；调整后 20,299,896.1 kWh；紧急 313,625.0 kWh；**合计 1,423.6 万元** |
| 四 | `result4-2.xlsx` / `result4-3.xlsx` | **待完成**（改用附件4 的波动电价重算问题二、问题三） |

## 三条必须记住的口径

1. **时间**：第 $t$ 个时段（$t=0\ldots143$）覆盖 $[10t,\,10t+10]$ 分钟；附件1 的时间戳是区间**右端点**。
2. **附件3 的整点预报是「时刻点值」，不是小时平均值**。按点值口径插值到 10 分钟时段，全年 MAE 191.7 kW；
   若误按小时均值对齐，MAE 恶化到 361.6 kW。
3. **储能执行统一用「逐槽被动平衡」规则**（`config.simulate_dispatch`），问题二/三/四同一口径，
   保证多阶段滚动自洽、LP 可精确刻画。详见 `模型与口径说明.md`。

## 注意事项

- **写 xlsx 前先关闭 Excel**，否则报 `PermissionError`。目录里的 `~$*.xlsx` 是 Excel 的锁文件，已加入 `.gitignore`。
- 不要改动 `result/` 里的官方模板。
- `q2_*.py`、`q3_*.py` 这些分析脚本是**直接运行**的（顶层无 `if __name__` 保护），不要 `import` 它们。

## 待办

- [ ] **问题四**：用附件4 波动电价重算问题二、问题三 → `result4-2.xlsx`、`result4-3.xlsx`
- [ ] **问题二安全裕量重新寻优**：`q2_参数寻优.txt` 是**旧口径（储能按计划执行）**下跑出来的，
      当前逐槽口径下需要重跑 `q2_tune.py` / `q2_hedge.py` 再定 `HEDGE_PARAM`。
      参考：问题三区块实测 `hedge=300` 可省约 9%，问题二很可能也有类似幅度。
- [ ] **问题三参数落地**：区块寻优给出 `mix=0.4, hedge=300, adapt=0.2`，
      但 `q3.py::main()` 目前还没把 `adapt` 传进 `run_year`，且 `HEDGE` 默认仍是 0。
- [ ] 决定 `q1.py` 与 `storage.py` 是否只保留一份（两者结果相同）。
