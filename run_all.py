# -*- coding: utf-8 -*-
"""一键复现：按依赖顺序跑完全部脚本

用法
    python run_all.py                 # 全部跑一遍（约 60~75 分钟）
    python run_all.py --list          # 只列出步骤，不执行
    python run_all.py --main          # 只跑 4 个主程序（出 result*.xlsx，约 25 分钟）
    python run_all.py --fast          # 主程序 + 快速分析（跳过 4 个耗时脚本）
    python run_all.py --only q3 q4    # 只跑名字里含 q3 / q4 的主程序
    python run_all.py --from q3_hedge_q.py   # 从指定步骤开始往后跑

设计要点
    · 步骤用 subprocess 逐个调用，任一失败不影响后续，最后统一汇总成败与耗时；
    · 主程序（q1~q4）产出交付物 `result*.xlsx`，放在最前面跑，保证交付物优先就位；
    · 重活（全年多次扫描）单独标注，可用 --fast 跳过；
    · 所有脚本都通过 config.resolve 定位附件，故从任意工作目录调用都可以。
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# (脚本, 说明, 是否重活/耗时)
STEPS = [
    # ---------- 一、交付物：四个主程序 ----------
    ("q1.py", "问题一 → result1.xlsx", False),
    ("q2.py", "问题二 → result2.xlsx", False),
    ("q3.py", "问题三 → result3.xlsx + 题目指定日期表格", False),
    ("q4.py", "问题四 → result4-2.xlsx / result4-3.xlsx", False),

    # ---------- 二、数据说明与口径核对（快，建议每次都跑） ----------
    ("data_overview.py", "四个附件的数据概览（数据说明出处）", False),
    ("data_analysis.py", "问题一数据探索", False),
    ("q3_anchor_check.py", "问题三口径递进三锚点复算", True),
    ("q4_price_effect.py", "问题四：固定 vs 波动电价受控分解", True),
    ("q3_mix_mae.py", "光伏混合权重标定（含各预报源 MAE 横向对比）", False),

    # ---------- 三、敏感性与寻优 ----------
    ("q1_sensitivity.py", "问题一敏感性", True),
    ("q2_sensitivity.py", "问题二敏感性", True),
    ("q2_fine_tune.py", "问题二裕量精细寻优（q × win）", True),
    ("q2_tune_robust.py", "问题二裕量样本外稳健性", True),
    ("q3_mix_recheck.py", "问题三 mix 最终口径复检", True),
    ("q3_mix_robust.py", "问题三 mix 分段稳健性", True),
    ("q3_mix_epoch.py", "分时段光伏权重（负面结果）", True),
    ("q3_hedge_q.py", "问题三裕量升级：区块 + 全年验证", True),
    ("q3_sensitivity.py", "问题三光伏预报受控扰动", True),
    ("q3_tune_l.py", "问题三 mix × 历史天数 网格", True),
    ("q2_multiday.py", "问题二逐日 LP vs 多日联合 LP", True),
    ("q4.py multiday", "问题四跨日套利分析", True),

    # ---------- 四、诊断与其他分析 ----------
    ("q2_diag.py", "问题二储能执行口径诊断", True),
    ("q2_rt.py", "问题二实时再调度对比", True),
    ("q2_periodicity.py", "负载周周期性分析", True),
    ("q2_pv_forecast.py", "光伏为何用 3 天滑动平均（含图）", True),
    ("q2_peer.py", "与队友方案对标", True),
    ("q3_embed_check.py", "逐槽规则嵌入 LP 的等价性验证", True),
    ("q3_hedge_err.py", "裕量误差口径对照（负载 vs 净负荷）", True),
    ("q3_perfect.py", "完美信息下界（预报误差代价上限）", True),

    # ---------- 五、文档与表格生成（放在最后，依赖前面已产出的结果） ----------
    ("make_q3_tables.py", "问题三三张表的 LaTeX（依赖 result3.xlsx）", False),
    ("make_q3_docx.py", "《问题三 模型与工作流.docx》", False),
    ("make_q4_docx.py", "《问题四 模型与工作流.docx》", False),
    ("paper_figs.py", "论文新增插图（3 张，含现场求解一天 LP）", False),
]

# --fast 时跳过的重活
HEAVY = {
    "q3_tune_l.py", "q3_mix_recheck.py", "q3_mix_robust.py", "q3_mix_epoch.py",
    "q3_hedge_q.py", "q2_multiday.py", "q4.py multiday",
    "q4_price_effect.py", "q3_anchor_check.py", "q2_pv_forecast.py",
    "q3_embed_check.py", "q3_hedge_err.py", "q3_perfect.py",
}


def show():
    print(f"{'序号':<5}{'脚本':<24}{'说明':<44}{'重活'}")
    print("-" * 82)
    for i, (s, d, h) in enumerate(STEPS, 1):
        print(f"{i:<5}{s:<24}{d:<44}{'是' if h else ''}")
    print("-" * 82)
    n_heavy = sum(1 for _s, _d, h in STEPS if _s in HEAVY)
    print(f"共 {len(STEPS)} 步，其中重活 {n_heavy} 步（--fast 会跳过）")


def main():
    argv = sys.argv[1:]
    if "--list" in argv:
        show()
        return

    todo = STEPS
    if "--main" in argv:
        todo = [t for t in STEPS if t[0] in
                ("q1.py", "q2.py", "q3.py", "q4.py")]
    elif "--fast" in argv:
        todo = [t for t in STEPS if t[0] not in HEAVY]
    if "--only" in argv:
        keys = argv[argv.index("--only") + 1:]
        todo = [t for t in todo if any(k in t[0] for k in keys)]
    if "--from" in argv:
        start = argv[argv.index("--from") + 1]
        names = [t[0] for t in todo]
        todo = todo[names.index(start):] if start in names else todo

    print("=" * 82)
    print(f"一键复现：共 {len(todo)} 步；Python = {PY}")
    print("=" * 82)

    ok, bad, skip = [], [], []
    t_all = time.time()
    for i, (cmd, desc, _heavy) in enumerate(todo, 1):
        parts = cmd.split()
        script = parts[0]
        if not os.path.exists(os.path.join(ROOT, script)):
            skip.append((cmd, "脚本不存在"))
            print(f"[{i}/{len(todo)}] 跳过 {cmd}（脚本不存在）")
            continue
        t0 = time.time()
        print(f"\n[{i}/{len(todo)}] {' '.join(parts)}  —— {desc}", flush=True)
        r = subprocess.run([PY, "-u"] + parts, cwd=ROOT)
        dt = time.time() - t0
        if r.returncode == 0:
            ok.append((cmd, dt))
            print(f"    [OK] 用时 {dt / 60:.1f} 分钟", flush=True)
        else:
            bad.append((cmd, r.returncode, dt))
            print(f"    [失败] 退出码 {r.returncode}，用时 {dt / 60:.1f} 分钟", flush=True)

    print("\n" + "=" * 82)
    print(f"汇总：成功 {len(ok)} / 失败 {len(bad)} / 跳过 {len(skip)}；"
          f"总耗时 {(time.time() - t_all) / 60:.1f} 分钟")
    print("=" * 82)
    for cmd, dt in ok:
        print(f"  OK   {cmd:<26}{dt / 60:>6.1f} 分钟")
    for cmd, rc, dt in bad:
        print(f"  FAIL {cmd:<26}退出码 {rc}，{dt / 60:.1f} 分钟")
    for cmd, why in skip:
        print(f"  SKIP {cmd:<26}{why}")
    if bad:
        print("\n提示：失败步骤请单独重跑，看完整输出定位原因。")
        sys.exit(1)


if __name__ == "__main__":
    main()
