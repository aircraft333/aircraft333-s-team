# -*- coding: utf-8 -*-
"""文档一致性审计：把「文档里的数字」逐一对回「报告文件」

本仓库反复出现过三类问题，这个脚本就是为了自动抓它们：

  ① 报告过时 —— 脚本改了口径后没重跑，磁盘上的报告还是旧结果；
     典型：`q4_跨日套利分析.txt`（旧版 multiday 写的结论）、
           `q2_结果汇总_wday7-ma3_h300_rt.txt`（0 字节）。
  ② 数字无出处 —— 文档里引用了某报告，但该数字其实不在那份报告里；
     典型：README 曾引「问题三 hedge=0 全年 14,236,243.7 元，见 q3_参数寻优.txt」，
           而该文件只覆盖 70 天区块，且真值是 14,152,041.6。
  ③ 数字本身写错 —— 抄写/推导时出错。
     典型：README 的「问题三省 38.9 万（2.7%）」（正确是 30.5 万 / 2.15%）。

检查项
  A. 空文件（0 字节）
  B. 报告新鲜度：产出文件比生成脚本旧 → 可能过时（config.py 改动会让全部受影响）
  C. 数字可溯源：两份文档里的大额数字（≥5 位整数 或 千分位分组）
     是否能在某个 .txt 报告里找到（按文档的小数位数比对，容忍四舍五入）

用法：python check_docs.py
输出：控制台 + 一致性审计.txt（有疑点时退出码 1，便于接进 CI）
"""
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
TXT = os.path.join(ROOT, "一致性审计.txt")

DOCS = ["README.md", "模型与口径说明.md"]

# 生成脚本 → 产出文件（用于新鲜度检查）
MAP = [
    ("q1.py", ["result1.xlsx"]),
    ("q2.py", ["result2.xlsx", "q2_结果汇总_wday7-ma3_q0.76w30_rt.txt"]),
    ("q3.py", ["result3.xlsx", "q3_题目指定日期表格.xlsx", "q3_结果汇总.txt"]),
    ("q4.py", ["result4-2.xlsx", "result4-3.xlsx", "q4_结果汇总.txt"]),
    ("q4.py", ["q4_跨日套利分析.txt"]),
    ("data_analysis.py", ["q1_数据分析结果.txt"]),
    ("q1_sensitivity.py", ["q1_敏感性分析.txt"]),
    ("q2_sensitivity.py", ["q2_敏感性分析.txt"]),
    ("q2_fine_tune.py", ["q2_裕量精细寻优.txt"]),
    ("q2_tune_robust.py", ["q2_裕量稳健性检验.txt"]),
    ("q2_tune_plot.py", ["q2_裕量寻优.txt"]),
    ("q2_diag.py", ["q2_口径诊断.txt"]),
    ("q2_rt.py", ["q2_实时再调度对比.txt"]),
    ("q2_multiday.py", ["q2_多日LP对比.txt"]),
    ("q2_peer.py", ["q2_对标队友.txt"]),
    ("q2_periodicity.py", ["q2_负载周周期性.txt"]),
    ("q3_anchor_check.py", ["q3_口径递进锚点.txt"]),
    ("q3_hedge_q.py", ["q3_裕量寻优.txt"]),
    ("q3_tune_plot.py", ["q3_裕量灵敏度.txt"]),
    ("q3_mix_recheck.py", ["q3_光伏权重复检.txt"]),
    ("q3_mix_robust.py", ["q3_光伏权重稳健性.txt"]),
    ("q3_mix_epoch.py", ["q3_分时段光伏权重.txt"]),
    ("q3_sensitivity.py", ["q3_光伏预报敏感性.txt"]),
    ("q4_price_effect.py", ["q4_电价波动效应分解.txt"]),
    ("make_q3_tables.py", ["问题三_表格.tex"]),
]

# 公共库：它一变，下游全部结论都可能失效
DEPS = ["config.py", "q2.py", "q3.py"]
TOL_MIN = 5.0        # 新鲜度容忍（分钟），避免「刚改完脚本就检查」的噪声

# 大额数字：千分位分组（可带小数）或 ≥5 位整数（可带小数），也认 LaTeX 的 {,}
NUM_RE = re.compile(r"\d{1,3}(?:\{,\}\d{3}|,\d{3})+(?:\.\d+)?|\d{5,}(?:\.\d+)?")
# LaTeX/正文里的「集合写法」，如参数网格 {0,200,500,1000,2000}：先清掉，
# 否则 "200,500,100" 会被当成一个带千分位的数字而误报。
GRID_RE = re.compile(r"\{[\d,\s]+\}")

# 物理参数 / 题面常量，本来就不该出现在结果报告里
PARAM_OK = {"6000", "1200", "10800", "12000", "5000", "9600", "2500"}

LOG = []


def log(s=""):
    try:
        print(s)
    except UnicodeEncodeError:        # Windows 管道下 cp936 遇到 U+2212 等会报错
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(s.encode(enc, "replace").decode(enc, "replace"))
    LOG.append(s)


def norm_num(s):
    """去掉千分位，得到纯数字串"""
    return s.replace("{,}", "").replace(",", "")


def scan_text(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return ""


def main():
    t0 = time.time()
    mt = lambda p: os.path.getmtime(p) if os.path.exists(p) else None  # noqa: E731

    log("=" * 78)
    log("文档一致性审计")
    log("=" * 78)
    problems = 0

    # ---------- A. 空文件 ----------
    log("\n【A】空文件（0 字节）")
    empty = [f for f in sorted(os.listdir(ROOT))
             if f.endswith((".txt", ".md", ".tex"))
             and os.path.getsize(os.path.join(ROOT, f)) == 0]
    if empty:
        problems += len(empty)
        for f in empty:
            log(f"  [!] {f}")
    else:
        log("  无")

    # ---------- B. 报告新鲜度 ----------
    log(f"\n【B】报告新鲜度（产出比自己的生成脚本旧 > {TOL_MIN:g} 分钟）")
    n_stale = 0
    for script, outs in MAP:
        sp = os.path.join(ROOT, script)
        if not os.path.exists(sp):
            continue
        ts = mt(sp)
        for o in outs:
            op = os.path.join(ROOT, o)
            if not os.path.exists(op):
                problems += 1
                log(f"  [!] 缺产出：{o}（应由 {script} 生成）")
                continue
            d = (ts - mt(op)) / 60.0
            if d > TOL_MIN:
                n_stale += 1
                log(f"  [~] {o} 比 {script} 旧 {d:.0f} 分钟（{script} 改动后未重跑）")
    if n_stale == 0:
        log("  无")
    problems += n_stale

    # 公共库：只汇总，不逐条刷屏
    dep_newer = set()
    for dep in DEPS:
        dp = os.path.join(ROOT, dep)
        if not os.path.exists(dp):
            continue
        td = mt(dp)
        for script, outs in MAP:
            sp = os.path.join(ROOT, script)
            if not os.path.exists(sp) or mt(sp) >= td:
                continue                     # 脚本本身比公共库还旧，已由上面报出
            for o in outs:
                op = os.path.join(ROOT, o)
                if os.path.exists(op) and mt(op) < td \
                        and (td - mt(op)) / 60.0 > TOL_MIN:
                    dep_newer.add(o)
    if dep_newer:
        log(f"\n  [~] {len(dep_newer)} 个报告的生成脚本早于公共库 "
            f"{'/'.join(DEPS)} 的最后修改：")
        log("      " + "、".join(sorted(dep_newer)[:10])
            + ("　…" if len(dep_newer) > 10 else ""))
        log("      若那些改动涉及口径/常量（而非仅新增工具函数），需整体重跑：")
        log("      python run_all.py --fast")
    else:
        log("\n  公共库改动未使任何报告失效")

    # ---------- C. 数字可溯源 ----------
    log("\n【C】文档中的大额数字能否在报告文件里找到")
    reports = [f for f in os.listdir(ROOT)
               if f.endswith(".txt") and f != os.path.basename(TXT)]
    corpus = " ".join(scan_text(os.path.join(ROOT, f)) for f in reports)
    rep_vals = []
    for m in NUM_RE.finditer(corpus):        # 必须先匹配再归一化：
        try:                                 # 若先去逗号，"1,237.9" 会因不足 5 位而漏掉
            rep_vals.append(float(norm_num(m.group())))
        except ValueError:
            pass
    # 报告里很多数字没有千分位（如 "5965.5"），且不足 5 位，
    # 若只用 NUM_RE 会漏掉 → 再补一遍放宽到“≥4 位整数（可带小数）”的扫描。
    RELAX = re.compile(r"\d{4,}(?:\.\d+)?")
    for m in RELAX.finditer(corpus):
        try:
            rep_vals.append(float(m.group()))
        except ValueError:
            pass
    log(f"  报告语料：{len(reports)} 个 .txt，{len(rep_vals)} 个大额数字")

    def direct(v, k):
        """v（文档值）能否直接对上某个报告数字；支持万元换算"""
        half = 0.5 * 10 ** (-k)
        for scale in (1.0, 1e4, 1e-4):
            tol = max(half * scale, 1e-6)
            for r in rep_vals:
                if abs(v * scale - r) <= tol:
                    return r
        return None

    n_missing = 0
    for doc in DOCS:
        dp = os.path.join(ROOT, doc)
        if not os.path.exists(dp):
            continue
        matched, unmatched = [], []
        for ln, line in enumerate(scan_text(dp).splitlines(), 1):
            # 先剔除 LaTeX 集合写法（如网格 {0,200,500,1000,2000}）——
            # 否则"200,500,100"会被当成一个带千分位的数字。
            line = GRID_RE.sub("{}", line)
            for m in NUM_RE.finditer(line):
                # 排除 git 短哈希之类的十六进制串（如 `519668e`）
                if m.end() < len(line) and line[m.end()] in "abcdefABCDEF":
                    continue
                s = norm_num(m.group())
                if s in PARAM_OK:                # 物理参数/题面常量，跳过
                    continue
                k = len(s.split(".")[1]) if "." in s else 0
                v = float(s)
                r = direct(v, k)
                (matched if r is not None else unmatched).append(
                    (ln, m.group(), v, k, line.strip(), r))

        # 第二轮：文档里常写「两数之差/之和」（如增量、节省额），用可溯源数字反推
        base = [x[5] for x in matched]
        still, derived = [], 0
        for ln, tok, v, k, line, _r in unmatched:
            half = 0.5 * 10 ** (-k)
            found = False
            for i in range(len(base)):
                for j in range(i + 1, len(base)):
                    a, b = base[i], base[j]
                    if abs(abs(a - b) - v) <= max(half, abs(a - b) * 1e-9) \
                            or abs(a + b - v) <= max(half, (a + b) * 1e-9):
                        found = True
                        break
                if found:
                    break
            if found:
                derived += 1
            else:
                still.append((ln, tok, line))

        log(f"\n  --- {doc} ---")
        log(f"    可溯源 {len(matched)} 个；"
            f"可由文档内其他数字相减/相加得到 {derived} 个")
        if still:
            n_missing += len(still)
            for ln, tok, line in still:
                log(f"    [!] L{ln}  {tok:>16s}   未在任何报告中找到")
                log(f"          {line[:96]}")
        else:
            log("    未发现无法溯源的数字")

    if n_missing:
        problems += n_missing
        log(f"\n  共 {n_missing} 处数字无法溯源"
            "（可能是手算值、参数或推导中间量，需人工确认）")

    # ---------- 汇总 ----------
    log("\n" + "=" * 78)
    if problems:
        log(f"审计结果：发现 {problems} 处疑点，请逐条确认")
    else:
        log("审计结果：全部通过")
    log(f"耗时 {time.time() - t0:.1f} 秒")
    log("=" * 78)

    with open(TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    print(f"\n审计结果已写入 {TXT}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
