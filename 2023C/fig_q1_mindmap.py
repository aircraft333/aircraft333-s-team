# -*- coding: utf-8 -*-
"""2023C 问题一 分析思路 思维导图（结构树）→ figures_2023C_q1/q1_思路导图.png"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figures_2023C_q1")
os.makedirs(FIG, exist_ok=True)

# 思维导图数据：键=节点文本(用\n换行)，值=子节点；叶子为空 dict
ROOT_TEXT = "2023C 问题一：品类/单品销量分布规律与关联分析"
TREE = {
    "① 数据准备": {
        "附件1：251个单品\n归属 6 大品类": {},
        "附件2：销售流水\n87.8万条 (2020.07~2023.06)": {},
        "剔除退货461条\n(销量≤0 无效数据)": {},
        "按单品编码\n合并品类名称": {},
    },
    "② 品类销量分布规律": {
        "三年总销量/占比\n花叶类占 42% 居首": {},
        "品种丰富度\n与销量结构一致": {},
        "日销量描述统计\n均值/中位数/偏度/峰度": {},
        "时间规律\n月度时令 + 周末峰值": {},
    },
    "③ 单品销量分布规律": {
        "Top20 头部单品\n(芜湖青椒/西兰花等)": {},
        "帕累托：前20%单品\n贡献 84.4% 销量": {},
        "长尾明显\n175 个滞销单品": {},
    },
    "④ 关联关系分析": {
        "品类间\nSpearman 日销量相关": {},
        "除茄类外\n均显著正相关": {},
        "单品间 K-means 分档\n(总销量/日最大/日均)": {},
        "热销6·畅销19\n平销46·滞销175": {},
    },
    "⑤ 结论与商超建议": {
        "花叶类为稳定基本盘\n重点保供": {},
        "按品类时令备货\n夏叶菜/秋冬菌菇/夏茄": {},
        "周末补货量上浮 30%~40%": {},
    },
}

# 配色：每大类的主题色
PALETTE = {
    "① 数据准备": "#3B6FB5",
    "② 品类销量分布规律": "#2E8B57",
    "③ 单品销量分布规律": "#E08A1E",
    "④ 关联关系分析": "#7A5BA6",
    "⑤ 结论与商超建议": "#C0392B",
}
ROOT_FC, ROOT_EC, ROOT_TC = "#2C3E50", "#2C3E50", "white"
LINE_EC = "#8A8A8A"

# 排版参数
PAD_X, PAD_Y = 0.16, 0.10          # 节点内边距(英寸)
LH = 1.5                            # 行高倍数(相对字号)
FS = {"root": 13, "l1": 10.5, "leaf": 8.6}
GAP_LEAF = 0.22                     # 叶节点纵向间距
GAP_MID = 0.6                       # 大类与其叶子之间的间距
GAP_ROOT = 1.0                      # 根与大类的间距
GAP_COL = 0.55                      # 大类的横向列间距
MARGIN_X, MARGIN_T, MARGIN_B = 0.7, 0.5, 0.5


def _units(s):
    """显示宽度：中文/全角≈2，半角≈1"""
    return sum(2 if ord(ch) > 0x2E00 else 1 for ch in s)


def wrap(text, maxu=32):
    """按显示宽度折行（保留显式 \\n）"""
    out = []
    for raw in text.split("\n"):
        line = raw.strip()
        cur, curw = "", 0
        for ch in line:
            w = 2 if ord(ch) > 0x2E00 else 1
            if cur and curw + w > maxu:
                out.append(cur)
                cur, curw = "", 0
            cur += ch
            curw += w
        if cur:
            out.append(cur)
    return out


def measure(lines, fs):
    em = fs / 72.0
    w = max((_units(ln) for ln in lines), default=1) * em * 0.5 + 2 * PAD_X
    h = len(lines) * em * LH + 2 * PAD_Y
    return w, h


def draw_node(ax, xc, yc, w, h, lines, fc, ec, tc, fs, bold=False):
    p = FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                       boxstyle="round,pad=0.012,rounding_size=0.10",
                       linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(p)
    ax.text(xc, yc, "\n".join(lines), ha="center", va="center",
            fontsize=fs, color=tc, weight="bold" if bold else "normal",
            linespacing=1.3, zorder=4)


def mix(hexc, alpha):
    """hex 颜色叠白得到浅色"""
    from matplotlib.colors import to_rgb, to_hex
    r, g, b = to_rgb(hexc)
    return to_hex(((1 - alpha) + alpha * r, (1 - alpha) + alpha * g,
                   (1 - alpha) + alpha * b))


def main():
    root_lines = wrap(ROOT_TEXT, maxu=44)
    root_w, root_h = measure(root_lines, FS["root"])

    # ---- 预计算各列几何 ----
    cols = []  # 每项: name, xc, w_col, l1(lines,w,h,top_yc), leaves[(lines,w,h,yc)]
    for name, kids in TREE.items():
        l1_lines = wrap(name, maxu=22)
        w1, h1 = measure(l1_lines, FS["l1"])
        leaves = []
        for ktxt in kids:
            kl = wrap(ktxt, maxu=30)
            wl, hl = measure(kl, FS["leaf"])
            leaves.append({"lines": kl, "w": wl, "h": hl})
        w_col = max([w1] + [l["w"] for l in leaves]) + 0.15
        cols.append({"name": name, "w_col": w_col, "l1_lines": l1_lines,
                     "w1": w1, "h1": h1, "leaves": leaves})

    n_col = len(cols)
    sum_w = sum(c["w_col"] for c in cols) + GAP_COL * (n_col - 1)
    W = max(root_w, sum_w) + 2 * MARGIN_X

    # 垂直：根在上，各列对齐到同一顶部线
    col_top = MARGIN_T + root_h + GAP_ROOT
    max_col_h = 0.0
    for c in cols:
        hh = c["h1"] + GAP_MID + sum(l["h"] for l in c["leaves"]) \
            + GAP_LEAF * (len(c["leaves"]) - 1)
        c["col_h"] = hh
        max_col_h = max(max_col_h, hh)
    H = col_top + max_col_h + MARGIN_B

    # 横向分配列位置
    x = MARGIN_X
    for c in cols:
        c["xc"] = x + c["w_col"] / 2
        x += c["w_col"] + GAP_COL

    # 逐列纵向布置（从顶部往下）
    for c in cols:
        y = col_top
        c["l1_yc"] = y - c["h1"] / 2
        y -= c["h1"] + GAP_MID
        for lf in c["leaves"]:
            lf["yc"] = y - lf["h"] / 2
            y -= lf["h"] + GAP_LEAF
        c["spine_bottom"] = y + GAP_LEAF  # 最后一片叶子的底
        c["spine_top"] = c["l1_yc"] + c["h1"] / 2

    root_xc = W / 2
    root_yc = H - MARGIN_T - root_h / 2

    # ---- 绘制 ----
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    draw_node(ax, root_xc, root_yc, root_w, root_h, root_lines,
              ROOT_FC, ROOT_EC, ROOT_TC, FS["root"], bold=True)

    # 根 → 各大类 连线
    for c in cols:
        ax.plot([root_xc, c["xc"]], [root_yc - root_h / 2, c["l1_yc"] + c["h1"] / 2],
                color=LINE_EC, lw=1.2, zorder=1, solid_capstyle="round")

    for c in cols:
        base = PALETTE[c["name"]]
        # 大类节点
        draw_node(ax, c["xc"], c["l1_yc"], c["w1"], c["h1"], c["l1_lines"],
                  base, base, "white", FS["l1"], bold=True)
        # 纵向主干
        ax.plot([c["xc"], c["xc"]], [c["spine_bottom"], c["spine_top"]],
                color=base, lw=1.6, zorder=1, alpha=0.75)
        # 叶节点
        for lf in c["leaves"]:
            draw_node(ax, c["xc"], lf["yc"], lf["w"], lf["h"], lf["lines"],
                      mix(base, 0.16), base, "#222222", FS["leaf"])
            ax.plot([c["xc"], c["xc"]], [lf["yc"] - lf["h"] / 2, lf["yc"] + lf["h"] / 2],
                    color=base, lw=1.6, zorder=1, alpha=0.75)

    out = os.path.join(FIG, "q1_思路导图.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("已保存:", out)


if __name__ == "__main__":
    main()
