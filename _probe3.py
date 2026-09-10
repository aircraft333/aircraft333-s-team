# -*- coding: utf-8 -*-
"""临时探针 v3：在 10 分钟粒度上标定附件3 的口径（小时均值 / 整点瞬时值）"""
from config import *

df3 = load_att3()
dates, L, G = load_att2()
D = len(dates)

FO = {r: np.full((D, 24), np.nan) for r in (0, 6, 12, 18)}
for i, d in enumerate(dates):
    for r in (0, 6, 12, 18):
        v = att3_forecast(df3, d, r)
        if v is not None:
            FO[r][i] = v

# 逐槽时间（小时）：槽 t 覆盖 [t/6, (t+1)/6]
U = (np.arange(N_SLOT) + 1) / 6.0            # 右端点小时数
KIDX = np.ceil(U).astype(int)                # 所属小时区间编号 1..24


def build_slots(rel, mode):
    """把附件3 的整点预报铺成 (D, 144) 的逐槽预报

    mode="avg"    : 预报k小时 = 区间 [H-1,H] 的平均值 -> 槽 t 取所属区间的值
    mode="point"  : 预报k小时 = H 时刻的瞬时值   -> 线性插值到槽右端点
    """
    out = np.full((D, N_SLOT), np.nan)
    hrs = list(range(0, 25))                  # 绝对时刻 0..24
    for i in range(D):
        fh = {0: 0.0, 24: 0.0}                # 0:00 / 24:00 光伏为 0
        for k in range(1, 25):
            H = rel + k
            if H <= 24:
                fh[H] = FO[rel][i, k - 1]
        if mode == "avg":
            for t in range(N_SLOT):
                K = KIDX[t]
                if K in fh:
                    out[i, t] = fh[K]
        else:
            xs = sorted(fh)
            ys = [fh[x] for x in xs]
            out[i] = np.interp(U, xs, ys)
    return out


print("--- 与附件2 实际光伏的逐槽 MAE（只用附件3 完整的日期）---")
for rel in (0, 6, 12, 18):
    for mode in ("avg", "point"):
        F = build_slots(rel, mode)
        m = ~np.isnan(F)
        print(f"  发布{rel:2d}:00  {mode:6s}  MAE={np.abs(F[m] - G[m]).mean():8.2f}"
              f"  bias={(F[m] - G[m]).mean():+8.2f}")

print("\n--- 逐步槽偏差（0:00 发布, avg 口径）---")
F = build_slots(0, "avg")
b = np.nanmean(F - G, axis=0)
print("  " + " ".join(f"{x:6.0f}" for x in b[::6]))
print("--- 逐步槽偏差（0:00 发布, point 口径）---")
F = build_slots(0, "point")
b = np.nanmean(F - G, axis=0)
print("  " + " ".join(f"{x:6.0f}" for x in b[::6]))

print("\n--- 与历史外推（逐槽 MAE）---")
for nm, F in (("ma:1", fc_ma(G, 1)), ("ma:3", fc_ma(G, 3)),
              ("ma:7", fc_ma(G, 7)), ("wday:7", fc_wday(G, 7))):
    print(f"  {nm:8s} MAE={np.abs(F - G).mean():8.2f}")
print(f"  {'附件1基准':8s} MAE="
      f"{np.abs(np.tile(att1_arrays(load_att1())['pv'], (D, 1)) - G).mean():8.2f}")

print("\n--- 组合：w*附件3(point) + (1-w)*ma:3 ---")
Fa = build_slots(0, "point")
F3 = fc_ma(G, 3)
m = ~np.isnan(Fa)
for w in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
    Fm = w * np.nan_to_num(Fa) + (1 - w) * F3
    print(f"  w={w:.1f}  MAE={np.abs(Fm[m] - G[m]).mean():8.2f}")

print("\n--- 亚小时时间平移（point 口径, 0:00 发布）---")
for s in range(-6, 7):
    Fs = np.roll(Fa, s, axis=1)
    print(f"  平移{s * 10:+3d} 分钟: MAE={np.abs(Fs[m] - G[m]).mean():8.2f}")
