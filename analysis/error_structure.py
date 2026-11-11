"""誤差結構分析：每位測試者的固定偏移、比例（斜率）、bin 位置偏差，以及事後個人校正的效果。

用法：python analysis/error_structure.py [runs_dir] [out.json]
預測取最後 3 個 epoch 的平均（pred_ep10–12）。校正分析是事後分析，不是訓練方法：
  offset：用測試者 k 張影像估計平均 yaw/pitch 偏差並扣除，在其餘影像上重算 3D 角度誤差（隨機抽 200 次平均）
  affine：同樣 k 張影像，各角度做線性迴歸（斜率 + 截距）校正；k 太小時不穩定
"""
import sys, os, glob, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "runs")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "results", "error_structure.json")
rng = np.random.default_rng(0)


def vec(y, p):
    y, p = np.radians(y), np.radians(p)
    return np.stack([-np.cos(p) * np.sin(y), -np.sin(p), -np.cos(p) * np.cos(y)], 1)


def ang(py, pp, gy, gp):
    return np.degrees(np.arccos(np.clip((vec(py, pp) * vec(gy, gp)).sum(1), -1, 1)))


def load(fold, v):
    ds = [pd.read_csv(p).sort_values("idx").reset_index(drop=True) for p in sorted(glob.glob(f"{D}/{fold}_{v}_pred_ep*.csv"))]
    gy, gp = ds[0].gt_yaw.values, ds[0].gt_pitch.values
    return gy, gp, np.mean([d.pred_yaw for d in ds], 0), np.mean([d.pred_pitch for d in ds], 0)


def calib(gy, gp, py, pp, k, mode, reps=200):
    out = []
    for _ in range(reps):
        idx = rng.permutation(len(gy)); c, t = idx[:k], idx[k:]
        if mode == "offset":
            cy, cp = py[t] - (py - gy)[c].mean(), pp[t] - (pp - gp)[c].mean()
        else:
            sy, iy = np.polyfit(py[c], gy[c], 1); sp, ip = np.polyfit(pp[c], gp[c], 1)
            cy, cp = sy * py[t] + iy, sp * pp[t] + ip
        out.append(ang(cy, cp, gy[t], gp[t]).mean())
    return float(np.mean(out))


folds = sorted({os.path.basename(p)[:6] for p in glob.glob(f"{D}/fold*_B_hist.json")})
res = {}
for fold in folds:
    res[fold] = {}
    for v in "BC":
        gy, gp, py, pp = load(fold, v)
        r = {"mae_no_calib": float(ang(py, pp, gy, gp).mean())}
        for a, g, p in (("yaw", gy, py), ("pitch", gp, pp)):
            e = p - g
            slope, icpt = np.polyfit(g, p, 1)
            frac = ((g + 42) % 3) / 3
            r[a] = {"bias": float(e.mean()), "abs_err": float(np.abs(e).mean()), "err_sd": float(e.std()),
                    "slope_pred_on_gt": float(slope),
                    "resid_sd_after_linear": float((p - (slope * g + icpt)).std()),
                    "bias_by_bin_position": [float(e[(frac >= lo) & (frac < hi)].mean()) for lo, hi in
                                             ((0, .25), (.25, .5), (.5, .75), (.75, 1.01))]}
        r["corr_err_yaw_pitch"] = float(np.corrcoef(py - gy, pp - gp)[0, 1])
        r["offset_calib"] = {k: calib(gy, gp, py, pp, k, "offset") for k in (1, 3, 9, 30)}
        r["affine_calib"] = {k: calib(gy, gp, py, pp, k, "affine") for k in (9, 30)}
        r["offset_oracle"] = float(ang(py - (py - gy).mean(), pp - (pp - gp).mean(), gy, gp).mean())
        res[fold][v] = r

json.dump(res, open(OUT, "w"), indent=1)
print(f"{'fold':7}{'var':4}{'MAE':>7}{'yaw bias':>10}{'pitch bias':>11}{'yaw slope':>10}{'pitch slope':>12}{'corr':>7}"
      f"{'off k=9':>9}{'aff k=9':>9}{'aff k=30':>9}{'oracle':>8}")
for f, rv in res.items():
    for v, r in rv.items():
        print(f"{f:7}{v:4}{r['mae_no_calib']:7.2f}{r['yaw']['bias']:+10.2f}{r['pitch']['bias']:+11.2f}"
              f"{r['yaw']['slope_pred_on_gt']:10.2f}{r['pitch']['slope_pred_on_gt']:12.2f}{r['corr_err_yaw_pitch']:+7.3f}"
              f"{r['offset_calib'][9]:9.2f}{r['affine_calib'][9]:9.2f}{r['affine_calib'][30]:9.2f}{r['offset_oracle']:8.2f}")
for v in "BC":
    m = lambda key, k=None: np.mean([res[f][v][key] if k is None else res[f][v][key][k] for f in res])
    print(f"mean {v}: no-calib {m('mae_no_calib'):.2f}  offset k=9 {m('offset_calib', 9):.2f}  "
          f"affine k=9 {m('affine_calib', 9):.2f}  affine k=30 {m('affine_calib', 30):.2f}  oracle {m('offset_oracle'):.2f}")
