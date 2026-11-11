"""Paired analysis of B (original loss) vs C (+3D angular loss). Usage: python analysis/analyze.py [runs_dir] [summary.json]"""
import sys, glob, json, os
import numpy as np, pandas as pd
from scipy import stats

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "runs")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "summary.json")
folds = sorted({os.path.basename(p)[:6] for p in glob.glob(f"{D}/fold*_B_hist.json")})
rng = np.random.default_rng(0)
out = {}

def load(fold, v):
    eps = sorted(glob.glob(f"{D}/{fold}_{v}_pred_ep*.csv"))
    dfs = [pd.read_csv(p).sort_values("idx").reset_index(drop=True) for p in eps]
    return dfs, [int(p[-6:-4]) for p in eps]

for fold in folds:
    B, epB = load(fold, "B"); C, epC = load(fold, "C")
    assert epB == epC and all((b.idx.values == c.idx.values).all() for b, c in zip(B, C))
    gt = B[0][["gt_yaw", "gt_pitch"]].values
    eB = np.mean([d.ang_err.values for d in B], 0)          # 每個樣本：最後 3 epoch 平均誤差
    eC = np.mean([d.ang_err.values for d in C], 0)
    diff = eC - eB                                          # 負值 = C 較好
    w = stats.wilcoxon(eC, eB)
    boot = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
    r = {"n": len(diff), "epochs": epB,
         "mae_B_per_epoch": [float(d.ang_err.mean()) for d in B],
         "mae_C_per_epoch": [float(d.ang_err.mean()) for d in C],
         "mae_B_last3": float(eB.mean()), "mae_C_last3": float(eC.mean()),
         "mean_diff_C_minus_B": float(diff.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
         "wilcoxon_p": float(w.pvalue), "frac_C_better": float((diff < 0).mean())}
    # 每個 epoch 各自的成對檢定（看結論是否隨 epoch 翻轉）
    r["per_epoch"] = [{"epoch": e, "B": float(b.ang_err.mean()), "C": float(c.ang_err.mean()),
                       "p": float(stats.wilcoxon(c.ang_err, b.ang_err).pvalue)} for e, b, c in zip(epB, B, C)]
    # 依視線偏離正前方的角度分組（3D 夾角，相對 yaw=pitch=0）
    mag = np.degrees(np.arccos(np.cos(np.radians(gt[:, 0])) * np.cos(np.radians(gt[:, 1]))))
    q = np.quantile(mag, [0, .25, .5, .75, 1])
    r["by_gaze_magnitude"] = []
    for i in range(4):
        m = (mag >= q[i]) & (mag <= q[i + 1]) if i == 3 else (mag >= q[i]) & (mag < q[i + 1])
        r["by_gaze_magnitude"].append({"range_deg": [float(q[i]), float(q[i + 1])], "n": int(m.sum()),
            "B": float(eB[m].mean()), "C": float(eC[m].mean()), "p": float(stats.wilcoxon(eC[m], eB[m]).pvalue)})
    # yaw / pitch 分量誤差，以及兩分量誤差的相關（回應「未建模 yaw–pitch 關係」）
    for v, dfs in (("B", B), ("C", C)):
        py = np.mean([d.pred_yaw.values for d in dfs], 0); pp = np.mean([d.pred_pitch.values for d in dfs], 0)
        ey, ep_ = py - gt[:, 0], pp - gt[:, 1]
        r[f"{v}_abs_err_yaw"] = float(np.abs(ey).mean()); r[f"{v}_abs_err_pitch"] = float(np.abs(ep_).mean())
        r[f"{v}_bias_yaw"] = float(ey.mean()); r[f"{v}_bias_pitch"] = float(ep_.mean())
        r[f"{v}_corr_err_yaw_pitch"] = float(stats.pearsonr(ey, ep_)[0])
    r["corr_gt_yaw_pitch"] = float(stats.pearsonr(gt[:, 0], gt[:, 1])[0])
    hB = json.load(open(f"{D}/{fold}_B_hist.json")); hC = json.load(open(f"{D}/{fold}_C_hist.json"))
    r["gamma"] = hC["gamma"]
    r["test_mae_curve_B"] = [h["test_mae"] for h in hB["history"]]
    r["test_mae_curve_C"] = [h["test_mae"] for h in hC["history"]]
    out[fold] = r

json.dump(out, open(OUT, "w"), indent=1)
for f, r in out.items():
    print(f"== {f}  gamma={r['gamma']:.3f}  n={r['n']}")
    print(f"  last-3 MAE  B {r['mae_B_last3']:.3f}  C {r['mae_C_last3']:.3f}  diff {r['mean_diff_C_minus_B']:+.3f} "
          f"CI95 [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  Wilcoxon p={r['wilcoxon_p']:.3g}  C better on {r['frac_C_better']*100:.1f}%")
    for e in r["per_epoch"]: print(f"  ep{e['epoch']}: B {e['B']:.3f}  C {e['C']:.3f}  p={e['p']:.3g}")
    for g in r["by_gaze_magnitude"]: print(f"  gaze {g['range_deg'][0]:5.1f}–{g['range_deg'][1]:5.1f}° (n={g['n']}): B {g['B']:.3f}  C {g['C']:.3f}  p={g['p']:.3g}")
    for v in "BC":
        print(f"  {v}: |e_yaw| {r[v+'_abs_err_yaw']:.3f}  |e_pitch| {r[v+'_abs_err_pitch']:.3f}  bias yaw {r[v+'_bias_yaw']:+.3f} pitch {r[v+'_bias_pitch']:+.3f}  corr(e_yaw,e_pitch) {r[v+'_corr_err_yaw_pitch']:+.3f}")
    print(f"  corr(gt_yaw, gt_pitch) {r['corr_gt_yaw_pitch']:+.3f}")
