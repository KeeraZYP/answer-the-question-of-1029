"""個人校正（事後分析），逐 epoch 版本：每個 epoch 的預測分別校正，再取最後 3 個 epoch 平均。
與表 2（各 epoch 誤差取平均）的定義一致。用法：python analysis/calib_per_epoch.py"""
import os, glob, json, importlib.util
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "results", "runs")
src = open(os.path.join(HERE, "error_structure.py")).read().split("folds = sorted")[0]
ns = {"__file__": os.path.join(HERE, "error_structure.py")}; exec(src, ns); ang, calib = ns["ang"], ns["calib"]
ns["rng"] = np.random.default_rng(0)
folds = sorted({os.path.basename(p)[:6] for p in glob.glob(f"{D}/fold*_B_hist.json")})
res = {}
for f in folds:
    res[f] = {}
    for v in "BC":
        acc = {"nocalib": [], **{f"off{k}": [] for k in (1, 3, 9, 30)}, **{f"aff{k}": [] for k in (9, 30)}}
        for p in sorted(glob.glob(f"{D}/{f}_{v}_pred_ep*.csv")):
            d = pd.read_csv(p).sort_values("idx")
            gy, gp, py, pp = d.gt_yaw.values, d.gt_pitch.values, d.pred_yaw.values, d.pred_pitch.values
            acc["nocalib"].append(ang(py, pp, gy, gp).mean())
            for k in (1, 3, 9, 30): acc[f"off{k}"].append(calib(gy, gp, py, pp, k, "offset"))
            for k in (9, 30): acc[f"aff{k}"].append(calib(gy, gp, py, pp, k, "affine"))
        res[f][v] = {k: float(np.mean(x)) for k, x in acc.items()}
summ = {}
for v in "BC":
    summ[v] = {k: float(np.mean([res[f][v][k] for f in folds])) for k in res[folds[0]][v]}
    summ[v]["improved_off9"] = int(sum(res[f][v]["off9"] < res[f][v]["nocalib"] for f in folds))
d9 = [res[f]["C"]["off9"] - res[f]["B"]["off9"] for f in folds]
summ["C_minus_B_off9"] = {"mean": float(np.mean(d9)), "p_ttest": float(stats.ttest_1samp(d9, 0).pvalue)}
json.dump({"per_fold": res, "summary": summ}, open(os.path.join(HERE, "..", "results", "calib_per_epoch.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
