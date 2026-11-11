#!/usr/bin/env python3
"""跑一個 fold 的完整對照實驗：先 B（官方損失），由 B 的最終訓練誤差決定 γ，再跑 C（+ 3D 角度損失）。

訓練程式每 --chunk 個 epoch 結束一次行程、再從 checkpoint 接續，避免記憶體洩漏累積；
子行程若被系統終止（例如記憶體不足），會自動從最後一個 checkpoint 重試。

用法（任何有 GPU 的 Linux 機器）：
  git clone --depth 1 https://github.com/Ahmednull/L2CS-Net.git /path/to/L2CS-Net
  tar -xf MPIIFaceGaze_processed.tar -C /path/to/data
  python train/run_fold.py --fold 3 --data /path/to/data/MPIIFaceGaze --l2cs /path/to/L2CS-Net --out runs/fold03
輸出：fold03_{B,C}_hist.json、fold03_{B,C}_pred_ep{10,11,12}.csv、fold03_{B,C}.pt（checkpoint，不需上傳）
"""
import argparse, json, math, os, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--l2cs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--resume-from", nargs="*", default=[])
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--chunk", type=int, default=3)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--max-attempts", type=int, default=12)
ap.add_argument("--test", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
TRAINER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_l2cs.py")


def hist_path(v):
    return os.path.join(args.out, f"fold{args.fold:02d}_{v}_hist.json")


def epochs_done(v):
    p = hist_path(v)
    return len(json.load(open(p))["history"]) if os.path.exists(p) else 0


def train(variant, gamma=0.0):
    for attempt in range(args.max_attempts):
        done = epochs_done(variant)
        if done >= args.epochs:
            break
        print(f"--- {variant}: launch trainer (done {done}/{args.epochs}, attempt {attempt + 1}) ---", flush=True)
        cmd = [sys.executable, "-u", TRAINER, variant, "--gamma", str(gamma), "--fold", str(args.fold),
               "--data", args.data, "--l2cs", args.l2cs, "--out", args.out, "--epochs", str(args.epochs),
               "--chunk", str(args.chunk), "--batch", str(args.batch), "--workers", str(args.workers)]
        if args.resume_from:
            cmd += ["--resume-from", *args.resume_from]
        if args.test:
            cmd.append("--test")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"!!! trainer exited with code {rc}（負值 -9 通常代表記憶體不足被系統終止）；從 checkpoint 重試", flush=True)
    assert epochs_done(variant) >= args.epochs, f"{variant} 未完成"
    return json.load(open(hist_path(variant)))["history"]


hist_B = train("B")
# γ：讓 γ·L_ang 與兩個 MSE 項在「B 收斂時的誤差水準」下，對預測角度的梯度大小相同。
#   ∂(MSE_yaw+MSE_pitch)/∂(ŷ, p̂) 的範數 ≈ 2·sqrt(MSE_sum)；∂L_ang/∂(ŷ, p̂) 的範數 ≈ 1（度/度）
#   → γ = 2·sqrt(B 最後一個 epoch 的訓練 MSE_sum)
gamma = 2 * math.sqrt(hist_B[-1]["train_mse"])
print(f"gamma = {gamma:.3f}  (B final train MSE_sum = {hist_B[-1]['train_mse']:.2f} deg²)", flush=True)
hist_C = train("C", gamma)
mean3 = lambda h: sum(r["test_mae"] for r in h[-3:]) / 3
print(f"final test MAE  B: {hist_B[-1]['test_mae']:.3f}°   C: {hist_C[-1]['test_mae']:.3f}°")
print(f"mean of last 3  B: {mean3(hist_B):.3f}°   C: {mean3(hist_C):.3f}°")
