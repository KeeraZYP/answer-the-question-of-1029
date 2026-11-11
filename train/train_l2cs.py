#!/usr/bin/env python3
"""L2CS-Net（MPIIFaceGaze，留一人）單一變體訓練程式：每次呼叫最多訓練 --chunk 個 epoch 後結束（釋放記憶體），
再次呼叫會從 checkpoint 接續。一般用 run_fold.py 呼叫，不需直接執行。

與 kaggle/02_train_foldXX.ipynb 內嵌的訓練程式邏輯相同，只把環境變數改成命令列參數。
  B：官方損失  CE + alpha*MSE（yaw、pitch 各自）
  C：B + gamma * 3D 角度損失
"""
import os, sys, glob, json, time, math, shutil, gc, argparse, importlib.util
import numpy as np, torch, torch.nn as nn, torchvision, psutil
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("variant", choices=["B", "C"])
ap.add_argument("--gamma", type=float, default=0.0)
ap.add_argument("--fold", type=int, required=True, help="測試者索引 0–14（依 Label/*.label 排序）")
ap.add_argument("--data", required=True, help="前處理後的 MPIIFaceGaze 目錄（含 Image/ 與 Label/）")
ap.add_argument("--l2cs", required=True, help="官方 L2CS-Net repo 目錄（只讀 l2cs/model.py）")
ap.add_argument("--out", required=True, help="checkpoint、紀錄、預測輸出目錄")
ap.add_argument("--resume-from", nargs="*", default=[], help="其他可找 checkpoint 的目錄（例如前一次的輸出）")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--chunk", type=int, default=3)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--test", action="store_true", help="冒煙測試：CPU、不載入 ImageNet 權重、每 epoch 3 個 batch")
args = ap.parse_args()

VARIANT, GAMMA, CHUNK = args.variant, args.gamma, args.chunk
TEST   = args.test
FOLD   = args.fold
EPOCHS = args.epochs
BATCH  = 4 if TEST else args.batch
LR, ALPHA, SEED = 1e-5, 1.0, 2026
NUM_BINS, BIN_W, ANG_MIN = 28, 3, -42
WORK  = args.out
DATA  = args.data
os.makedirs(WORK, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() and not TEST else "cpu"
NGPU = torch.cuda.device_count() if DEV == "cuda" else 0
NAME = f"fold{FOLD:02d}_{VARIANT}"

_spec = importlib.util.spec_from_file_location("l2cs_model", os.path.join(args.l2cs, "l2cs", "model.py"))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
L2CS = _mod.L2CS                                  # 官方模型類別

# ---------- 資料 ----------
# 標籤第 8 欄 2DGaze（弧度）；依 GazeHub GazeTo2d，第 0 個是 yaw、第 1 個是 pitch。
# （官方 datasets.py 把第 0 個命名為 pitch，但它對應模型第一個輸出 fc_yaw_gaze，實際學的是 yaw。）
class MPIIFace(Dataset):
    def __init__(self, label_files, root, transform):
        self.root, self.transform, self.items = root, transform, []
        for lf in label_files:
            for line in open(lf).read().strip().split("\n")[1:]:
                t = line.split(" ")
                yaw, pitch = np.degrees([float(x) for x in t[7].split(",")])
                if abs(yaw) <= 42 and abs(pitch) <= 42:              # 與官方相同的過濾
                    self.items.append((t[0], float(yaw), float(pitch)))
        self.bins = np.array(range(-42, 42, 3))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, yaw, pitch = self.items[i]
        with Image.open(os.path.join(self.root, path)) as im:
            img = self.transform(im.convert("RGB"))
        b = np.digitize([yaw, pitch], self.bins) - 1                   # 與官方相同的分箱
        return img, torch.tensor(b, dtype=torch.long), torch.tensor([yaw, pitch], dtype=torch.float32), i

tf = torchvision.transforms.Compose([
    torchvision.transforms.Resize(448), torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
labels = sorted(glob.glob(f"{DATA}/Label/*.label"))
train_ds = MPIIFace([f for k, f in enumerate(labels) if k != FOLD], f"{DATA}/Image", tf)
test_ds  = MPIIFace([labels[FOLD]], f"{DATA}/Image", tf)

def make_loader(ds, shuffle, epoch=0):
    g = torch.Generator(); g.manual_seed(SEED + epoch)                # 每個 epoch 的順序由 SEED+epoch 決定
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, generator=g if shuffle else None,
                      num_workers=0 if TEST else (args.workers if shuffle else max(1, args.workers // 2)),
                      pin_memory=False, persistent_workers=False)

# ---------- 模型、最佳化器（照官方 train.py） ----------
def build_model():
    torch.manual_seed(SEED)                                          # B、C 相同初始權重
    m = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], NUM_BINS)
    if not TEST:
        url = "https://download.pytorch.org/models/resnet50-19c8e357.pth"   # 官方 getArch_weights 的網址
        try:
            sd = torch.hub.load_state_dict_from_url(url, progress=False)
        except Exception as e:
            print("official URL failed, fallback to torchvision IMAGENET1K_V1:", e)
            sd = torchvision.models.resnet50(weights="IMAGENET1K_V1").state_dict()
        md = m.state_dict(); md.update({k: v for k, v in sd.items() if k in md}); m.load_state_dict(md)
    return m.to(DEV).to(memory_format=torch.channels_last)

def official_bn_mode(m):
    # 官方 train.py 從未呼叫 model.train()；建立 optimizer 時 get_*_params 把
    # layer1–4 中名稱含 'bn' 的子模組設為 eval。這裡重現該狀態。
    m.train()
    for layer in (m.layer1, m.layer2, m.layer3, m.layer4):
        for name, mod in layer.named_modules():
            if "bn" in name:
                mod.eval()

def build_optimizer(m):
    ignored  = [p for mod in (m.conv1, m.bn1, m.fc_finetune) for p in mod.parameters()]
    backbone = [p for mod in (m.layer1, m.layer2, m.layer3, m.layer4) for p in mod.parameters()]
    fc       = [p for mod in (m.fc_yaw_gaze, m.fc_pitch_gaze) for p in mod.parameters()]
    return torch.optim.Adam([{"params": ignored, "lr": 0}, {"params": backbone, "lr": LR},
                             {"params": fc, "lr": LR}], LR)

IDX = torch.arange(NUM_BINS, dtype=torch.float32, device=DEV)
ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()

def expect_deg(logits):                          # 官方：sum(softmax * idx) * 3 - 42
    return (torch.softmax(logits, 1) * IDX).sum(1) * BIN_W + ANG_MIN

def to3d(yaw_deg, pitch_deg):                    # 與 l2cs.utils.gazeto3d 相同（輸入 [yaw, pitch]）
    y, p = torch.deg2rad(yaw_deg), torch.deg2rad(pitch_deg)
    return torch.stack([-torch.cos(p) * torch.sin(y), -torch.sin(p), -torch.cos(p) * torch.cos(y)], 1)

def angular_deg(py, pp, gy, gp):                 # 3D 夾角（度），逐樣本
    c = (to3d(py, pp) * to3d(gy, gp)).sum(1).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.rad2deg(torch.acos(c))

def losses(out_yaw, out_pitch, bins, cont):
    out_yaw, out_pitch = out_yaw.float(), out_pitch.float()
    py, pp = expect_deg(out_yaw), expect_deg(out_pitch)
    l_yaw   = ce(out_yaw,   bins[:, 0]) + ALPHA * mse(py, cont[:, 0])
    l_pitch = ce(out_pitch, bins[:, 1]) + ALPHA * mse(pp, cont[:, 1])
    l_ang = angular_deg(py, pp, cont[:, 0], cont[:, 1]).mean()
    return l_yaw, l_pitch, l_ang, mse(py, cont[:, 0]) + mse(pp, cont[:, 1])

@torch.no_grad()
def evaluate(m, keep=False):
    m.eval(); errs, rows = [], []
    for img, bins, cont, idx in make_loader(test_ds, False):
        img, cont = img.to(DEV).to(memory_format=torch.channels_last), cont.to(DEV)
        with torch.autocast("cuda", enabled=DEV == "cuda"):
            oy, op = m(img)
        py, pp = expect_deg(oy.float()), expect_deg(op.float())
        e = angular_deg(py, pp, cont[:, 0], cont[:, 1])
        errs.append(e.cpu())
        if keep:
            rows += np.column_stack([idx.numpy(), cont.cpu().numpy(), py.cpu().numpy(),
                                     pp.cpu().numpy(), e.cpu().numpy()]).tolist()
    official_bn_mode(m)
    return float(torch.cat(errs).mean()), rows

def mem_report():
    vm = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    return f"RAM avail {vm.available/1e9:.1f}/{vm.total/1e9:.1f} GB, trainer RSS {rss/1e9:.1f} GB"

# ---------- checkpoint ----------
def find_ckpt():
    best, best_ep = None, -1
    for root in [WORK] + args.resume_from:
        for h in glob.glob(f"{root}/**/{NAME}.pt", recursive=True):
            ep = torch.load(h, map_location="cpu", weights_only=False)["epoch"]
            if ep > best_ep: best, best_ep = h, ep
    return best

def save_hist(gamma, hist):
    json.dump({"fold": FOLD, "test_person": os.path.basename(labels[FOLD]), "variant": VARIANT,
               "gamma": gamma, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "alpha": ALPHA, "seed": SEED,
               "gpus": [torch.cuda.get_device_name(i) for i in range(NGPU)], "history": hist},
              open(f"{WORK}/{NAME}_hist.json", "w"), indent=1)

# ---------- 主程式 ----------
m = build_model(); opt = build_optimizer(m); official_bn_mode(m)
net = nn.DataParallel(m) if NGPU > 1 else m
scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
hist, start, gamma = [], 0, GAMMA
ck = find_ckpt()
if ck:
    s = torch.load(ck, map_location=DEV, weights_only=False)
    m.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"])
    hist, start, gamma = s["hist"], s["epoch"] + 1, s["gamma"]
    del s
    print(f"resume {NAME} from epoch {start + 1} ({ck})", flush=True)

for ep in range(start, min(EPOCHS, start + CHUNK)):
    t0, agg, n = time.time(), np.zeros(5), 0
    for k, (img, bins, cont, _) in enumerate(make_loader(train_ds, True, ep)):
        img = img.to(DEV, non_blocking=True).to(memory_format=torch.channels_last)
        bins, cont = bins.to(DEV), cont.to(DEV)
        with torch.autocast("cuda", enabled=DEV == "cuda"):
            oy, op = net(img)
        l_yaw, l_pitch, l_ang, l_mse = losses(oy, op, bins, cont)
        loss = l_yaw + l_pitch + (gamma * l_ang if VARIANT == "C" else 0.0)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        agg += [l_yaw.item(), l_pitch.item(), l_ang.item(), loss.item(), l_mse.item()]; n += 1
        if k == 199 and ep == start:
            per = (time.time() - t0) / 200
            print(f"  speed: {per*1000:.0f} ms/iter → ~{per*len(train_ds)/BATCH/60:.1f} min/epoch", flush=True)
        if TEST and k >= 2: break
    keep = ep >= EPOCHS - 3                                           # 最後 3 個 epoch 存逐樣本預測
    mae, rows = evaluate(m, keep=keep)
    agg /= n
    hist.append({"epoch": ep + 1, "loss_yaw": agg[0], "loss_pitch": agg[1], "train_ang": agg[2],
                 "loss_total": agg[3], "train_mse": agg[4], "test_mae": mae, "sec": time.time() - t0})
    print(f"[{VARIANT}] ep {ep+1:2d}/{EPOCHS}  L_yaw {agg[0]:.3f}  L_pitch {agg[1]:.3f}  "
          f"train_ang {agg[2]:.2f}°  test MAE {mae:.3f}°  ({time.time()-t0:.0f}s)   {mem_report()}", flush=True)
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "hist": hist, "epoch": ep, "gamma": gamma}, f"{WORK}/{NAME}.pt.tmp")
    os.replace(f"{WORK}/{NAME}.pt.tmp", f"{WORK}/{NAME}.pt")          # 寫完才換名，避免中途被砍留下壞檔
    if keep:
        np.savetxt(f"{WORK}/{NAME}_pred_ep{ep+1:02d}.csv", np.array(rows), delimiter=",", fmt="%.6f",
                   header="idx,gt_yaw,gt_pitch,pred_yaw,pred_pitch,ang_err", comments="")
    save_hist(gamma, hist)
    gc.collect()
save_hist(gamma, hist)
