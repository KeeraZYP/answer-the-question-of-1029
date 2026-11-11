#!/usr/bin/env python3
"""MPIIFaceGaze 下載與前處理（GazeHub 流程），與 kaggle/01_preprocess_mpiifacegaze.ipynb 相同。

前處理程式碼改寫自 GazeHub（Cheng et al., 2021, arXiv:2104.12668）的 data_processing_core.py
與 FaceBased data_processing_mpii.py；修改處以 # [patched] 標註。僅在 Linux 測試（multiprocessing 用 fork）。

用法：
  python preprocess/prep_mpiifacegaze.py --work /tmp/mpii --out-dir ./data
  # 已自行下載並解壓 MPIIFaceGaze（含 p00/p00.txt）時：
  python preprocess/prep_mpiifacegaze.py --raw /path/to/raw --skip-download --out-dir ./data
輸出：<out-dir>/MPIIFaceGaze_processed.tar、dataset_stats.json、samples.png、angle_hist.png
"""
import argparse, os, sys, glob, json, time, shutil, subprocess, math
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="/tmp/mpii", help="暫存目錄（下載、解壓、處理結果）")
ap.add_argument("--raw", default=None, help="原始資料目錄；預設 <work>/raw")
ap.add_argument("--out-dir", default=".", help="tar 與統計圖的輸出目錄")
ap.add_argument("--skip-download", action="store_true", help="不下載，直接使用 --raw 裡的資料")
ap.add_argument("--keep-raw", action="store_true")
args = ap.parse_args()

TEST = args.skip_download
URL_FACE = "http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip"
URL_GAZE = "http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz"
RAW   = args.raw or os.path.join(args.work, "raw")
OUT   = os.path.join(args.work, "MPIIFaceGaze")
FINAL = args.out_dir
for d in (RAW, FINAL):
    os.makedirs(d, exist_ok=True)
import requests

def check(url):
    try:
        r = requests.head(url, allow_redirects=True, timeout=60)
        size = int(r.headers.get("Content-Length", 0)) / 1e9
        print(f"{r.status_code}  {size:.2f} GB  {url}")
        return r.status_code == 200
    except Exception as e:
        print("FAIL", url, e)
        return False

if not TEST:
    ok_face = check(URL_FACE)
    ok_gaze = check(URL_GAZE)
    assert ok_face, "MPIIFaceGaze.zip 無法連線，請把這段輸出傳給 Claude"
    if not ok_gaze:
        print("⚠ MPIIGaze.tar.gz 無法連線：改用 MPIIFaceGaze 標註檔最後一欄重建 sample list（見第 3 節）")
else:
    ok_gaze = False

if not TEST:
    t = time.time()
    subprocess.run(["wget", "-c", "-q", "-O", f"{RAW}/MPIIFaceGaze.zip", URL_FACE], check=True)
    subprocess.run(["unzip", "-q", "-o", f"{RAW}/MPIIFaceGaze.zip", "-d", RAW], check=True)
    os.remove(f"{RAW}/MPIIFaceGaze.zip")
    print(f"done in {time.time()-t:.0f}s")

hits = sorted(glob.glob(f"{RAW}/**/p00/p00.txt", recursive=True))
assert hits, "找不到 p00/p00.txt，請把 RAW 目錄結構傳給 Claude"
FACE_ROOT = os.path.dirname(os.path.dirname(hits[0]))
persons = sorted(d for d in os.listdir(FACE_ROOT) if d.startswith("p") and os.path.isdir(os.path.join(FACE_ROOT, d)))
print("FACE_ROOT =", FACE_ROOT)
print(len(persons), "persons:", persons)

SAMPLE_ROOT = None
if ok_gaze:
    t = time.time()
    cmd = (f"wget -q -O - '{URL_GAZE}' | tar -xz -C '{RAW}' --wildcards "
           f"'*Evaluation Subset/sample list for eye image/*'")
    subprocess.run(cmd, shell=True, check=False)
    found = glob.glob(f"{RAW}/**/sample list for eye image", recursive=True)
    if found:
        SAMPLE_ROOT = found[0]
        print(f"official sample list: {SAMPLE_ROOT}  ({time.time()-t:.0f}s)")

if SAMPLE_ROOT is None:                      # --skip-download 時，若 raw 中已有官方清單就用它
    found = glob.glob(f"{RAW}/**/sample list for eye image", recursive=True)
    if found:
        SAMPLE_ROOT = found[0]
        print(f"official sample list (local): {SAMPLE_ROOT}")
if SAMPLE_ROOT is None:
    SAMPLE_ROOT = f"{RAW}/sample_list_rebuilt"
    os.makedirs(SAMPLE_ROOT, exist_ok=True)
    for p in persons:
        with open(f"{FACE_ROOT}/{p}/{p}.txt") as f, open(f"{SAMPLE_ROOT}/{p}.txt", "w") as g:
            for line in f:
                tok = line.strip().split(" ")
                g.write(f"{tok[0]} {tok[-1]}\n")
    SAMPLE_SOURCE = "rebuilt from MPIIFaceGaze annotation (last column)"
else:
    SAMPLE_SOURCE = "official MPIIGaze evaluation subset"
print("SAMPLE_SOURCE:", SAMPLE_SOURCE)

for p in persons:
    n = sum(1 for _ in open(f"{SAMPLE_ROOT}/{p}.txt"))
    print(p, n, end=" | ")

import cv2
import scipy.io as sio

# ---------- data_processing_core.py (GazeHub) ----------
# [patched] 移除不存在的 `import im_plot`；Euler2RotMat 未使用，刪除
class norm:
    def __init__(self, center, gazetarget, headrotvec, imsize, camparams, newfocal=960, newdistance=600):
        self.center = np.array(center)
        self.headrotvec = np.array(headrotvec)
        self.target = np.array(gazetarget)
        self.imsize = np.array(imsize)
        self.cameraparams = np.array(camparams)
        self.newfocal = newfocal
        self.newdistance = newdistance
        self.curdistance = np.linalg.norm(self.center)
        if self.headrotvec.shape == (3,):
            self.headrotvec = cv2.Rodrigues(self.headrotvec)[0]
        self._params()

    def _params(self):
        self.S_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, self.newdistance / self.curdistance]])
        xaxis = self.headrotvec[:, 0]
        z = self.center / self.curdistance
        y = np.cross(z, xaxis); y = y / np.linalg.norm(y)
        x = np.cross(y, z);     x = x / np.linalg.norm(x)
        self.R_mat = np.array([x, y, z])
        self.C_mat = np.array([[self.newfocal, 0, self.imsize[0] / 2],
                               [0, self.newfocal, self.imsize[1] / 2], [0, 0, 1]])
        self.M_mat = self.S_mat @ self.R_mat
        self.W_mat = self.C_mat @ self.M_mat @ np.linalg.inv(self.cameraparams)
        self.gaze = self.target - self.center

    def GetParams(self):
        return cv2.Rodrigues(self.R_mat)[0].flatten(), np.diagonal(self.S_mat)

    def GetImage(self, image):
        self.im = cv2.warpPerspective(image, self.W_mat, (int(self.imsize[0]), int(self.imsize[1])))
        return self.im

    def GetCoordinate(self, c):
        return (self.M_mat @ np.reshape(c, (3, 1))).flatten()

    def GetGaze(self, scale=True):
        g = (self.M_mat if scale else self.R_mat) @ self.gaze
        return g / np.linalg.norm(g)

    def GetHeadRot(self, vector=True):
        m = self.M_mat @ self.headrotvec
        return cv2.Rodrigues(m)[0].T[0] if vector else m

    def GetNewPos(self, position):
        pos = np.append(np.array(position).astype("float32"), 1)
        r = pos @ self.W_mat.T
        return np.array([r[0] / r[2], r[1] / r[2]])

    def CropEye(self, lcorner, rcorner):
        x, y = list(zip(lcorner, rcorner))
        cx, cy = np.mean(x), np.mean(y)
        width = np.abs(x[0] - x[1]) * 1.5
        height = 36 * width / 60
        x1 = [max(cx - width / 2, 0), max(cy - height / 2, 0)]
        x2 = [min(x1[0] + width, self.imsize[0]), min(x1[1] + height, self.imsize[1])]
        im = self.im[int(x1[1]):int(x2[1]), int(x1[0]):int(x2[0])]
        return cv2.resize(im, (60, 36))

def GazeTo2d(gaze):                      # 輸出 [yaw, pitch]（弧度）
    yaw = np.arctan2(-gaze[0], -gaze[2])
    pitch = np.arcsin(-gaze[1])
    return np.array([yaw, pitch])

def HeadTo2d(head):
    M = cv2.Rodrigues(head)[0]           # [patched] 移除每張都 print 矩陣的那一行
    vec = M[:, 2]
    return np.array([np.arctan2(vec[0], vec[2]), np.arcsin(vec[1])])

def GazeFlip(g):
    return np.array([-g[0], g[1], g[2]])

def HeadFlip(head):                      # 與原版行為相同（原版回傳未翻轉的 rot_mat；只影響 head 欄位，不影響 gaze 標籤）
    return cv2.Rodrigues(cv2.Rodrigues(np.array(head))[0])[0].T[0]

def EqualizeHist(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.equalizeHist(img)

# ---------- FaceBased data_processing_mpii.py (GazeHub) ----------
def AnnoDecode(a):
    a = np.array(a).astype("float32")
    return {"left_left_corner": a[2:4], "left_right_corner": a[4:6],
            "right_left_corner": a[6:8], "right_right_corner": a[8:10],
            "headrotvectors": a[14:17], "headtransvectors": a[17:20],
            "facecenter": a[20:23], "target": a[23:26]}

def process_person(person):
    im_root = os.path.join(FACE_ROOT, person)
    im_out = os.path.join(OUT, "Image", person)
    for sub in ("face", "left", "right"):
        os.makedirs(os.path.join(im_out, sub), exist_ok=True)
    os.makedirs(os.path.join(OUT, "Label"), exist_ok=True)

    camera = sio.loadmat(os.path.join(im_root, "Calibration", "Camera.mat"))["cameraMatrix"]
    with open(os.path.join(im_root, f"{person}.txt")) as f:
        anno = {l.split(" ")[0]: l.strip().split(" ")[1:-1] for l in f}
    with open(os.path.join(SAMPLE_ROOT, f"{person}.txt")) as f:
        im_list = f.readlines()

    out = open(os.path.join(OUT, "Label", f"{person}.label"), "w")
    out.write("Face Left Right Origin WhichEye 3DGaze 3DHead 2DGaze 2DHead Rmat Smat GazeOrigin\n")
    missing = 0
    for count, info in enumerate(im_list):
        im_info, which_eye = info.strip().split(" ")
        if im_info not in anno:
            missing += 1
            continue
        day, im_name = im_info.split("/")
        im = cv2.imread(os.path.join(im_root, day, im_name))
        a = AnnoDecode(anno[im_info])

        n = norm(a["facecenter"], a["target"], a["headrotvectors"], (224, 224), camera)
        im_face = n.GetImage(im)
        im_left = EqualizeHist(n.CropEye(n.GetNewPos(a["left_left_corner"]), n.GetNewPos(a["left_right_corner"])))
        im_right = EqualizeHist(n.CropEye(n.GetNewPos(a["right_left_corner"]), n.GetNewPos(a["right_right_corner"])))
        gaze = n.GetGaze(scale=True)
        head = n.GetHeadRot(vector=True)
        origin = n.GetCoordinate(a["facecenter"])
        rvec, svec = n.GetParams()

        if which_eye == "right":          # 右眼樣本：整張臉水平翻轉
            im_face = cv2.flip(im_face, 1)
            im_left, im_right = cv2.flip(im_right, 1), cv2.flip(im_left, 1)
            gaze = GazeFlip(gaze)
            head = HeadFlip(head)
            origin[0] = -origin[0]

        gaze2d, head2d = GazeTo2d(gaze), HeadTo2d(head)
        k = str(count + 1) + ".jpg"
        cv2.imwrite(os.path.join(im_out, "face", k), im_face)
        cv2.imwrite(os.path.join(im_out, "left", k), im_left)
        cv2.imwrite(os.path.join(im_out, "right", k), im_right)
        j = lambda v: ",".join(np.asarray(v).astype("str"))
        out.write(" ".join([f"{person}/face/{k}", f"{person}/left/{k}", f"{person}/right/{k}",
                            im_info, which_eye, j(gaze), j(head), j(gaze2d), j(head2d),
                            j(rvec), j(svec), j(origin)]) + "\n")
    out.close()
    return person, len(im_list), missing


from multiprocessing import Pool
t = time.time()
with Pool(min(4, os.cpu_count())) as pool:
    results = pool.map(process_person, persons)
for p, n, miss in results:
    print(f"{p}: {n} samples, {miss} missing annotation")
print(f"total {time.time()-t:.0f}s")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

stats = {"sample_source": SAMPLE_SOURCE, "persons": {}}
all_g = []
for p in persons:
    lines = open(f"{OUT}/Label/{p}.label").read().strip().split("\n")[1:]
    g = np.array([[float(x) for x in l.split(" ")[7].split(",")] for l in lines])  # [yaw, pitch] rad
    imgs_ok = all(os.path.exists(f"{OUT}/Image/{l.split(' ')[0]}") for l in lines)
    stats["persons"][p] = {"n": len(lines), "images_ok": imgs_ok}
    all_g.append(g)
g = np.degrees(np.concatenate(all_g))
within = (np.abs(g) <= 42).all(1)
stats.update({
    "total": int(len(g)),
    "yaw_deg":   {"min": float(g[:,0].min()), "max": float(g[:,0].max()), "mean": float(g[:,0].mean()), "std": float(g[:,0].std())},
    "pitch_deg": {"min": float(g[:,1].min()), "max": float(g[:,1].max()), "mean": float(g[:,1].mean()), "std": float(g[:,1].std())},
    "within_42deg": int(within.sum()), "outside_42deg": int((~within).sum()),
})
print(json.dumps(stats, indent=1))
json.dump(stats, open(f"{FINAL}/dataset_stats.json", "w"), indent=1)
assert all(v["images_ok"] for v in stats["persons"].values()), "有影像沒寫出來"

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].hist(g[:,0], bins=60); ax[0].set_title("Yaw (deg)")
ax[1].hist(g[:,1], bins=60); ax[1].set_title("Pitch (deg)")
for a in ax: a.axvline(-42, ls="--", c="gray"); a.axvline(42, ls="--", c="gray")
plt.tight_layout(); plt.savefig(f"{FINAL}/angle_hist.png", dpi=150)

# 抽 8 張畫視線箭頭（目測方向是否合理）
rng = np.random.default_rng(0)
fig, axs = plt.subplots(2, 4, figsize=(10, 5))
for a in axs.flat:
    p = persons[rng.integers(len(persons))]
    lines = open(f"{OUT}/Label/{p}.label").read().strip().split("\n")[1:]
    l = lines[rng.integers(len(lines))].split(" ")
    yaw, pitch = [float(x) for x in l[7].split(",")]
    im = cv2.cvtColor(cv2.imread(f"{OUT}/Image/{l[0]}"), cv2.COLOR_BGR2RGB)
    dx, dy = -np.sin(yaw) * np.cos(pitch), -np.sin(pitch)
    a.imshow(im); a.arrow(112, 112, 90*dx, 90*dy, color="yellow", width=2.5)
    a.set_title(f"{p} {l[4]}  y={np.degrees(yaw):.0f}° p={np.degrees(pitch):.0f}°", fontsize=8); a.axis("off")
plt.tight_layout(); plt.savefig(f"{FINAL}/samples.png", dpi=150)
print("saved figures")

t = time.time()
tar_path = f"{FINAL}/MPIIFaceGaze_processed.tar"
subprocess.run(["tar", "-cf", tar_path, "-C", os.path.dirname(OUT), os.path.basename(OUT)], check=True)
print(f"{os.path.getsize(tar_path)/1e9:.2f} GB, {time.time()-t:.0f}s")
if not TEST and not args.keep_raw:
    shutil.rmtree(RAW, ignore_errors=True)
print("✅ 完成。請確認 Output 有 MPIIFaceGaze_processed.tar、dataset_stats.json、samples.png、angle_hist.png")