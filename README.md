# L2CS-Net：yaw–pitch 分開預測的檢驗與誤差結構分析

## 研究問題

L2CS-Net 用兩個全連接層**分別**預測 yaw 與 pitch，損失也各自計算；論文沒有檢驗這個設計，也沒有做「分開 vs 聯合」的消融實驗。

1. 在損失中加入 3D 角度誤差、讓兩個角度一起被懲罰，誤差會不會下降？大角度樣本會不會改善？
2. 分開預測的模型，yaw 與 pitch 的誤差是否有結構性關聯？
3. 如果不是 yaw–pitch 關係，誤差主要從哪裡來？

## 主要結果（MPIIFaceGaze，留一人，最後 3 個 epoch 平均角度誤差，度）

| 測試者 | B：官方損失 | C：+ 3D 角度損失 | C − B | corr(e_yaw, e_pitch)（B） |
|---|---|---|---|---|
| p00 | 2.83 | 2.95 | +0.12 | +0.004 |
| p03 | 3.00 | 2.79 | −0.21 | +0.086 |
| p05 | 3.68 | 3.73 | +0.05 | +0.007 |
| p10 | 5.36 | 5.35 | −0.01 | −0.048 |
| 平均 | 3.72 | 3.70 | −0.01 | |

p07 執行中。論文 Table 1（15 fold 平均）為 3.92°；本實驗只測 4 人、訓練 12 epoch，只能說量級一致。

1. **3D 角度損失沒有一致的效果**：4 個 fold 的 C − B 分別為 +0.12、−0.21、+0.05、−0.01，平均 −0.01°。p03 的 C 在最後 3 個 epoch、4 個角度組都較好；p00 則相反。每個條件只訓練一次，無法區分這是受試者差異還是訓練本身的隨機性。
2. **yaw 與 pitch 誤差幾乎不相關**（|r| ≤ 0.09，可解釋變異 < 1%），資料本身的 yaw、pitch 也幾乎獨立（|r| < 0.04）。
3. **誤差主要來自每位受試者的固定偏移**：pitch 偏差 p00 −1.6°、p03 +1.7°、p05 +0.4°、p10 +4.9°，方向因人而異；偏差不隨角度在 3° bin 中的位置變化，所以不是官方「bin 對應左邊界」造成。
   事後分析：用測試者 9 張影像估計並扣除偏移，4 人平均 3.66° → 2.90°（p10：5.33° → 2.87°）；p05 的誤差以分散與比例壓縮（yaw 斜率 0.77）為主，校正幾乎無效。

逐樣本 Wilcoxon 檢定在不同 epoch 會翻轉方向（p00、p05、p10 各有一個 epoch 是 C 顯著較好），代表 3,000 個樣本來自同一次訓練、彼此不獨立；方法層級的重複單位是 fold。

完整數字：`results/summary.json`（成對比較、依角度分組）、`results/error_structure.json`（偏移、斜率、校正）、`docs/實驗結果整理.md`。

## 實驗設計

| 項目 | 設定 |
|---|---|
| 資料 | MPIIFaceGaze，GazeHub 前處理，官方 MPIIGaze evaluation subset（每人 3,000 筆，左右眼各 1,500，右眼樣本整張臉水平翻轉） |
| 協定 | 留一人；測試者 p00、p05、p10（事先依編號平均挑選），追加 p03、p07（組員 5 人篩選組的前兩位） |
| 模型 | 官方 `l2cs/model.py`：ResNet-50、28 bins × 3°（−42°～42°） |
| B | 官方損失：CE + α·MSE（yaw、pitch 各自），α = 1 |
| C | B + γ·L_ang，L_ang 為預測與真值 3D 視線向量夾角（度）；γ = 2·sqrt(B 最終訓練 MSE_sum) ≈ 2.5，使兩者在收斂附近對預測角度的梯度大小相同 |
| 公平性 | B、C 相同初始權重（seed 2026）、每個 epoch 相同資料順序（seed 2026 + epoch）、同一批測試影像 |
| 其餘 | 照官方 train.py：Adam lr 1e-5、batch 16、輸入 448×448、conv1/bn1 學習率 0、layer1–4 中名稱含 `bn` 的 BN 層維持 eval |
| 與官方不同 | 12 epoch（官方 50）、混合精度、從 ImageNet 權重開始（官方未提供 MPIIGaze 權重） |

## 目錄

```
preprocess/prep_mpiifacegaze.py   下載 + GazeHub 前處理（= kaggle/01）
train/train_l2cs.py               單一變體訓練，每次最多 --chunk 個 epoch，可接續
train/run_fold.py                 一個 fold：B → 決定 γ → C；分段執行、失敗自動重試
kaggle/                           Kaggle 版 notebook（01 前處理；02 各 fold 訓練）
analysis/analyze.py               B vs C 成對比較、依角度分組、yaw/pitch 誤差相關
analysis/error_structure.py       偏移、斜率、bin 位置偏差、事後校正
results/runs/                     各 fold 的 hist.json 與最後 3 epoch 逐樣本預測（00、03、05、10）
results/*.json                    分析輸出
docs/                             結果整理
```

## 如何執行

### 有 GPU 的 Linux 機器（例如國網）

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/Ahmednull/L2CS-Net.git ../L2CS-Net

# 1. 前處理（CPU，約 5–10 分鐘；需能連到 datasets.d2.mpi-inf.mpg.de）
python preprocess/prep_mpiifacegaze.py --work /tmp/mpii --out-dir ./data
mkdir -p data/extracted && tar -xf data/MPIIFaceGaze_processed.tar -C data/extracted

# 2. 一個 fold（B 與 C 依序；T4 x2 約 6 小時）
python train/run_fold.py --fold 3 --data data/extracted/MPIIFaceGaze --l2cs ../L2CS-Net --out runs/fold03

# 其餘 fold 依序跑
for f in 1 2 4 6 8 9 11 12 13 14; do
  python train/run_fold.py --fold $f --data data/extracted/MPIIFaceGaze --l2cs ../L2CS-Net --out runs/fold$(printf %02d $f)
done
```

- `--fold` 是 `Label/*.label` 排序後的索引，0 = p00、14 = p14。
- 冒煙測試：加 `--test --epochs 4 --chunk 2`（CPU、每 epoch 3 個 batch）。
- 中斷後重下同一指令即可接續；checkpoint 在其他目錄時加 `--resume-from <dir>`。

跑完把 `runs/foldXX/` 裡的 `*_hist.json` 與 `*_pred_ep*.csv` 放進 `results/runs/`，再執行：

```bash
python analysis/analyze.py
python analysis/error_structure.py
```

`.pt` checkpoint（約 300 MB）不放進 repo。

### Kaggle

1. `kaggle/01_preprocess_mpiifacegaze.ipynb`：Accelerator None、Internet On，Save & Run All；約 6 分鐘。
2. `kaggle/02_train_foldXX.ipynb`：Accelerator GPU T4 x2、Internet On，Add Input 選 01 的輸出；約 6 小時。
3. 下載 Output 中的 `*_hist.json` 與 `*_pred_ep*.csv`，放進 `results/runs/`。

## 復現時發現的官方程式碼問題

1. `train.py` MPIIGaze 分支呼叫 `Mpiigaze(..., True, fold)`，fold 被傳到 `angle` 參數位置，真正的 fold 恆為 0（組員也獨立發現）。本實驗不使用官方 train.py。
2. `train.py` 從未呼叫 `model.train()`，layer1–4 中名稱含 `bn` 的 BN 層停在 eval 模式；本實驗照原樣重現。
3. `datasets.py`、`train.py` 把標籤第 0 欄命名為 `pitch`，實際是 yaw（GazeHub `GazeTo2d` 輸出 `[yaw, pitch]`）；因為它對應 `fc_yaw_gaze`，結果正確，只是命名錯。

## 與組員實驗的差異

組員用每人**全部**標註影格（共 37,667 張）、50 epoch；本實驗用官方評估清單（每人 3,000 筆）、12 epoch。兩邊數字趨勢一致（p00 3.29 vs 2.83、p05 3.86 vs 3.68、p10 5.60 vs 5.36），但不能直接比較；組員觀察到 50 epoch 明顯過擬合，與本實驗數字較低一致。

## 限制

- 目前 4 fold、單一 seed、12 epoch；γ 未調參。
- MPIIFaceGaze 視線多在 25° 以內；大角度下 yaw–pitch 耦合可能較明顯，但官方 Gaze360 網站無法存取，未驗證。
- 校正分析是在既有預測上的事後分析，不是訓練方法；偏移來源（kappa 角、眼睛外觀、標註誤差）未區分。

## 參考資料

- Abdelrahman et al., *L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments*, arXiv:2203.03339；程式碼 https://github.com/Ahmednull/L2CS-Net
- Zhang et al., *It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation*, CVPRW 2017（MPIIFaceGaze）
- Zhang et al., *MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation*, TPAMI 2019
- Cheng et al., *Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark*, arXiv:2104.12668（GazeHub 前處理）
