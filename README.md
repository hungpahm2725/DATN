Dataset tải ở: 

[ACDC](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb)


# Định lượng Độ tin cậy Thể tích trên Ảnh Y tế 3D bằng Conformal Risk Control trong Không gian Đặc trưng (`CRC-FS`)

Hệ thống thực nghiệm và đánh giá thuật toán **CRC-FS** (`CRC-FS-L` và `CRC-FS-J`) hậu kiểm trên mô hình **3D Full-Resolution nnU-Net** cho bộ dữ liệu cộng hưởng từ tim **ACDC** (150 bệnh nhân = 300 frames 3D gồm 100 ca Training và 50 ca Unseen Testing), đối chiếu trực tiếp với `Split CP`, `CRC (ICLR 2024)`, `Adaptive SCP`, `COMPASS-L` và `COMPASS-J (ICLR 2026)`.

---

## 1. Cấu trúc thư mục dự án

```text
DATN/
├── data_ACDC/                       # Bộ dữ liệu ACDC gốc (training/ & testing/)
├── nnunet_data/                     # Thư mục chuẩn của nnU-Net v1
│   ├── nnUNet_raw_data_base/
│   ├── nnUNet_preprocessed/
│   └── RESULTS_FOLDER/nnUNet/3d_fullres/Task027_ACDC/nnUNetTrainerV2__nnUNetPlansv2.1/
├── nnunet_input/                    # Ảnh NIfTI đầu vào nnU-Net cho tập training
├── nnunet_input_testing/            # Ảnh NIfTI đầu vào nnU-Net cho tập testing
├── nnunet_output/                   # Kết quả suy luận 3D + xác suất .npz (training)
├── nnunet_output_testing/           # Kết quả suy luận 3D + xác suất .npz (testing)
├── nnunet_output_testing_2d/        # Kết quả suy luận 2D tùy chọn (testing)
├── src/
│   ├── conformal/
│   │   ├── crc_fs.py                # Thuật toán CRC-FS-L, CRC-FS-J & CRC-FS_EF
│   │   ├── crc.py                   # Conformal Risk Control (ICLR 2024)
│   │   ├── compass.py               # COMPASS-L (ICLR 2026)
│   │   ├── compass_j.py             # COMPASS-J (Logit & Deep 32-ch Jacobian)
│   │   ├── split_conformal.py       # Split Conformal Prediction
│   │   ├── adaptive_scores.py       # Adaptive SCP (Entropy / Sensitivity)
│   │   └── mondrian.py              # Mondrian Conformal Prediction
│   ├── data/acdc_dataset.py         # Nạp bệnh nhân ACDC & metadata nhóm bệnh lý
│   └── metrics/clinical_metrics.py  # Tính EDV, ESV, LV_EF (%), Myo_mass
├── experiments/
│   ├── run_nnunet_inference.py      # Chạy nnU-Net inference và xuất tensor .npz
│   ├── run_crc_fs_experiment.py     # Chạy benchmark 5-fold CV ở alpha=0.10
│   ├── generate_paper_tables.py     # Đánh giá đa ngưỡng alpha in {0.05, 0.10, 0.15}
│   └── plot_paper_figures.py        # Vẽ toàn bộ hình vẽ nghiên cứu & biểu đồ MRI
├── tests/
│   └── test_crc_fs.py               # Bộ kiểm thử đơn vị cho thuật toán CRC-FS
└── results/
    ├── crc_fs_results.csv
    ├── crc_fs_fold_summary.csv
    ├── tables/                      # benchmark_summary.csv, paper_table_multialpha.csv, ef_results.csv, per_sample_results.csv
    └── figures/                     # Các biểu đồ .png chất lượng cao
```

---

## 2. Yêu cầu hệ thống & Cài đặt môi trường

### 2.1. Yêu cầu phần cứng & phần mềm khuyến nghị
- **Hệ điều hành**: Windows 10/11 (PowerShell) hoặc Linux (Ubuntu 20.04+/22.04+).
- **Python**: `3.11.x` (đã kiểm thử ổn định trên `Python 3.11.9`).
- **GPU**: NVIDIA GPU có CUDA >= 12.x (tối thiểu ~8 GB VRAM nếu tự chạy `nnU-Net` inference; nếu đã có sẵn các file `.npz` đầu ra thì bước Conformal `CRC-FS` có thể chạy cả trên CPU lẫn GPU).

### 2.2. Tạo môi trường ảo (`venv` hoặc `conda`)

**Trên Windows (PowerShell):**
```powershell
cd D:\Hoc_Tap\DATN
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Trên Linux / macOS:**
```bash
cd /path/to/DATN
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2.3. Cài đặt thư viện Python

1. **Cài PyTorch có hỗ trợ CUDA** (hoặc bản CPU nếu máy không có GPU — xem thêm tại [pytorch.org](https://pytorch.org/get-started/locally/)):
   ```powershell
   # Ví dụ cho CUDA 12.8 (đã kiểm thử với torch 2.11.0+cu128):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
   *(Nếu chỉ chạy phần Conformal trên CPU từ `.npz` có sẵn, có thể dùng `pip install torch torchvision` bản mặc định).*

2. **Cài đặt `nnU-Net v1` và các thư viện tính toán khoa học / y tế:**
   ```powershell
   pip install nnunet==1.7.1 nibabel numpy scipy pandas scikit-learn scikit-image matplotlib pytest python-docx
   ```

---

## 3. Tải dữ liệu ACDC & Pretrained Model nnU-Net

Nếu chuyển sang máy mới chưa copy sẵn `data_ACDC/` và `nnunet_data/`:

### 3.1. Tải bộ dữ liệu ACDC
- Tải từ trang chính thức: [ACDC Challenge Database](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb).
- Giải nén sao cho cấu trúc bên trong thư mục dự án là:
  - `<PROJECT_ROOT>/data_ACDC/training/patient001/...` đến `patient100`
  - `<PROJECT_ROOT>/data_ACDC/testing/patient101/...` đến `patient150`
  - *(Lưu ý: tránh để lồng 2 cấp thư mục `data_ACDC/data_ACDC/`).*

### 3.2. Thiết lập thư mục `nnunet_data` và tải trọng số `Task027_ACDC`

Lưu ý quan trọng: Toàn bộ mã nguồn trong `experiments/` **không dùng** đường dẫn `C:\Users\T.Hung\nnunet_v1\...` như trước nữa mà trỏ về `<PROJECT_ROOT>/nnunet_data/`.

**Trên Windows (PowerShell):**
```powershell
$ROOT = (Get-Location).Path   # Chạy tại thư mục gốc DATN
$NNUNET_DIR = Join-Path $ROOT "nnunet_data"

New-Item -ItemType Directory -Force -Path `
    "$NNUNET_DIR\nnUNet_raw_data_base", `
    "$NNUNET_DIR\nnUNet_preprocessed", `
    "$NNUNET_DIR\RESULTS_FOLDER" | Out-Null

# Thiết lập biến môi trường (có thể chọn "Process" để chỉ dùng trong phiên hiện tại hoặc "User" để lưu vĩnh viễn)
[System.Environment]::SetEnvironmentVariable("nnUNet_raw_data_base", "$NNUNET_DIR\nnUNet_raw_data_base", "User")
[System.Environment]::SetEnvironmentVariable("nnUNet_preprocessed", "$NNUNET_DIR\nnUNet_preprocessed", "User")
[System.Environment]::SetEnvironmentVariable("RESULTS_FOLDER", "$NNUNET_DIR\RESULTS_FOLDER", "User")

# Áp dụng ngay cho cửa sổ PowerShell hiện tại:
$env:nnUNet_raw_data_base = "$NNUNET_DIR\nnUNet_raw_data_base"
$env:nnUNet_preprocessed  = "$NNUNET_DIR\nnUNet_preprocessed"
$env:RESULTS_FOLDER       = "$NNUNET_DIR\RESULTS_FOLDER"

# Tải và cài pretrained model Task027_ACDC (từ Zenodo)
Invoke-WebRequest -Uri "https://zenodo.org/records/4003545/files/Task027_ACDC.zip?download=1" -OutFile "$NNUNET_DIR\Task027_ACDC.zip"
nnUNet_install_pretrained_model_from_zip "$NNUNET_DIR\Task027_ACDC.zip"
```

**Trên Linux / macOS:**
```bash
export ROOT="$(pwd)"
export nnUNet_raw_data_base="$ROOT/nnunet_data/nnUNet_raw_data_base"
export nnUNet_preprocessed="$ROOT/nnunet_data/nnUNet_preprocessed"
export RESULTS_FOLDER="$ROOT/nnunet_data/RESULTS_FOLDER"
mkdir -p "$nnUNet_raw_data_base" "$nnUNet_preprocessed" "$RESULTS_FOLDER"

wget -O "$ROOT/nnunet_data/Task027_ACDC.zip" "https://zenodo.org/records/4003545/files/Task027_ACDC.zip?download=1"
nnUNet_install_pretrained_model_from_zip "$ROOT/nnunet_data/Task027_ACDC.zip"
```

---

## 4. Các đường dẫn cần lưu ý & thay đổi khi chạy trên máy khác

Tất cả script chính trong `experiments/` và `tests/` đều tự động xác định thư mục gốc bằng `WORKSPACE_DIR = Path(__file__).resolve().parent.parent`, nên bạn **không cần sửa đường dẫn** nếu giữ đúng cấu trúc mặc định bên trong thư mục dự án (`data_ACDC/`, `nnunet_data/`, `nnunet_output*/`, `results/`).

Tuy nhiên, nếu trên máy mới bạn đặt dữ liệu hoặc trọng số ở một ổ đĩa khác (ví dụ `E:\Datasets\ACDC` để tiết kiệm dung lượng ổ C/D), hãy cập nhật tại **các vị trí chính xác dưới đây** (hoặc dùng symlink/junction từ thư mục gốc):

| Thành phần cần đổi | Vị trí chính xác trong mã nguồn | Mặc định hiện tại | Cách xử lý trên máy khác |
|---|---|---|---|
| **Biến môi trường nnU-Net v1** | Biến môi trường OS + `experiments/run_nnunet_inference.py` (dòng `60–65`) | `WORKSPACE_DIR / "nnunet_data"` | Sửa `NNUNET_DATA_DIR` nếu lưu trọng số nnU-Net ở ổ đĩa khác |
| **Checkpoint lớp cuối `3d_fullres` (cho `CRC-FS-J`)** | `experiments/run_crc_fs_experiment.py` trong `load_head_weight()` (dòng `71–76`) | `WORKSPACE_DIR / "nnunet_data/RESULTS_FOLDER/nnUNet" / model / "Task027_ACDC/..."` | Đổi phần đầu đường dẫn nếu `RESULTS_FOLDER` nằm ngoài repo |
| **Thư mục dataset ACDC (`training` / `testing`)** | • `experiments/run_nnunet_inference.py` (dòng `138`)<br>• `experiments/run_crc_fs_experiment.py` (dòng `669, 671, 673`)<br>• `experiments/generate_paper_tables.py` (dòng `533`)<br>• `experiments/plot_paper_figures.py` (dòng `176`) | `WORKSPACE_DIR / "data_ACDC/testing"` và `WORKSPACE_DIR / "data_ACDC/training"` | Trỏ tới nơi chứa `patient001..150` trên máy mới |
| **Thư mục chứa `.npz` / `.nii.gz` sau inference** | • `experiments/run_nnunet_inference.py` (dòng `139–140`)<br>• `experiments/run_crc_fs_experiment.py` (dòng `669, 671, 673`)<br>• `experiments/generate_paper_tables.py` (dòng `534`)<br>• `experiments/plot_paper_figures.py` (dòng `177`) | `nnunet_output_testing/`, `nnunet_output/`, `nnunet_output_testing_2d/` | Sửa nếu bạn để file `.npz` dung lượng lớn ở phân vùng khác |
| **Thư mục xuất CSV & hình ảnh** | • `experiments/run_crc_fs_experiment.py` (dòng `659–661`)<br>• `experiments/generate_paper_tables.py` (dòng `524–525`)<br>• `experiments/plot_paper_figures.py` (dòng `807–809`) | `WORKSPACE_DIR / "results"` | Chỉ cần đổi nếu muốn xuất báo cáo ra thư mục riêng |

> **Mẹo nhanh (không cần sửa code):** Trên Windows, nếu bạn đặt `data_ACDC` hoặc `nnunet_output_testing` ở ổ khác (ví dụ `E:\ACDC_cache\...`), chỉ cần mở CMD/PowerShell quyền Admin tạo Directory Junction:
> ```cmd
> mklink /J D:\NewPath\DATN\data_ACDC E:\Datasets\data_ACDC
> mklink /J D:\NewPath\DATN\nnunet_output_testing E:\ACDC_cache\nnunet_output_testing
> ```

---

## 5. Hướng dẫn chạy toàn bộ quy trình (Pipeline)

Đứng tại thư mục gốc của dự án (`DATN/`) và kích hoạt `.venv` trước khi chạy các bước dưới đây:

### Bước 1: Chạy Unit Tests kiểm tra thuật toán `CRC-FS`
```powershell
python tests/test_crc_fs.py
# Hoặc dùng pytest:
pytest tests/test_crc_fs.py -v
```
Kỳ vọng kết quả cuối cùng: `ALL UNIT TESTS PASSED!`.

### Bước 2: Chạy suy luận `nnU-Net` để sinh mặt nạ & xác suất Softmax `.npz`
*(Bỏ qua bước này nếu bạn đã copy sẵn 2 thư mục `nnunet_output_testing/` và `nnunet_output/` chứa các file `.npz` và `.nii.gz`)*.

```powershell
# Bắt buộc kèm --save_npz để lưu tensor xác suất Softmax 4D phục vụ không gian đặc trưng
python experiments/run_nnunet_inference.py --split testing --model 3d_fullres --prepare --save_npz
python experiments/run_nnunet_inference.py --split training --model 3d_fullres --prepare --save_npz
```
- **Lưu ý trên máy mới**: Nếu muốn chạy thử trước với vài ca bệnh để kiểm tra GPU, dùng thêm `--limit 5`. Không cần bật `--tta` nếu muốn chạy nhanh hơn (nhưng chạy chuẩn benchmark không TTA/hoặc đồng nhất với cấu hình đã dùng).

### Bước 3: Chạy Benchmark chính (`alpha = 0.10`, 5-Fold Patient-Clustered CV)
```powershell
# Chạy đầy đủ cả tập testing_3d + training_3d cho cả LV, RV, MYO và LV_EF (%)
python experiments/run_crc_fs_experiment.py --split all --model 3d_fullres --alpha 0.10
```
Các tùy chọn hữu ích khác:
- Chỉ chạy nhanh trên tập kiểm thử độc lập (`50` bệnh nhân): `--split testing`
- Chỉ chạy một cơ quan cụ thể: `--organs LV` (hoặc `RV`, `MYO`)
- Bỏ qua tính `LV_EF (%)`: `--skip_ef`

Kết quả tự động lưu vào:
- `results/tables/benchmark_summary.csv` (bảng tổng hợp Coverage, Mean/Median Width, `Corr(Width, |Error|)`, Min Group Coverage, Wilcoxon *p-value*)
- `results/tables/per_sample_results.csv` (dự đoán chi tiết từng frame)
- `results/tables/ef_results.csv` (kết quả trên phân suất tống máu `LV_EF %`)

### Bước 4: Sinh bảng tổng hợp đa ngưỡng `alpha ∈ {0.05, 0.10, 0.15}` & Ablation
```powershell
python experiments/generate_paper_tables.py
```
Sinh các tệp tại `results/tables/` và `results/figures/`:
- `paper_table_multialpha.csv` (Mean ± STD qua 5 folds cho `alpha = 0.05, 0.10, 0.15`)
- `paper_table1_coverage_light.png`, `paper_table2_width_light.png`, `paper_table3_ablation_light.png` (kèm bản nền tối trong `results/figures/`).
- Nếu đã có sẵn `paper_table_multialpha.csv` và chỉ muốn vẽ lại ảnh bảng nhanh:
  ```powershell
  python experiments/generate_paper_tables.py --plot-only
  ```

### Bước 5: Vẽ bộ hình ảnh trực quan hóa khoa học
*(Yêu cầu đã hoàn thành Bước 3 để có các tệp CSV trong `results/tables/`)*
```powershell
python experiments/plot_paper_figures.py
```
Tạo đầy đủ các biểu đồ trong `results/figures/`:
- `crc_fs_multi_organ.png`, `crc_fs_pareto.png`, `crc_fs_full_comparison.png`
- `fig1_crc_risk_curve_theorem21.png` (đường cong kiểm soát rủi ro $R_n(\lambda)$)
- `fig2_compass_mri_mask_perturbation.png` (trực quan hóa lát cắt MRI thực `DCM`, `HCM`, `MINF`)
- `fig3_sorted_prediction_intervals.png` (Ribbon Plot)
- `fig4_adaptivity_and_width_distribution.png` (tương quan thích ứng & phân phối Violin)
- `fig5_pathology_group_conditional.png` (độ phủ theo 5 nhóm bệnh lý & `LV_EF %`)

---

## 6. Xử lý một số lỗi thường gặp khi chuyển sang máy mới

1. **`torch.load` lỗi `WeightsUnpicklingError` trên PyTorch >= 2.6**:
   - Đã được xử lý sẵn trong [`experiments/run_nnunet_inference.py`](experiments/run_nnunet_inference.py) (`weights_only=False`) và [`experiments/run_crc_fs_experiment.py`](experiments/run_crc_fs_experiment.py).
2. **Lỗi multiprocessing khi inference `nnU-Net` trên Windows**:
   - Đã vá bằng `ThreadPool` và `preprocess_multithreaded_patched` trong [`experiments/run_nnunet_inference.py`](experiments/run_nnunet_inference.py). Nếu lỗi `Can't pickle` vẫn xuất hiện ở một môi trường khác, kiểm tra chạy lại bằng Python 3.11.
3. **Thông báo `[SKIP] Missing .npz predictions in ...`**:
   - Do chưa có file xác suất `.npz` trong `nnunet_output_testing/` hoặc `nnunet_output/`. Hãy chạy lại Bước 2 với cờ `--save_npz`.
4. **Lỗi thiếu font `Times New Roman` hoặc `DejaVu Serif` khi vẽ hình trên Linux**:
   - Trên Ubuntu/Debian, cài `sudo apt-get install fonts-dejavu msttcorefonts` (hoặc script `matplotlib` sẽ tự fallback về `DejaVu Sans`).
