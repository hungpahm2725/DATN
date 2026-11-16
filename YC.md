# YC.md — Hướng dẫn chạy dự án ĐATN (`CRC-FS` trên ACDC)

> **Tên đề tài**: *Định lượng độ tin cậy thể tích trên ảnh y tế 3D bằng thuật toán Conformal Risk Control trong không gian đặc trưng (`CRC-FS`)*  
> **Phạm vi nghiên cứu**: Tập trung **100% vào bộ dữ liệu ACDC** (MRI tim 4D, 5 nhóm bệnh lý: `DCM`, `HCM`, `MINF`, `NOR`, `RV`, 3 cấu trúc giải phẫu: `LV`, `RV`, `MYO` + chỉ số lâm sàng dẫn xuất `LV_EF`), bám sát đề cương [`Topic1_Conformal_Risk_Control.md`](file:///d:/Hoc_Tap/DATN/Topic1_Conformal_Risk_Control.md) và đánh bại **Angelopoulos et al. — *Conformal Risk Control* (ICLR 2024)** cũng như **COMPASS (Cheung et al., ICLR 2026)** và **Split Conformal Prediction**.

---

## 1. THỨ TỰ CHẠY FILE (CHUẨN HÓA)

Mở PowerShell tại thư mục gốc của dự án:

```powershell
cd d:\Hoc_Tap\DATN
```

### Bước 1 (Tùy chọn — Đã chạy sẵn 100%): Chạy nnU-Net Inference xuất Softmax `.npz`
> **Lưu ý**: Toàn bộ 300 file `.npz` đã được trích xuất sẵn trên ổ cứng (`nnunet_output/`, `nnunet_output_testing/`, `nnunet_output_testing_2d/`). Bạn **không cần chạy lại Bước 1** trừ khi muốn tự tái tạo từ file NIfTI gốc.

```powershell
# 1a. Xuất softmax cho 50 bệnh nhân unseen testing (100 frames) với 3d_fullres -> nnunet_output_testing/
python experiments/run_nnunet_inference.py --split testing --model 3d_fullres --save_npz

# 1b. Xuất softmax cho 100 bệnh nhân training (200 frames) với 3d_fullres -> nnunet_output/
python experiments/run_nnunet_inference.py --split training --model 3d_fullres --save_npz
```

### Bước 2: Chạy Kiểm thử Đơn vị (Unit Tests) cho mô hình `CRC-FS` (~2 giây)
Kiểm tra tính đúng đắn toán học của [`src/conformal/crc_fs.py`](file:///d:/Hoc_Tap/DATN/src/conformal/crc_fs.py) (`CRC-FS-L` và `CRC-FS-J`, bảo chứng độ phủ hữu hạn mẫu $\ge 90\%$ và kiểm soát rủi ro):

```powershell
python tests/test_crc_fs.py
```

### Bước 3: Chạy Thực nghiệm Đánh giá Toàn diện trên ACDC (~3–7 phút)
Chạy kịch bản thực nghiệm chính [`experiments/run_crc_fs_experiment.py`](file:///d:/Hoc_Tap/DATN/experiments/run_crc_fs_experiment.py) với **5-fold Cross-Validation theo từng bệnh nhân (Patient-level 5-Fold CV)** trên các cấu trúc `LV`, `RV`, `MYO` và `LV_EF`:

```powershell
# Chạy đồng thời cả tập Unseen Testing (50 BN) và Training (100 BN) với nnU-Net 3d_fullres:
python experiments/run_crc_fs_experiment.py --split all --model 3d_fullres

# Hoặc chỉ chạy riêng tập Unseen Testing (nhanh hơn, ~1.2 phút):
python experiments/run_crc_fs_experiment.py --split testing --model 3d_fullres
```

### Bước 4: Vẽ Bộ Biểu Đồ Chuẩn Bài Báo Khoa Học (~5 giây)
Chạy script [`experiments/plot_paper_figures.py`](file:///d:/Hoc_Tap/DATN/experiments/plot_paper_figures.py) để xuất bộ biểu đồ độ phân giải cao (260 DPI) theo phong cách của **Angelopoulos & Bates (2023)**, **Conformal Risk Control (ICLR 2024)** và **COMPASS (ICLR 2026)** vào thư mục `results/figures/`:

```powershell
python experiments/plot_paper_figures.py
```

---

## 2. CẤU TRÚC MÃ NGUỒN & KẾT QUẢ ĐẦU RA

### Mã nguồn cốt lõi (`src/conformal/`)
1. [`src/conformal/crc_fs.py`](file:///d:/Hoc_Tap/DATN/src/conformal/crc_fs.py): **Mô hình đề xuất `CRC-FS`** (*Conformal Risk Control in Feature Space*: **`CRC-FS-L`** và **`CRC-FS-J`**).
2. [`src/conformal/crc.py`](file:///d:/Hoc_Tap/DATN/src/conformal/crc.py): Baseline **Conformal Risk Control (Angelopoulos et al., ICLR 2024)**.
3. [`src/conformal/split_conformal.py`](file:///d:/Hoc_Tap/DATN/src/conformal/split_conformal.py): Baseline **Split Conformal Prediction**.
4. [`src/conformal/adaptive_scores.py`](file:///d:/Hoc_Tap/DATN/src/conformal/adaptive_scores.py): Baseline **Adaptive SCP**.
5. [`src/conformal/compass.py`](file:///d:/Hoc_Tap/DATN/src/conformal/compass.py) & [`src/conformal/compass_j.py`](file:///d:/Hoc_Tap/DATN/src/conformal/compass_j.py): Baseline **COMPASS-L** và **COMPASS-J** (Cheung et al., ICLR 2026).

### File kết quả tự động sinh ra (`results/`)
- [`results/tables/benchmark_summary.csv`](file:///d:/Hoc_Tap/DATN/results/tables/benchmark_summary.csv): Bảng tổng hợp các chỉ số đánh giá (`Empirical Coverage`, `Width`, `Severe Risk`, `Corr(Width, |Error|)`, `Min Group Coverage`, `Group Coverage Gap`).
- [`results/tables/per_sample_results.csv`](file:///d:/Hoc_Tap/DATN/results/tables/per_sample_results.csv): Kết quả chi tiết trên từng mẫu.
- [`results/tables/ef_results.csv`](file:///d:/Hoc_Tap/DATN/results/tables/ef_results.csv): Kết quả định lượng bất định cho Phân suất tống máu thất trái (`LV_EF`).
- [`results/figures/`](file:///d:/Hoc_Tap/DATN/results/figures): Bộ biểu đồ so sánh (`fig1` đến `fig5`, `crc_fs_pareto.png`, `crc_fs_multi_organ.png`, `crc_fs_full_comparison.png`).
