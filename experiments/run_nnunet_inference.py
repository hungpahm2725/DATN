import os
import shutil
import sys
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

try:
    import numpy.core.multiarray
    torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
except Exception:
    pass

import nnunet.utilities.nd_softmax
import torch.nn.functional as F

def softmax_helper(x):
    return F.softmax(x, 1)

softmax_helper.__module__ = "nnunet.utilities.nd_softmax"
softmax_helper.__name__ = "softmax_helper"
nnunet.utilities.nd_softmax.softmax_helper = softmax_helper
# -----------------------------------------------------------------------------

from src.data.acdc_dataset import ACDCDataset



import multiprocessing.pool
import nnunet.inference.predict

# Patch Pool to ThreadPool to avoid multiprocessing pickling on Windows
nnunet.inference.predict.Pool = multiprocessing.pool.ThreadPool

# Patch preprocess_multithreaded to run in the main thread
def preprocess_multithreaded_patched(trainer, list_of_lists, output_files, num_processes=2, segs_from_prev_stage=None):
    if segs_from_prev_stage is None:
        segs_from_prev_stage = [None] * len(list_of_lists)
    
    for i, l in enumerate(list_of_lists):
        output_file = output_files[i]
        print(f"Preprocessing {output_file} (single-threaded patch)...")
        d, _, dct = trainer.preprocess_patient(l)
        yield (output_file, (d, dct))

nnunet.inference.predict.preprocess_multithreaded = preprocess_multithreaded_patched

from nnunet.inference.predict import predict_from_folder

# Cau hinh duong dan
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
NNUNET_DATA_DIR = WORKSPACE_DIR / "nnunet_data"

# Thiết lập environment variables cho nnU-Net (vẫn cần thiết vì nnU-Net đọc chúng khi import/chạy)
os.environ["nnUNet_raw_data_base"] = str(NNUNET_DATA_DIR / "nnUNet_raw_data_base")
os.environ["nnUNet_preprocessed"] = str(NNUNET_DATA_DIR / "nnUNet_preprocessed")
os.environ["RESULTS_FOLDER"] = str(NNUNET_DATA_DIR / "RESULTS_FOLDER")


def prepare_input_data(data_dir: Path, input_dir: Path, output_dir: Path, limit=None):
    """
    Copy ED and ES frames from dataset directory into nnU-Net input folder
    with naming convention patientXXX_frameYY_0000.nii.gz.
    """
    print(f"Preparing input data for nnU-Net from {data_dir}...")
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ACDCDataset(str(data_dir))
    records = dataset.load_all()

    if limit:
        records = records[:limit]
        print(f"Limiting to first {limit} patients for testing.")

    count = 0
    for r in records:
        for key in ("mri_ed_path", "mri_es_path"):
            src = Path(r[key])
            dest_name = f"{src.stem.split('.')[0]}_0000.nii.gz"
            shutil.copy2(src, input_dir / dest_name)
            count += 1

    print(f"Prepared {count} images in {input_dir}")


def run_inference(input_dir: Path, output_dir: Path, model="3d_fullres", tta=False, save_npz=True):
    print(f"Running nnU-Net inference using {model} model (TTA={tta}, save_npz={save_npz})...")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_folder = NNUNET_DATA_DIR / "RESULTS_FOLDER" / "nnUNet" / model / "Task027_ACDC" / "nnUNetTrainerV2__nnUNetPlansv2.1"
    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder not found: {model_folder}")

    print(f"Using model folder: {model_folder}")

    predict_from_folder(
        model=str(model_folder),
        input_folder=str(input_dir),
        output_folder=str(output_dir),
        folds=None,
        save_npz=save_npz,
        num_threads_preprocessing=2,
        num_threads_nifti_save=2,
        lowres_segmentations=None,
        part_id=0,
        num_parts=1,
        tta=tta,
        mixed_precision=True,
        overwrite_existing=True
    )
    print("Inference completed successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run nnU-Net inference on ACDC")
    parser.add_argument("--split", type=str, default="testing", choices=["training", "testing"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of patients to predict")
    parser.add_argument("--model", type=str, default="3d_fullres", choices=["2d", "3d_fullres"], help="nnU-Net model mode")
    parser.add_argument("--prepare", action="store_true", help="Rebuild nnU-Net input folder from raw dataset")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation (slower)")
    parser.add_argument("--save_npz", action="store_true", help="Save softmax probabilities as .npz")
    args = parser.parse_args()

    suffix = "" if args.split == "training" else "_testing"
    model_suffix = "_2d" if (args.model == "2d" and args.split == "testing") else ""
    data_dir = WORKSPACE_DIR / f"data_ACDC/{args.split}"
    input_dir = WORKSPACE_DIR / f"nnunet_input{suffix}"
    output_dir = WORKSPACE_DIR / f"nnunet_output{suffix}{model_suffix}"

    if args.prepare or not input_dir.exists():
        prepare_input_data(data_dir, input_dir, output_dir, limit=args.limit)
    run_inference(input_dir, output_dir, model=args.model, tta=args.tta, save_npz=args.save_npz)
