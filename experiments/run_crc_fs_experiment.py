"""
experiments/run_crc_fs_experiment.py

Comprehensive ACDC Benchmark for Conformal Risk Control on Clinical Metrics
(aligned with Topic1_Conformal_Risk_Control.md):
  1. Empirical Coverage vs. 90% target
  2. Interval Width / Efficiency (vs. Split CP, CRC ICLR 2024, COMPASS ICLR 2026)
  3. Group-Conditional Coverage & Coverage Gap across ACDC pathologies (DCM, HCM, MINF, NOR, RV)
  4. Correlation between Interval Width and Case Difficulty / Error
"""

import argparse
import sys
from pathlib import Path
from time import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_DIR))

from src.conformal.adaptive_scores import (
    calibrate_normalized,
    compute_uncertainty_from_probs,
    predict_interval_normalized,
)
from src.conformal.compass import (
    calibrate_compass,
    compass_l_score,
    predict_interval_compass,
)
from src.conformal.compass_j import (
    calibrate_compass_j,
    compass_j_score,
    compute_hidden_feature_subspace,
    compute_multiclass_logit_jacobian,
    compute_shared_directions,
    hidden_feature_direction_for_sample,
    predict_interval_compass_j,
)
from src.conformal.crc import find_lambda
from src.conformal.crc_fs import (
    bounded_severity_risk,
    calibrate_and_predict_crc_fs,
    calibrate_and_predict_ef_crc_fs,
    calibrate_crc_fs_l_from_scores,
    extract_directional_features,
    extract_ef_directional_features,
    predict_interval_crc_fs_l,
)
from src.conformal.split_conformal import (
    calibrate,
    conformal_quantile,
    predict_interval,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ORGAN_CLASSES = {"LV": 3, "RV": 1, "MYO": 2}


def load_head_weight(model: str = "3d_fullres") -> np.ndarray | None:
    """Load averaged final 1x1 segmentation head weight W (C, D_h) from nnU-Net folds."""
    base = (
        WORKSPACE_DIR
        / "nnunet_data/RESULTS_FOLDER/nnUNet"
        / model
        / "Task027_ACDC/nnUNetTrainerV2__nnUNetPlansv2.1"
    )
    weights = []
    for f_idx in range(5):
        ckpt_p = base / f"fold_{f_idx}/model_final_checkpoint.model"
        if not ckpt_p.exists():
            continue
        ckpt = torch.load(str(ckpt_p), map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        seg_keys = sorted(
            [k for k in sd if k.startswith("seg_outputs.") and k.endswith(".weight")],
            key=lambda k: int(k.split(".")[1]),
        )
        if seg_keys:
            w = sd[seg_keys[-1]].squeeze().numpy()
            if w.ndim == 2:
                weights.append(w)
    if not weights:
        return None
    return np.mean(weights, axis=0)


def parse_info_cfg(cfg_path: Path) -> dict:
    info = {}
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    info[k.strip()] = v.strip()
    return info


def load_samples(data_dir: Path, pred_dir: Path, target_class: int, head_weight: np.ndarray | None = None) -> list:
    """Load softmax .npz + GT .nii.gz pairs and compute directional feature proxies."""
    npz_files = sorted(pred_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {pred_dir}")

    samples = []
    for npz_path in npz_files:
        parts = npz_path.stem.split("_")
        if len(parts) < 2:
            continue
        pid, fid = parts[0], parts[1]
        gt_path = data_dir / pid / f"{pid}_{fid}_gt.nii.gz"
        if not gt_path.exists():
            continue

        info = parse_info_cfg(data_dir / pid / "Info.cfg")
        group = info.get("Group", "UNK")
        ed_frame = f"frame{int(info.get('ED', 1)):02d}"
        es_frame = f"frame{int(info.get('ES', 1)):02d}"

        probs = np.load(str(npz_path))["softmax"].astype(np.float32)
        gt_img = nib.load(str(gt_path))
        gt_mask = gt_img.get_fdata()
        sp = gt_img.header.get_zooms()
        voxel_vol = float(sp[0] * sp[1] * sp[2]) / 1000.0
        y_true = float(np.sum(gt_mask == target_class)) * voxel_vol

        feats = extract_directional_features(
            probs, target_class, voxel_vol, dz=float(sp[2]), head_weight=head_weight
        )
        ent = compute_uncertainty_from_probs(probs, target_class, method="entropy")
        sigma_jac = 1.0 / max(feats["sens"], 1e-8)

        probs_t = torch.tensor(probs, dtype=torch.float32, device=DEVICE)
        mfn = lambda m, v=voxel_vol: float(m.sum().item()) * v
        sc_l = compass_l_score(probs_t, y_true, target_class, mfn, b_max=15.0, n_iter=14)

        J_z = compute_multiclass_logit_jacobian(probs_t, target_class, voxel_vol)
        dir_j = J_z / max(float(J_z.abs().max().item()), 1e-8)

        samples.append({
            "patient_id": pid,
            "frame": fid,
            "group": group,
            "is_ed": int(fid == ed_frame),
            "ed_frame": ed_frame,
            "es_frame": es_frame,
            "y_true": y_true,
            "probs": probs_t,
            "voxel_vol": voxel_vol,
            "metric_fn": mfn,
            "entropy": ent,
            "sigma_jac": sigma_jac,
            "sc_l": sc_l,
            "dir_j": dir_j,
            **feats,
        })
    return samples


def evaluate_organ_kfold(
    samples: list,
    target_class: int,
    organ_name: str,
    split_name: str,
    head_weight: np.ndarray | None,
    alpha: float = 0.1,
    n_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run patient-level 5-fold CV comparing all baselines and DA-CRC-FS."""
    pids = sorted({s["patient_id"] for s in samples})
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    methods = [
        "scp_soft",
        "scp_hard",
        "crc",
        "ascp_ent",
        "ascp_sens",
        "compass_l",
        "compass_j",
        "compass_j_32ch",
        "crc_fs_legacy",
        "dacrc_oof",
        "dacrc_cv_plus",
        "dacrc_manifold",
        "dacrc_mondrian",
        "dacrc_risk_crc",
    ]
    if head_weight is None:
        methods.remove("compass_j_32ch")

    per_sample_rows = []
    fold_rows = []

    for fold_i, (tr_p, te_p) in enumerate(kf.split(pids)):
        tr_set = {pids[i] for i in tr_p}
        cal = [s for s in samples if s["patient_id"] in tr_set]
        tst = [s for s in samples if s["patient_id"] not in tr_set]

        cy = np.array([s["y_true"] for s in cal])
        ty = np.array([s["y_true"] for s in tst])
        cyh = np.array([s["y_hard"] for s in cal])
        tyh = np.array([s["y_hard"] for s in tst])
        cys = np.array([s["y_soft"] for s in cal])
        tys = np.array([s["y_soft"] for s in tst])
        tau_scale = float(np.std(np.abs(cy - cyh))) + 1e-6
        # Patient-cluster finite-sample correction (2 correlated frames ED & ES per patient)
        alpha_cluster = max(alpha - 1.5 / max(len(tr_p), 1), 0.05)

        bounds = {}

        # 1. Split CP (soft legacy + hard fair)
        scp_s = calibrate(cy, cys, alpha_cluster, mode="abs")
        bounds["scp_soft"] = predict_interval(tys, scp_s["q_hat"], mode="abs")

        scp_h = calibrate(cy, cyh, alpha_cluster, mode="abs")
        bounds["scp_hard"] = predict_interval(tyh, scp_h["q_hat"], mode="abs")

        # 2. Conformal Risk Control (Angelopoulos et al., ICLR 2024)
        crc_r = find_lambda(cy, cyh, alpha_cluster, mode="abs", n_grid=2000)
        bounds["crc"] = predict_interval(tyh, crc_r["lambda"], mode="abs")

        # 3. Adaptive Split CP (entropy & Jacobian sensitivity)
        c_ent = np.array([s["entropy"] for s in cal])
        t_ent = np.array([s["entropy"] for s in tst])
        ascp_e = calibrate_normalized(cy, cyh, c_ent, alpha_cluster)
        bounds["ascp_ent"] = predict_interval_normalized(tyh, t_ent, ascp_e["q_hat"])

        c_sens = np.array([s["sens"] for s in cal])
        t_sens = np.array([s["sens"] for s in tst])
        ascp_s = calibrate_normalized(cy, cyh, c_sens, alpha_cluster)
        bounds["ascp_sens"] = predict_interval_normalized(tyh, t_sens, ascp_s["q_hat"])

        # 4. COMPASS-L (Cheung et al., ICLR 2026)
        sc_l_cal = np.array([s["sc_l"] for s in cal])
        beta_l = calibrate_compass(sc_l_cal, alpha_cluster)
        lo_l, hi_l = zip(*[
            predict_interval_compass(s["probs"], beta_l, target_class, s["metric_fn"])
            for s in tst
        ])
        bounds["compass_l"] = (np.array(lo_l), np.array(hi_l))

        # 5. COMPASS-J (multi-class logit Jacobian PCA)
        V_L_j, cal_dirs_j = compute_shared_directions(
            [s["probs"] for s in cal],
            target_class,
            [s["voxel_vol"] for s in cal],
            n_components=4,
        )
        V_L_jt = torch.tensor(V_L_j, dtype=torch.float32, device=DEVICE)
        P_c = V_L_jt.T @ V_L_jt
        sc_j_cal = np.array([
            compass_j_score(
                s["probs"], s["y_true"], target_class, s["metric_fn"],
                cal_dirs_j[i], b_max=15.0, n_iter=14,
            )
            for i, s in enumerate(cal)
        ])
        beta_j = calibrate_compass_j(sc_j_cal, alpha_cluster)
        lo_j, hi_j = [], []
        for s in tst:
            J_z = compute_multiclass_logit_jacobian(s["probs"], target_class, s["voxel_vol"])
            d_t = (P_c @ J_z.reshape(J_z.shape[0], -1)).reshape(J_z.shape)
            amax = d_t.abs().max()
            if amax > 1e-8:
                d_t = d_t / amax
            lo_v, hi_v = predict_interval_compass_j(
                s["probs"], beta_j, target_class, s["metric_fn"], d_t
            )
            lo_j.append(lo_v)
            hi_j.append(hi_v)
        bounds["compass_j"] = (np.array(lo_j), np.array(hi_j))

        # 6. COMPASS-J 32ch (true intermediate-feature PCA via seg_outputs[-1] weight W)
        if head_weight is not None:
            V_L_h, cal_dirs_h = compute_hidden_feature_subspace(
                [s["probs"] for s in cal],
                target_class,
                [s["voxel_vol"] for s in cal],
                head_weight,
                n_components=5,
            )
            sc_h_cal = np.array([
                compass_j_score(
                    s["probs"], s["y_true"], target_class, s["metric_fn"],
                    cal_dirs_h[i], b_max=15.0, n_iter=14,
                )
                for i, s in enumerate(cal)
            ])
            beta_h = calibrate_compass_j(sc_h_cal, alpha_cluster)
            lo_h, hi_h = zip(*[
                predict_interval_compass_j(
                    s["probs"], beta_h, target_class, s["metric_fn"],
                    hidden_feature_direction_for_sample(
                        s["probs"], target_class, s["voxel_vol"], head_weight, V_L_h
                    ),
                )
                for s in tst
            ])
            bounds["compass_j_32ch"] = (np.array(lo_h), np.array(hi_h))

        # 7. Legacy CRC-FS-L
        c_sig_jac = np.array([s["sigma_jac"] for s in cal])
        cfs_leg = calibrate_crc_fs_l_from_scores(sc_l_cal, c_sig_jac, alpha, mode="coverage")
        lo_leg, hi_leg = zip(*[
            predict_interval_crc_fs_l(
                s["probs"], cfs_leg["lambda_hat"], s["sigma_jac"], 0.0,
                target_class, s["metric_fn"],
            )
            for s in tst
        ])
        bounds["crc_fs_legacy"] = (np.array(lo_leg), np.array(hi_leg))

        # 8. CRC-FS family (Ours: CRC-FS-L & CRC-FS-J)
        r_oof = calibrate_and_predict_crc_fs(
            cal, tst, alpha=alpha, mode="crc_fs_j", target_class=target_class,
            project_manifold=False, seed=fold_i,
        )
        bounds["dacrc_oof"] = (r_oof["lower"], r_oof["upper"])

        r_cvp = calibrate_and_predict_crc_fs(
            cal, tst, alpha=alpha, mode="crc_fs_l", target_class=target_class,
            project_manifold=False, seed=fold_i,
        )
        bounds["dacrc_cv_plus"] = (r_cvp["lower"], r_cvp["upper"])

        r_man = calibrate_and_predict_crc_fs(
            cal, tst, alpha=alpha, mode="crc_fs_j", target_class=target_class,
            project_manifold=True, seed=fold_i,
        )
        bounds["dacrc_manifold"] = (r_man["lower"], r_man["upper"])

        r_mon = calibrate_and_predict_crc_fs(
            cal, tst, alpha=alpha, mode="mondrian_shrink", target_class=target_class,
            project_manifold=False, seed=fold_i,
        )
        bounds["dacrc_mondrian"] = (r_mon["lower"], r_mon["upper"])

        r_risk = calibrate_and_predict_crc_fs(
            cal, tst, alpha=alpha, mode="risk_crc", target_class=target_class,
            project_manifold=False, seed=fold_i,
        )
        bounds["dacrc_risk_crc"] = (r_risk["lower"], r_risk["upper"])

        fold_rec = {"dataset": split_name, "organ": organ_name, "fold": fold_i + 1}
        for m in methods:
            lo_m, hi_m = bounds[m]
            lo_m = np.maximum(lo_m, 0.0)
            hi_m = np.maximum(hi_m, lo_m)
            cov_m = float(np.mean((ty >= lo_m) & (ty <= hi_m)))
            wid_m = float(np.mean(hi_m - lo_m))
            sev_m = bounded_severity_risk(ty, lo_m, hi_m, tau_scale)
            fold_rec[f"{m}_cov"] = cov_m
            fold_rec[f"{m}_wid"] = wid_m
            fold_rec[f"{m}_sev"] = sev_m
        fold_rows.append(fold_rec)

        for i, s in enumerate(tst):
            row = {
                "dataset": split_name,
                "organ": organ_name,
                "fold": fold_i + 1,
                "patient_id": s["patient_id"],
                "frame": s["frame"],
                "group": s["group"],
                "y_true": s["y_true"],
                "y_pred_hard": s["y_hard"],
                "y_pred_soft": s["y_soft"],
                "abs_err_hard": abs(s["y_true"] - s["y_hard"]),
                "tau_scale": tau_scale,
            }
            for m in methods:
                lo_v = float(max(bounds[m][0][i], 0.0))
                hi_v = float(max(bounds[m][1][i], lo_v))
                row[f"{m}_covered"] = int(lo_v <= s["y_true"] <= hi_v)
                row[f"{m}_width"] = hi_v - lo_v
                row[f"{m}_miss_dist"] = max(lo_v - s["y_true"], s["y_true"] - hi_v, 0.0)
            per_sample_rows.append(row)

    return pd.DataFrame(per_sample_rows), pd.DataFrame(fold_rows)


def evaluate_ef_kfold(
    samples_lv: list,
    split_name: str,
    alpha: float = 0.1,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Evaluate LV Ejection Fraction (%) across patients."""
    pat_map = {}
    for s in samples_lv:
        pid = s["patient_id"]
        pat_map.setdefault(pid, {"group": s["group"]})
        if s["frame"] == s["ed_frame"]:
            pat_map[pid]["ed"] = s
        elif s["frame"] == s["es_frame"]:
            pat_map[pid]["es"] = s

    ef_samples = []
    for pid, d in sorted(pat_map.items()):
        if "ed" not in d or "es" not in d:
            continue
        ed, es = d["ed"], d["es"]
        ef_true = float(np.clip((ed["y_true"] - es["y_true"]) / max(ed["y_true"], 1e-6) * 100.0, 0, 100))
        ef_feats = extract_ef_directional_features(ed, es)

        edv_h = max(float(ed["y_hard"]), 1.0)
        esv_h = max(float(es["y_hard"]), 0.0)
        dedv = 100.0 * esv_h / (edv_h ** 2)
        desv = 100.0 / edv_h
        sig_compass = max(float(np.hypot(dedv * float(ed["sens"]), desv * float(es["sens"]))), 0.05)
        sc_ef_l = abs(ef_true - float(ef_feats["y_hard"])) / sig_compass

        ef_samples.append({
            "patient_id": pid,
            "group": d["group"],
            "y_true": ef_true,
            "sc_ef_l": sc_ef_l,
            "sig_compass": sig_compass,
            **ef_feats,
        })

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rows = []
    for fold_i, (tr_i, te_i) in enumerate(kf.split(ef_samples)):
        cal = [ef_samples[i] for i in tr_i]
        tst = [ef_samples[i] for i in te_i]
        cy = np.array([s["y_true"] for s in cal])
        ty = np.array([s["y_true"] for s in tst])
        cyh = np.array([s["y_hard"] for s in cal])
        tyh = np.array([s["y_hard"] for s in tst])

        # Cluster-safe alpha adjustments on EF
        alpha_ef_crc = max(alpha - 1.5 / max(len(cal), 1), 0.05)
        qh = conformal_quantile(np.abs(cy - cyh), alpha)
        lo_s, hi_s = np.clip(tyh - qh, 0, 100), np.clip(tyh + qh, 0, 100)

        crc_ef = find_lambda(cy, cyh, alpha_ef_crc, mode="abs", n_grid=2000)
        lo_crc, hi_crc = np.clip(tyh - crc_ef["lambda"], 0, 100), np.clip(tyh + crc_ef["lambda"], 0, 100)

        # Normalize COMPASS-L on EF cleanly across scales (sensitivity in mL -> %EF units)
        sc_cal_ef = np.array([s["sc_ef_l"] for s in cal])
        bl = conformal_quantile(sc_cal_ef, alpha)
        sig_c_tst = np.array([s["sig_compass"] for s in tst])
        lo_c2, hi_c2 = np.clip(tyh - bl * sig_c_tst, 0, 100), np.clip(tyh + bl * sig_c_tst, 0, 100)

        r_ef = calibrate_and_predict_ef_crc_fs(cal, tst, alpha=alpha, seed=fold_i)
        lo_d, hi_d = r_ef["lower"], r_ef["upper"]

        for j, s in enumerate(tst):
            rows.append({
                "dataset": split_name,
                "patient_id": s["patient_id"],
                "group": s["group"],
                "y_true": s["y_true"],
                "y_pred_hard": s["y_hard"],
                "scp_hard_covered": int(lo_s[j] <= s["y_true"] <= hi_s[j]),
                "scp_hard_width": float(hi_s[j] - lo_s[j]),
                "crc_covered": int(lo_crc[j] <= s["y_true"] <= hi_crc[j]),
                "crc_width": float(hi_crc[j] - lo_crc[j]),
                "compass_l_covered": int(lo_c2[j] <= s["y_true"] <= hi_c2[j]),
                "compass_l_width": float(hi_c2[j] - lo_c2[j]),
                "dacrc_ef_covered": int(lo_d[j] <= s["y_true"] <= hi_d[j]),
                "dacrc_ef_width": float(hi_d[j] - lo_d[j]),
            })
    return pd.DataFrame(rows)


def summarize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics per (dataset, organ, method)."""
    base_methods = [
        "scp_soft", "scp_hard", "crc", "ascp_ent", "ascp_sens",
        "compass_l", "compass_j", "compass_j_32ch", "crc_fs_legacy",
        "dacrc_oof", "dacrc_cv_plus", "dacrc_manifold",
        "dacrc_mondrian", "dacrc_risk_crc",
    ]
    summary_rows = []
    for (ds, org), gdf in df.groupby(["dataset", "organ"], sort=False):
        ref_crc = gdf["crc_width"].values
        ref_cl = gdf["compass_l_width"].values
        for m in base_methods:
            if f"{m}_width" not in gdf.columns or gdf[f"{m}_width"].isna().all():
                continue
            cov = float(gdf[f"{m}_covered"].mean()) * 100.0
            wid = float(gdf[f"{m}_width"].mean())
            med_w = float(gdf[f"{m}_width"].median())
            sev = float((np.minimum(1.0, gdf[f"{m}_miss_dist"] / gdf["tau_scale"])).mean())
            w_arr = gdf[f"{m}_width"].values
            e_arr = gdf["abs_err_hard"].values
            corr = float(np.corrcoef(w_arr, e_arr)[0, 1]) if np.std(w_arr) > 1e-8 else 0.0

            grp_covs = [float(gg[f"{m}_covered"].mean()) * 100.0 for _, gg in gdf.groupby("group")]
            min_grp_cov = min(grp_covs)
            cond_cov_gap = max(grp_covs) - min(grp_covs)

            try:
                p_vs_crc = float(wilcoxon(w_arr, ref_crc).pvalue)
            except Exception:
                p_vs_crc = 1.0
            try:
                p_vs_cl = float(wilcoxon(w_arr, ref_cl).pvalue)
            except Exception:
                p_vs_cl = 1.0

            summary_rows.append({
                "dataset": ds,
                "organ": org,
                "method": m,
                "coverage_pct": cov,
                "mean_width": wid,
                "median_width": med_w,
                "severity_risk": sev,
                "corr_width_err": corr,
                "min_group_cov_pct": min_grp_cov,
                "cond_cov_gap_pct": cond_cov_gap,
                "vs_crc_pct": (wid - float(np.mean(ref_crc))) / float(np.mean(ref_crc)) * 100.0,
                "vs_compass_l_pct": (wid - float(np.mean(ref_cl))) / float(np.mean(ref_cl)) * 100.0,
                "p_vs_crc": p_vs_crc,
                "p_vs_compass_l": p_vs_cl,
            })
    return pd.DataFrame(summary_rows)


def generate_figures(summary_df: pd.DataFrame, fig_dir: Path, alpha: float) -> None:
    """Create comparison plots in results/figures/."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    test_3d = summary_df[summary_df["dataset"] == "testing_3d"]
    if not test_3d.empty:
        sel_methods = [
            ("scp_hard", "Split CP\n(Hard)", "#7F8C8D"),
            ("crc", "CRC\n(ICLR'24)", "#E74C3C"),
            ("ascp_sens", "Adaptive\nSCP", "#E67E22"),
            ("compass_l", "COMPASS-L\n(ICLR'26)", "#1ABC9C"),
            ("compass_j_32ch", "COMPASS-J\n(ICLR'26)", "#F39C12"),
            ("dacrc_cv_plus", "CRC-FS-L\n(Ours)", "#27AE60"),
            ("dacrc_oof", "CRC-FS-J\n(Ours)", "#16A085"),
        ]
        organs = [o for o in ("LV", "RV", "MYO") if o in test_3d["organ"].values]
        fig, axes = plt.subplots(2, len(organs), figsize=(5.6 * len(organs), 8.5), sharex=False)
        if len(organs) == 1:
            axes = axes[:, None]
        fig.suptitle(
            f"CRC-FS (CRC-FS-L & CRC-FS-J) vs. Conformal Risk Control (ICLR 2024) & COMPASS (ICLR 2026) — ACDC Unseen Test Set (Target {(1-alpha)*100:.0f}%)",
            fontsize=12.5, fontweight="bold",
        )
        for col_i, org in enumerate(organs):
            sub = test_3d[test_3d["organ"] == org].set_index("method")
            ms = [(k, lbl, c) for k, lbl, c in sel_methods if k in sub.index]
            xs = np.arange(len(ms))
            covs = [sub.loc[k, "coverage_pct"] for k, _, _ in ms]
            wids = [sub.loc[k, "mean_width"] for k, _, _ in ms]
            cols = [c for _, _, c in ms]
            lbls = [lbl for _, lbl, _ in ms]

            ax_top = axes[0, col_i]
            bars_c = ax_top.bar(xs, covs, color=cols, edgecolor="black", linewidth=0.7)
            ax_top.axhline((1 - alpha) * 100, color="red", linestyle="--", linewidth=1.4)
            ax_top.set_title(f"{org} Volume — Coverage (%)", fontweight="bold")
            ax_top.set_ylim(75, 102)
            ax_top.set_xticks(xs)
            ax_top.set_xticklabels(lbls, fontsize=7.5)
            ax_top.grid(axis="y", linestyle=":", alpha=0.4)
            for b, v in zip(bars_c, covs):
                ax_top.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

            ax_bot = axes[1, col_i]
            bars_w = ax_bot.bar(xs, wids, color=cols, edgecolor="black", linewidth=0.7)
            ax_bot.set_title(f"{org} Volume — Mean Width (mL, Lower=Better)", fontweight="bold")
            ax_bot.set_ylim(0, max(wids) * 1.2)
            ax_bot.set_xticks(xs)
            ax_bot.set_xticklabels(lbls, fontsize=7.5)
            ax_bot.grid(axis="y", linestyle=":", alpha=0.4)
            for b, v in zip(bars_w, wids):
                ax_bot.text(b.get_x() + b.get_width() / 2, v + max(wids) * 0.015, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        plt.tight_layout()
        out_multi = fig_dir / "crc_fs_multi_organ.png"
        plt.savefig(str(out_multi), dpi=250, bbox_inches="tight")
        plt.close(fig)

    primary = summary_df[(summary_df["dataset"] == "testing_3d") & (summary_df["organ"] == "LV")]
    if primary.empty:
        primary = summary_df[summary_df["organ"] == "LV"].head(15)
    if not primary.empty:
        sub = primary.set_index("method")
        plot_ms = [
            ("scp_hard", "Split CP\n(Hard)", "#7F8C8D"),
            ("crc", "CRC\n(ICLR'24)", "#E74C3C"),
            ("ascp_sens", "Adaptive\nSCP", "#E67E22"),
            ("compass_l", "COMPASS-L", "#1ABC9C"),
            ("compass_j_32ch", "COMPASS-J\n(32ch)", "#F39C12"),
            ("dacrc_cv_plus", "CRC-FS-L\n(Ours)", "#27AE60"),
            ("dacrc_oof", "CRC-FS-J\n(Ours)", "#16A085"),
        ]
        plot_ms = [(k, l, c) for k, l, c in plot_ms if k in sub.index]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        xs = np.arange(len(plot_ms))
        covs = [sub.loc[k, "coverage_pct"] for k, _, _ in plot_ms]
        wids = [sub.loc[k, "mean_width"] for k, _, _ in plot_ms]
        cols = [c for _, _, c in plot_ms]
        lbls = [l for _, l, _ in plot_ms]

        ax1.bar(xs, covs, color=cols, edgecolor="black", linewidth=0.8)
        ax1.axhline((1 - alpha) * 100, color="red", linestyle="--", linewidth=1.5)
        ax1.set_title("Empirical Coverage (%) — LV Volume", fontweight="bold")
        ax1.set_ylim(75, 102)
        ax1.set_xticks(xs); ax1.set_xticklabels(lbls, fontsize=8.5)
        ax1.grid(axis="y", linestyle=":", alpha=0.4)
        for i, v in enumerate(covs):
            ax1.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax2.bar(xs, wids, color=cols, edgecolor="black", linewidth=0.8)
        ax2.set_title("Mean Interval Width (mL, Smaller = Better)", fontweight="bold")
        ax2.set_ylim(0, max(wids) * 1.2)
        ax2.set_xticks(xs); ax2.set_xticklabels(lbls, fontsize=8.5)
        ax2.grid(axis="y", linestyle=":", alpha=0.4)
        for i, v in enumerate(wids):
            ax2.text(i, v + max(wids) * 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        plt.tight_layout()
        plt.savefig(str(fig_dir / "crc_fs_full_comparison.png"), dpi=250, bbox_inches="tight")
        plt.close(fig)

        fig2, ax = plt.subplots(figsize=(9, 6))
        for k, lbl, c in plot_ms:
            lbl_clean = lbl.replace("\n", " ")
            size = 240 if "dacrc" in k else 130
            ax.scatter(sub.loc[k, "mean_width"], sub.loc[k, "coverage_pct"], c=c, s=size, edgecolors="black", label=lbl_clean, zorder=4 if "dacrc" in k else 3)
        ax.axhline((1 - alpha) * 100, color="red", linestyle="--", linewidth=1.4, label=f"Target {(1-alpha)*100:.0f}%")
        ax.set_xlabel("Mean Interval Width (mL) — Smaller is Better")
        ax.set_ylabel("Empirical Coverage (%) — Higher is Better")
        ax.set_title("Width vs. Coverage Pareto Frontier (Top-Left = Optimal)", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8.5, loc="lower right")
        plt.tight_layout()
        plt.savefig(str(fig_dir / "crc_fs_pareto.png"), dpi=250, bbox_inches="tight")
        plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="Run ACDC DA-CRC-FS vs CRC (ICLR 2024) & COMPASS benchmark")
    parser.add_argument("--split", type=str, default="all", choices=["testing", "training", "all"])
    parser.add_argument("--model", type=str, default="3d_fullres", choices=["3d_fullres", "2d", "both"])
    parser.add_argument("--organs", nargs="+", default=["LV", "RV", "MYO"], choices=["LV", "RV", "MYO"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--skip_ef", action="store_true")
    args = parser.parse_args()

    results_dir = WORKSPACE_DIR / "results"
    tables_dir = results_dir / "tables"
    fig_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    if args.split in ("testing", "all"):
        if args.model in ("3d_fullres", "both"):
            runs.append(("testing_3d", WORKSPACE_DIR / "data_ACDC/testing", WORKSPACE_DIR / "nnunet_output_testing", "3d_fullres"))
        if args.model in ("2d", "both") and (WORKSPACE_DIR / "nnunet_output_testing_2d").exists():
            runs.append(("testing_2d", WORKSPACE_DIR / "data_ACDC/testing", WORKSPACE_DIR / "nnunet_output_testing_2d", "2d"))
    if args.split in ("training", "all"):
        runs.append(("training_3d", WORKSPACE_DIR / "data_ACDC/training", WORKSPACE_DIR / "nnunet_output", "3d_fullres"))

    all_sample_dfs = []
    all_fold_dfs = []
    ef_dfs = []

    t0 = time()
    for ds_name, data_dir, pred_dir, model_type in runs:
        if not pred_dir.exists() or not any(pred_dir.glob("*.npz")):
            print(f"[SKIP] Missing .npz predictions in {pred_dir}", flush=True)
            continue
        head_w = load_head_weight(model_type)
        print(f"\n>>> Dataset: {ds_name} | Head weight: {None if head_w is None else head_w.shape}", flush=True)

        for org in args.organs:
            t_org = time()
            cls_id = ORGAN_CLASSES[org]
            samples = load_samples(data_dir, pred_dir, cls_id, head_weight=head_w)
            print(f"  [{ds_name} | {org}] Loaded {len(samples)} frames...", flush=True)
            df_s, df_f = evaluate_organ_kfold(samples, cls_id, org, ds_name, head_w, alpha=args.alpha)
            all_sample_dfs.append(df_s)
            all_fold_dfs.append(df_f)
            print(f"  [{ds_name} | {org}] Completed in {time() - t_org:.1f}s", flush=True)

            if org == "LV" and not args.skip_ef:
                ef_df = evaluate_ef_kfold(samples, ds_name, alpha=args.alpha)
                ef_dfs.append(ef_df)

    df_all = pd.concat(all_sample_dfs, ignore_index=True)
    df_folds = pd.concat(all_fold_dfs, ignore_index=True)
    summary_df = summarize_benchmark(df_all)

    summary_df.to_csv(str(tables_dir / "benchmark_summary.csv"), index=False)
    df_all.to_csv(str(tables_dir / "per_sample_results.csv"), index=False)
    primary_mask = (df_all["dataset"] == runs[0][0]) & (df_all["organ"] == args.organs[0])
    df_all[primary_mask].to_csv(str(results_dir / "crc_fs_results.csv"), index=False)
    df_folds.to_csv(str(results_dir / "crc_fs_fold_summary.csv"), index=False)

    if ef_dfs:
        df_ef = pd.concat(ef_dfs, ignore_index=True)
        df_ef.to_csv(str(tables_dir / "ef_results.csv"), index=False)

    generate_figures(summary_df, fig_dir, args.alpha)

    print("\n" + "=" * 108)
    print(f"BENCHMARK SUMMARY — ACDC (Target Coverage = {(1 - args.alpha)*100:.0f}%, Patient-Level 5-Fold CV)")
    print("=" * 108)
    display_names = {
        "dacrc_oof": "crc_fs_j",
        "dacrc_cv_plus": "crc_fs_l",
        "dacrc_manifold": "crc_fs_manifold",
        "dacrc_mondrian": "crc_fs_mondrian",
        "dacrc_risk_crc": "crc_fs_risk",
        "dacrc_ef": "crc_fs_ef",
    }
    for (ds, org), sub in summary_df.groupby(["dataset", "organ"], sort=False):
        print(f"\n--- {ds} | {org} ---")
        print(
            f"{'Method':<18} {'Cov(%)':>7} {'MeanW':>8} {'MedW':>8} {'SevRisk':>8} "
            f"{'Corr(W,e)':>10} {'MinGrp':>7} {'GrpGap':>7} {'vs CRC':>9} {'vs COMPASS-L':>13}"
        )
        for _, r in sub.iterrows():
            m_disp = display_names.get(r["method"], r["method"])
            print(
                f"{m_disp:<18} {r['coverage_pct']:6.1f}% {r['mean_width']:8.3f} "
                f"{r['median_width']:8.3f} {r['severity_risk']:8.4f} {r['corr_width_err']:+10.3f} "
                f"{r['min_group_cov_pct']:6.1f}% {r['cond_cov_gap_pct']:6.1f}% "
                f"{r['vs_crc_pct']:+8.1f}% {r['vs_compass_l_pct']:+12.1f}%"
            )

    if ef_dfs:
        print("\n--- LV EJECTION FRACTION (EF %) ---")
        for ds, sub in df_ef.groupby("dataset", sort=False):
            for m in ("scp_hard", "crc", "compass_l", "dacrc_ef"):
                m_disp = display_names.get(m, m)
                print(
                    f"  {ds:<12} {m_disp:<12}: cov={sub[f'{m}_covered'].mean()*100:5.1f}%  "
                    f"width={sub[f'{m}_width'].mean():6.2f} %"
                )

    print(f"\nAll results saved to {results_dir} (total elapsed: {time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
