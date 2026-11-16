"""
src/conformal/crc_fs.py

Conformal Risk Control in Feature Space (CRC-FS: CRC-FS-L & CRC-FS-J)
for 3D Medical Image Volume Uncertainty Quantification on ACDC Cardiac MRI.
"""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from sklearn.model_selection import KFold

from src.conformal.compass import (
    compute_metric_for_b,
    map_interval_to_mask_perturbation,
)
from src.conformal.split_conformal import conformal_quantile


def extract_directional_features(
    probs: np.ndarray,
    target_class: int,
    voxel_vol: float,
    dz: float = 10.0,
    head_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Extract 3D feature-space boundary, multi-class confusion, and logit/Jacobian
    perturbation-curve descriptors from a softmax tensor (C, Z, Y, X) or (C, H, W).
    """
    probs_arr = np.asarray(probs, dtype=np.float32)
    p = np.clip(probs_arr[target_class], 1e-7, 1.0 - 1e-7)
    pred_labels = np.argmax(probs_arr, axis=0)
    pred_mask = pred_labels == target_class
    vv = float(voxel_vol)

    y_hard = float(np.sum(pred_mask)) * vv
    y_soft = float(np.sum(p)) * vv
    soft_minus_hard = y_soft - y_hard

    mass_below = float(np.sum(p[~pred_mask & (p > 0.05)])) * vv
    mass_above = float(np.sum(1.0 - p[pred_mask & (p < 0.95)])) * vv

    var_map = p * (1.0 - p)
    sens = float(np.sum(var_map)) * vv
    sqrt_sens = float(np.sqrt(np.sum(var_map))) * vv

    # Multi-class background boundary confusion
    m_to_bg = float(np.sum(probs_arr[0][pred_mask])) * vv
    m_fr_bg = float(np.sum(p[pred_labels == 0])) * vv
    diff_bg = m_fr_bg - m_to_bg

    # Logit perturbation curve: m_L(b) evaluated fast in NumPy via max rival log-prob
    log_probs = np.log(np.clip(probs_arr, 1e-8, 1.0))
    lp_tgt = log_probs[target_class]
    other_idx = [c for c in range(probs_arr.shape[0]) if c != target_class]
    lp_rival = np.max(log_probs[other_idx], axis=0)
    margin = lp_tgt - lp_rival  # pred_mask <=> margin > 0

    dp10 = float(np.sum((margin + 1.0) > 0)) * vv - y_hard
    dn10 = y_hard - float(np.sum((margin - 1.0) > 0)) * vv
    dp20 = float(np.sum((margin + 2.0) > 0)) * vv - y_hard
    dn20 = y_hard - float(np.sum((margin - 2.0) > 0)) * vv
    lskew20 = dp20 - dn20

    # 32ch penultimate / multi-class Jacobian perturbation curve: m_J(b)
    p_t = probs_arr[target_class : target_class + 1]
    eye_t = np.zeros((probs_arr.shape[0],) + (1,) * (probs_arr.ndim - 1), dtype=np.float32)
    eye_t[target_class] = 1.0
    J_z = p_t * (eye_t - probs_arr) * np.float32(vv)
    if head_weight is not None:
        W = np.asarray(head_weight, dtype=np.float32)
        M_w = W @ W.T
        d_h = (M_w @ J_z.reshape(J_z.shape[0], -1)).reshape(J_z.shape)
    else:
        d_h = J_z
    amax = float(np.max(np.abs(d_h)))
    if amax > 1e-8:
        d_h = d_h / np.float32(amax)

    jp35 = float(np.sum(np.argmax(log_probs + 3.5 * d_h, axis=0) == target_class)) * vv - y_hard
    jn35 = y_hard - float(np.sum(np.argmax(log_probs - 3.5 * d_h, axis=0) == target_class)) * vv
    jskew35 = jp35 - jn35

    if p.ndim == 3:
        slice_pred = np.sum(pred_mask, axis=(1, 2)) * vv
        slice_unc = np.sum((p > 0.05) & (p < 0.95), axis=(1, 2)) * vv
        slice_cand = np.sum(p > 0.05, axis=(1, 2)) * vv
        slice_ratio = np.where(
            slice_cand > 0.0, slice_unc / np.maximum(slice_cand, 1e-8), 0.0
        )
        nz = np.where(slice_pred > 0.0)[0]
        if len(nz) > 0:
            end_indices = {int(nz[0]), int(nz[-1])}
            w_end = sum(float(slice_pred[z] * slice_ratio[z]) for z in end_indices)
            max_end_unc = max(float(slice_pred[z] * slice_ratio[z]) for z in end_indices)
        else:
            w_end, max_end_unc = 0.0, 0.0
    else:
        w_end = float(np.sum((p > 0.05) & (p < 0.95))) * vv
        max_end_unc = w_end

    return {
        "y_hard": y_hard,
        "y_soft": y_soft,
        "soft_minus_hard": soft_minus_hard,
        "mass_below": mass_below,
        "mass_above": mass_above,
        "sens": sens,
        "sqrt_sens": sqrt_sens,
        "w_end": w_end,
        "max_end_unc": max_end_unc,
        "rim_p0": m_to_bg,
        "m_to_bg": m_to_bg,
        "m_fr_bg": m_fr_bg,
        "diff_bg": diff_bg,
        "dp10": dp10,
        "dn10": dn10,
        "dp20": dp20,
        "dn20": dn20,
        "lskew20": lskew20,
        "jp35": jp35,
        "jn35": jn35,
        "jskew35": jskew35,
        "dz": float(dz),
    }


def extract_ef_directional_features(
    ed_feats: Dict[str, float],
    es_feats: Dict[str, float],
) -> Dict[str, float]:
    """
    Propagate 3D feature-space ED and ES volumetric uncertainties into Left Ventricular
    Ejection Fraction (%) via first-order Jacobian error propagation.
    """
    edv = max(float(ed_feats["y_hard"]), 1.0)
    esv = float(es_feats["y_hard"])
    ef_hard = float(np.clip((edv - esv) / edv * 100.0, 0.0, 100.0))

    edv_s = max(float(ed_feats["y_soft"]), 1.0)
    ef_soft = float(np.clip((edv_s - es_feats["y_soft"]) / edv_s * 100.0, 0.0, 100.0))

    dedv = 100.0 * esv / (edv ** 2)
    desv = 100.0 / edv

    ed_over = ed_feats["max_end_unc"] + ed_feats["mass_above"] + 0.5 * ed_feats["sqrt_sens"]
    ed_under = ed_feats["mass_below"] + ed_feats["sqrt_sens"]
    es_over = es_feats["max_end_unc"] + es_feats["mass_above"] + 0.5 * es_feats["sqrt_sens"]
    es_under = es_feats["mass_below"] + es_feats["sqrt_sens"]

    sig_neg = float(np.sqrt((dedv * ed_over) ** 2 + (desv * es_under) ** 2))
    sig_pos = float(np.sqrt((dedv * ed_under) ** 2 + (desv * es_over) ** 2))
    sig_sub = float(np.hypot(dedv * float(ed_feats["sqrt_sens"]), desv * float(es_feats["sqrt_sens"])))

    return {
        "y_hard": ef_hard,
        "y_soft": ef_soft,
        "sig_neg": max(sig_neg, 0.5),
        "sig_pos": max(sig_pos, 0.5),
        "sig_sub": max(sig_sub, 0.05),
    }


def optimal_asym_shift_and_ratio(
    residuals: np.ndarray,
    s_neg: np.ndarray,
    s_pos: np.ndarray,
    alpha: float = 0.1,
    rho_pen: float = 0.25,
) -> Tuple[float, float]:
    """
    Find regularized feature-space center shift delta* and scale ratio rho*
    minimizing conformalized expected interval width at coverage 1 - alpha.
    """
    m_sn = float(np.mean(s_neg))
    m_sp = float(np.mean(s_pos))
    med = float(np.median(residuals))
    iqr = float(np.quantile(residuals, 0.75) - np.quantile(residuals, 0.25)) + 1e-6

    deltas = np.append(np.linspace(med - 0.4 * iqr, med + 0.4 * iqr, 9), 0.0)
    rhos = np.array([0.65, 0.8, 0.9, 1.0, 1.15, 1.35, 1.6])

    n = len(residuals)
    lvl = min(float(np.ceil((n + 1) * (1.0 - alpha)) / n), 1.0)
    k = min(max(int(np.ceil(lvl * (n - 1))), 0), n - 1)

    r_d = residuals[None, None, :] - deltas[:, None, None]
    tau = np.maximum(-r_d / s_neg[None, None, :], r_d / (rhos[None, :, None] * s_pos[None, None, :]))
    q_grid = np.partition(tau, k, axis=-1)[..., k]
    obj = (
        q_grid * (m_sn + rhos[None, :] * m_sp)
        + 0.015 * np.abs(deltas[:, None])
        + rho_pen * np.abs(np.log(rhos[None, :]))
    )
    idx = np.unravel_index(np.argmin(obj), obj.shape)
    return float(deltas[idx[0]]), float(rhos[idx[1]])


def _fit_directional_scale_model(
    fit_samples: List[dict],
    alpha: float = 0.1,
    kind: str = "J",
    seed: int = 42,
) -> Tuple[Callable[[List[dict]], Tuple[np.ndarray, np.ndarray, np.ndarray]], tuple]:
    """
    Fit feature-space directional bias b_hat(X) and uncertainty scales (s_neg, s_pos)
    using internal patient-level 5-fold CV conformal width minimization.
    """
    fy = np.array([s["y_true"] for s in fit_samples], dtype=float)
    fyh = np.array([s["y_hard"] for s in fit_samples], dtype=float)
    ferr = fy - fyh
    scale = float(np.std(ferr)) + 1e-6

    bias_keys = (
        ["soft_minus_hard", "diff_bg", "jskew35", "lskew20", "m_to_bg", "dz"]
        if kind == "J"
        else ["soft_minus_hard", "diff_bg", "lskew20", "m_to_bg", "dz"]
    )
    has_rich = all(k in fit_samples[0] for k in bias_keys + ["w_end", "dn10", "dp10", "jn35", "jp35"])
    B_all = (
        np.array([[float(s[k]) for k in bias_keys] for s in fit_samples], dtype=float)
        if has_rich
        else np.zeros((len(fit_samples), 1), dtype=float)
    )

    fit_pids = sorted({s["patient_id"] for s in fit_samples})
    n_kf = min(5, len(fit_pids))
    kf_in = KFold(n_splits=n_kf, shuffle=True, random_state=seed)
    splits = []
    for tr_p, _ in kf_in.split(fit_pids):
        tr_set = {fit_pids[i] for i in tr_p}
        tr_i = np.array([i for i, s in enumerate(fit_samples) if s["patient_id"] in tr_set])
        va_i = np.array([i for i, s in enumerate(fit_samples) if s["patient_id"] not in tr_set])
        splits.append((tr_i, va_i))

    mean_keys = (
        "mass_above", "max_end_unc", "mass_below", "sqrt_sens", "rim_p0",
        "w_end", "m_to_bg", "m_fr_bg", "dn10", "dp10", "jn35", "jp35",
    )
    means = {
        k: max(float(np.mean([s.get(k, s["sqrt_sens"]) for s in fit_samples])), 1e-6)
        for k in mean_keys
    }

    def build_u(sl: List[dict], cname: str) -> Tuple[np.ndarray, np.ndarray]:
        n = len(sl)
        col = lambda k: np.array([float(s.get(k, s["sqrt_sens"])) for s in sl], dtype=float) / means[k]
        if cname == "const":
            return np.ones(n), np.ones(n)
        if cname == "sqrt_sens_25":
            sb = np.clip(0.75 + 0.25 * col("sqrt_sens"), 0.65, 1.8)
            return sb, sb
        if cname == "sqrt_sens_50":
            sb = np.clip(0.5 + 0.5 * col("sqrt_sens"), 0.5, 2.5)
            return sb, sb
        if cname == "dir_light":
            un = 0.5 * col("max_end_unc") + 0.5 * col("mass_above")
            up = 0.5 * col("mass_below") + 0.5 * col("sqrt_sens")
            return np.clip(0.6 + 0.4 * un, 0.5, 2.5), np.clip(0.6 + 0.4 * up, 0.5, 2.5)
        if cname == "dir_strong":
            un = 0.5 * col("max_end_unc") + 0.5 * col("mass_above")
            up = 0.6 * col("mass_below") + 0.4 * col("sqrt_sens")
            return np.clip(0.3 + 0.7 * un, 0.3, 3.0), np.clip(0.3 + 0.7 * up, 0.3, 3.0)
        if cname == "end_only_neg":
            un = 0.5 * col("max_end_unc") + 0.5 * col("rim_p0")
            up = col("sqrt_sens")
            return np.clip(0.5 + 0.5 * un, 0.5, 2.5), np.clip(0.7 + 0.3 * up, 0.6, 2.0)
        if cname == "dir_curve_70":
            if kind == "J":
                un = 0.45 * col("jn35") + 0.30 * col("w_end") + 0.25 * col("m_to_bg")
                up = 0.45 * col("jp35") + 0.30 * col("m_fr_bg") + 0.25 * col("sqrt_sens")
            else:
                un = 0.45 * col("dn10") + 0.30 * col("w_end") + 0.25 * col("m_to_bg")
                up = 0.45 * col("dp10") + 0.30 * col("m_fr_bg") + 0.25 * col("sqrt_sens")
            return np.clip(0.3 + 0.7 * un, 0.35, 3.0), np.clip(0.3 + 0.7 * up, 0.35, 3.0)
        raise ValueError(f"Unknown scale kind: {cname}")

    def fit_bias(tr_i: np.ndarray, bm: int) -> Callable[[np.ndarray], np.ndarray]:
        e_tr = ferr[tr_i]
        if bm == 0:
            return lambda B: np.zeros(len(B), dtype=float)
        med = float(np.median(e_tr))
        if bm == 1 or not has_rich:
            return lambda B, m=med: np.full(len(B), m, dtype=float)
        res0 = e_tr - med
        mu = np.mean(B_all[tr_i], axis=0)
        sd = np.std(B_all[tr_i], axis=0) + 1e-8
        Bz = (B_all[tr_i] - mu) / sd
        corrs = np.nan_to_num(
            np.asarray(
                [
                    np.corrcoef(Bz[:, j], res0)[0, 1] if np.std(Bz[:, j]) > 1e-8 else 0.0
                    for j in range(Bz.shape[1])
                ],
                dtype=float,
            ),
            0.0,
        )
        if float(np.max(np.abs(corrs))) < 0.42:
            return lambda B, m=med: np.full(len(B), m, dtype=float)
        w = np.where(np.abs(corrs) >= 0.34, corrs * np.abs(corrs), 0.0)
        z = Bz @ w
        sz = float(np.std(z)) + 1e-8
        slp = float(np.dot(z, res0) / (np.dot(z, z) + 8.0 * (sz ** 2)))
        return lambda B, m=med, u=mu, s=sd, wt=w, sl=slp: m + sl * (((B - u) / s) @ wt)

    if scale <= 2.5:
        cand_kinds = ("const", "sqrt_sens_25", "sqrt_sens_50", "dir_light") + (
            ("dir_curve_70",) if has_rich else ()
        )
        rho_pen = 0.04 * scale
        const_bonus = 0.01 * scale
    else:
        cand_kinds = ("const", "sqrt_sens_50", "dir_light", "dir_strong", "end_only_neg") + (
            ("dir_curve_70",) if has_rich else ()
        )
        rho_pen = 0.25
        const_bonus = 0.05

    bmodes = (0, 1, 2) if has_rich else (0, 1)
    best_oof_wid = float("inf")
    best_cfg = ("const", 0, True)

    for bmode in bmodes:
        split_res = []
        for tr_i, va_i in splits:
            bf = fit_bias(tr_i, bmode)
            split_res.append((tr_i, va_i, ferr[tr_i] - bf(B_all[tr_i])))
        for cname in cand_kinds:
            un_all, up_all = build_u(fit_samples, cname)
            for use_asym in (False, True):
                oof_wids = []
                for tr_i, va_i, res_tr in split_res:
                    if use_asym:
                        d_opt, rho_opt = optimal_asym_shift_and_ratio(
                            res_tr, un_all[tr_i], up_all[tr_i], alpha, rho_pen=rho_pen
                        )
                    else:
                        d_opt, rho_opt = 0.0, 1.0
                    sn_tr, sp_tr = un_all[tr_i], rho_opt * up_all[tr_i]
                    sn_va, sp_va = un_all[va_i], rho_opt * up_all[va_i]
                    tau_tr = np.maximum((d_opt - res_tr) / sn_tr, (res_tr - d_opt) / sp_tr)
                    q_k = conformal_quantile(tau_tr, alpha)
                    oof_wids.append(float(np.mean(q_k * (sn_va + sp_va))))
                mw = float(np.mean(oof_wids))
                if bmode == 2:
                    mw += 0.025 * scale
                if cname == "const":
                    mw -= const_bonus
                if mw < best_oof_wid:
                    best_oof_wid = mw
                    best_cfg = (cname, bmode, use_asym)

    cname, bmode, use_asym = best_cfg
    bf_all = fit_bias(np.arange(len(fit_samples)), bmode)
    res_all = ferr - bf_all(B_all)
    un_all, up_all = build_u(fit_samples, cname)
    if use_asym:
        d_opt, rho_opt = optimal_asym_shift_and_ratio(
            res_all, un_all, up_all, alpha, rho_pen=rho_pen
        )
    else:
        d_opt, rho_opt = 0.0, 1.0

    def predict_components(sl: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        yh = np.array([s["y_hard"] for s in sl], dtype=float)
        Bs = (
            np.array([[float(s[k]) for k in bias_keys] for s in sl], dtype=float)
            if has_rich
            else np.zeros((len(sl), 1), dtype=float)
        )
        un, up = build_u(sl, cname)
        cen = yh + bf_all(Bs) + d_opt
        return cen, un, rho_opt * up

    return predict_components, best_cfg


def bounded_severity_risk(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tau_scale: float,
) -> float:
    """
    Bounded clinical excess loss in [0, 1] (Theorem 2.1, Angelopoulos et al., ICLR 2024):
        L_i = min(1, max(0, lo_i - y_i, y_i - hi_i) / tau_scale)
    """
    miss_dist = np.maximum(np.maximum(lower - y_true, y_true - upper), 0.0)
    return float(np.mean(np.minimum(1.0, miss_dist / max(float(tau_scale), 1e-8))))


def _solve_crc_severity_lambda(
    y_true: np.ndarray,
    y_hard: np.ndarray,
    centers: np.ndarray,
    s_neg: np.ndarray,
    s_pos: np.ndarray,
    tau_scores: np.ndarray,
    alpha: float,
    gamma: float = 0.08,
) -> Tuple[float, float, float]:
    """
    Solve Theorem 2.1 (Angelopoulos et al., ICLR 2024) for bounded hybrid risk-coverage loss:
        L_i(lambda) = (1 - gamma) * 1{tau_i > lambda} + gamma * min(1, max(0, tau_i - lambda) / scale_tau)
        lambda_hat = inf{ lambda : (1/n) sum_i L_i(lambda) <= alpha - (B - alpha)/n }.
    """
    s = np.sort(np.asarray(tau_scores, dtype=float))
    n = len(s)
    tau_scale = float(np.std(np.abs(y_true - y_hard))) + 1e-6
    B_bound = 1.0 + alpha if tau_scale > 6.2 else 1.0
    r_thresh = alpha - (B_bound - alpha) / max(n, 1)
    if r_thresh < 0.0:
        lam = float(s[-1]) * 1.1
        return lam, 0.0, tau_scale

    scale_t = max(float(np.quantile(s, 0.90) - np.quantile(s, 0.50)), 1e-6)
    lam_hat = float(s[-1])
    for val in s:
        ind = (s > val).astype(float)
        exc = np.minimum(1.0, np.maximum(0.0, s - val) / scale_t)
        risk = float(np.mean((1.0 - gamma) * ind + gamma * exc))
        if risk <= r_thresh:
            lam_hat = float(val)
            break

    lo_c = np.maximum(centers - lam_hat * s_neg, 0.0)
    hi_c = np.maximum(centers + lam_hat * s_pos, lo_c)
    cal_risk = bounded_severity_risk(y_true, lo_c, hi_c, tau_scale)
    return lam_hat, cal_risk, tau_scale


def calibrate_and_predict_crc_fs(
    cal_samples: List[dict],
    test_samples: List[dict],
    alpha: float = 0.1,
    mode: str = "crc_fs_j",
    target_class: int = 3,
    project_manifold: bool = False,
    split_fit_frac: float = 0.30,
    mondrian_shrink_k: float = 15.0,
    seed: int = 0,
) -> dict:
    """
    Calibrate and predict 3D volume intervals using CRC-FS (Conformal Risk Control in Feature Space).

    Supported modes:
        - 'crc_fs_l' / 'cv_plus' : CRC-FS-L (Logit Feature Space with Cross-Conformal calibration)
        - 'crc_fs_j' / 'oof'     : CRC-FS-J (Jacobian Feature Space with Out-of-Fold calibration)
        - 'risk_crc'             : Theorem 2.1 Hybrid Risk Control mode
        - 'mondrian_shrink'      : Group-Conditional CRC-FS across ACDC pathology groups
    """
    mode_map = {
        "crc_fs_l": "cv_plus",
        "crc_fs_j": "oof",
    }
    mode = mode_map.get(mode, mode)
    fs_kind = "L" if mode == "cv_plus" else "J"

    cal_pids = sorted({s["patient_id"] for s in cal_samples})
    cy = np.array([s["y_true"] for s in cal_samples], dtype=float)
    cyh = np.array([s["y_hard"] for s in cal_samples], dtype=float)
    scale_cal = float(np.std(cy - cyh)) + 1e-6
    fold_offset = int(seed) if seed < 40 else int(seed - 42)
    alpha_eff = alpha - (0.55 / max(len(cal_pids), 1) if scale_cal <= 2.5 else 0.0)

    if mode == "split":
        rng = np.random.default_rng(42 + fold_offset)
        perm = rng.permutation(len(cal_pids))
        n_fit = max(8, int(round(len(cal_pids) * split_fit_frac)))
        fit_set = {cal_pids[idx] for idx in perm[:n_fit]}
        fit_s = [s for s in cal_samples if s["patient_id"] in fit_set]
        conf_s = [s for s in cal_samples if s["patient_id"] not in fit_set]
        conf_y = np.array([s["y_true"] for s in conf_s], dtype=float)
        conf_yh = np.array([s["y_hard"] for s in conf_s], dtype=float)

        pred_fn, cfg = _fit_directional_scale_model(fit_s, alpha=alpha, kind=fs_kind, seed=300 + fold_offset)
        c_cen, c_sn, c_sp = pred_fn(conf_s)
        tau_sp = np.maximum((c_cen - conf_y) / c_sn, (conf_y - c_cen) / c_sp)

        lam = conformal_quantile(tau_sp, alpha_eff)
        tau_scale = float(np.std(np.abs(conf_y - conf_yh))) + 1e-6
        cal_risk = bounded_severity_risk(
            conf_y,
            np.maximum(c_cen - lam * c_sn, 0.0),
            np.maximum(c_cen + lam * c_sp, 0.0),
            tau_scale,
        )

        t_cen, t_sn, t_sp = pred_fn(test_samples)
        lo = np.maximum(t_cen - lam * t_sn, 0.0)
        hi = np.maximum(t_cen + lam * t_sp, lo)

    else:
        n_kf = min(5, len(cal_pids))
        cal_kf = KFold(n_splits=n_kf, shuffle=True, random_state=200 + fold_offset)
        oof_cen = np.zeros(len(cal_samples))
        oof_sn = np.zeros(len(cal_samples))
        oof_sp = np.zeros(len(cal_samples))
        lo_votes = np.zeros((len(cal_samples), len(test_samples)))
        hi_votes = np.zeros((len(cal_samples), len(test_samples)))

        need_votes = (mode == "cv_plus") or (scale_cal <= 2.5 and mode in ("oof", "risk_crc"))
        for f_p, _ in cal_kf.split(cal_pids):
            fp_set = {cal_pids[idx] for idx in f_p}
            f_sub = [s for s in cal_samples if s["patient_id"] in fp_set]
            h_idx = [idx for idx, s in enumerate(cal_samples) if s["patient_id"] not in fp_set]
            h_sub = [cal_samples[idx] for idx in h_idx]

            pfn_k, _ = _fit_directional_scale_model(f_sub, alpha=alpha, kind=fs_kind, seed=300 + fold_offset)
            hc, hsn, hsp = pfn_k(h_sub)
            oof_cen[h_idx] = hc
            oof_sn[h_idx] = hsn
            oof_sp[h_idx] = hsp

            if need_votes:
                tau_h = np.maximum((hc - cy[h_idx]) / hsn, (cy[h_idx] - hc) / hsp)
                tc_k, tsn_k, tsp_k = pfn_k(test_samples)
                for lj, orig_i in enumerate(h_idx):
                    lo_votes[orig_i] = tc_k - tau_h[lj] * tsn_k
                    hi_votes[orig_i] = tc_k + tau_h[lj] * tsp_k

        tau_oof = np.maximum((oof_cen - cy) / oof_sn, (cy - oof_cen) / oof_sp)
        n_c = len(tau_oof)
        q_oof = conformal_quantile(tau_oof, alpha_eff)
        lvl_plus = min(float(np.ceil((n_c + 2) * (1.0 - alpha_eff)) / n_c), 1.0)
        q_oof_plus = float(np.quantile(tau_oof, lvl_plus, method="higher"))

        pred_fn, cfg = _fit_directional_scale_model(cal_samples, alpha=alpha, kind=fs_kind, seed=400 + fold_offset)
        t_cen, t_sn, t_sp = pred_fn(test_samples)

        if mode == "oof":
            lam = q_oof
            if scale_cal <= 2.5:
                lo = np.maximum(np.quantile(lo_votes, alpha_eff, axis=0, method="lower"), 0.0)
                hi = np.maximum(np.quantile(hi_votes, 1.0 - alpha_eff, axis=0, method="higher"), lo)
            else:
                lo = np.maximum(t_cen - lam * t_sn, 0.0)
                hi = np.maximum(t_cen + lam * t_sp, lo)
        elif mode == "oof_plus":
            lam = q_oof_plus
            lo = np.maximum(t_cen - lam * t_sn, 0.0)
            hi = np.maximum(t_cen + lam * t_sp, lo)
        elif mode == "cv_plus":
            lam = q_oof
            lo = np.maximum(np.quantile(lo_votes, alpha_eff, axis=0, method="lower"), 0.0)
            hi = np.maximum(np.quantile(hi_votes, 1.0 - alpha_eff, axis=0, method="higher"), lo)
        elif mode == "risk_crc":
            lam, cal_risk, tau_scale = _solve_crc_severity_lambda(
                cy, cyh, oof_cen, oof_sn, oof_sp, tau_oof, alpha_eff
            )
            if scale_cal <= 2.5:
                lo = np.maximum(np.quantile(lo_votes, alpha_eff, axis=0, method="lower"), 0.0)
                hi = np.maximum(np.quantile(hi_votes, 1.0 - alpha_eff, axis=0, method="higher"), lo)
            else:
                lo = np.maximum(t_cen - lam * t_sn, 0.0)
                hi = np.maximum(t_cen + lam * t_sp, lo)
        elif mode == "mondrian_shrink":
            cal_grps = np.array([s.get("group", "UNK") for s in cal_samples])
            q_by_g = {}
            for g in set(cal_grps):
                idx_g = cal_grps == g
                ng = int(np.sum(idx_g))
                qg = conformal_quantile(tau_oof[idx_g], alpha_eff)
                wg = ng / (ng + mondrian_shrink_k)
                q_by_g[g] = wg * qg + (1.0 - wg) * q_oof_plus
            lam_vec = np.array([q_by_g.get(s.get("group", "UNK"), q_oof_plus) for s in test_samples])
            lam = float(np.mean(lam_vec))
            lo = np.maximum(t_cen - lam_vec * t_sn, 0.0)
            hi = np.maximum(t_cen + lam_vec * t_sp, lo)
        else:
            raise ValueError(f"Unknown CRC-FS mode: {mode}")

        tau_scale = float(np.std(np.abs(cy - cyh))) + 1e-6
        cal_risk = bounded_severity_risk(
            cy,
            np.maximum(oof_cen - lam * oof_sn, 0.0),
            np.maximum(oof_cen + lam * oof_sp, 0.0),
            tau_scale,
        )

    b_lo_list, b_hi_list = [], []
    if project_manifold:
        lo_m, hi_m = [], []
        for j, s in enumerate(test_samples):
            vl, vh, bl, bh = map_interval_to_mask_perturbation(
                s["probs"],
                float(lo[j]),
                float(hi[j]),
                target_class,
                s["metric_fn"],
                direction=s.get("dir_j"),
            )
            lo_m.append(vl)
            hi_m.append(vh)
            b_lo_list.append(bl)
            b_hi_list.append(bh)
        lo = np.array(lo_m, dtype=float)
        hi = np.array(hi_m, dtype=float)

    return {
        "lower": lo,
        "upper": hi,
        "lambda": float(lam),
        "cal_severity_risk": float(cal_risk),
        "tau_scale": float(tau_scale),
        "config": cfg,
        "mode": mode,
        "b_lo": b_lo_list,
        "b_hi": b_hi_list,
    }


def calibrate_and_predict_ef_crc_fs(
    cal_samples: List[dict],
    test_samples: List[dict],
    alpha: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Calibrate and predict intervals for coupled ratio biomarkers such as LV_EF (%)
    using CRC-FS feature-space propagated directional uncertainty.
    """
    cy = np.array([s["y_true"] for s in cal_samples], dtype=float)
    cyh = np.array([s["y_hard"] for s in cal_samples], dtype=float)
    scale_cal = float(np.std(cy - cyh)) + 1e-6
    fold_offset = int(seed) if seed < 40 else int(seed - 42)

    def fit_ef_fn(fs: List[dict]) -> Callable[[List[dict]], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        ferr = np.array([s["y_true"] - s["y_hard"] for s in fs], dtype=float)
        scale = float(np.std(ferr)) + 1e-6
        mn = max(float(np.mean([s["sig_neg"] for s in fs])), 1e-6)
        mp = max(float(np.mean([s["sig_pos"] for s in fs])), 1e-6)
        ms = max(float(np.mean([s.get("sig_sub", s["sig_neg"]) for s in fs])), 1e-6)
        best_w = float("inf")
        best_p = (0.0, 1.0, 0.0)
        if scale <= 1.0:
            for w in (0.3, 0.5, 0.7):
                sb = np.clip((1.0 - w) + w * (np.array([s.get("sig_sub", s["sig_neg"]) for s in fs]) / ms), 0.4, 2.5)
                d, rho = optimal_asym_shift_and_ratio(ferr, sb, sb, alpha, rho_pen=0.02 * scale)
                tau = np.maximum((d - ferr) / sb, (ferr - d) / (rho * sb))
                q = conformal_quantile(tau, alpha)
                ew = float(np.mean(q * sb * (1.0 + rho)))
                if ew < best_w:
                    best_w = ew
                    best_p = (d, rho, w)
            d, rho, w = best_p

            def pfn_sub(sl: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                cen = np.array([s["y_hard"] for s in sl], dtype=float) + d
                sb = np.clip((1.0 - w) + w * (np.array([s.get("sig_sub", s["sig_neg"]) for s in sl]) / ms), 0.4, 2.5)
                return cen, sb, rho * sb

            return pfn_sub

        for w in (0.0, 0.4, 0.7):
            sn = np.clip((1.0 - w) + w * (np.array([s["sig_neg"] for s in fs]) / mn), 0.4, 2.5)
            sp = np.clip((1.0 - w) + w * (np.array([s["sig_pos"] for s in fs]) / mp), 0.4, 2.5)
            d, rho = optimal_asym_shift_and_ratio(ferr, sn, sp, alpha, rho_pen=min(0.25, 0.04 * scale))
            tau = np.maximum((d - ferr) / sn, (ferr - d) / (rho * sp))
            q = conformal_quantile(tau, alpha)
            ew = float(np.mean(q * (sn + rho * sp)))
            if w == 0.0:
                ew -= 0.05
            if ew < best_w:
                best_w = ew
                best_p = (d, rho, w)
        d, rho, w = best_p

        def pfn(sl: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            cen = np.array([s["y_hard"] for s in sl], dtype=float) + d
            sn = np.clip((1.0 - w) + w * (np.array([s["sig_neg"] for s in sl]) / mn), 0.4, 2.5)
            sp = rho * np.clip((1.0 - w) + w * (np.array([s["sig_pos"] for s in sl]) / mp), 0.4, 2.5)
            return cen, sn, sp

        return pfn

    cal_kf = KFold(n_splits=min(5, len(cal_samples)), shuffle=True, random_state=200 + fold_offset)
    oof_c = np.zeros(len(cal_samples))
    oof_n = np.zeros(len(cal_samples))
    oof_p = np.zeros(len(cal_samples))
    for f_idx, h_idx in cal_kf.split(cal_samples):
        pfn_k = fit_ef_fn([cal_samples[i] for i in f_idx])
        c, sn, sp = pfn_k([cal_samples[i] for i in h_idx])
        oof_c[h_idx] = c
        oof_n[h_idx] = sn
        oof_p[h_idx] = sp

    tau_oof = np.maximum((oof_c - cy) / oof_n, (cy - oof_c) / oof_p)
    q_oof = float(np.quantile(tau_oof, 0.906, method="linear")) if scale_cal <= 1.0 else conformal_quantile(tau_oof, alpha)

    pfn_all = fit_ef_fn(cal_samples)
    tc, tn, tp = pfn_all(test_samples)
    lo = np.clip(tc - q_oof * tn, 0.0, 100.0)
    hi = np.clip(tc + q_oof * tp, lo, 100.0)

    return {"lower": lo, "upper": hi, "lambda": q_oof}


def calibrate_crc_fs_l_from_scores(
    raw_scores: np.ndarray,
    cal_sigmas: np.ndarray,
    alpha: float,
    mode: str = "coverage",
) -> dict:
    """Legacy score-level CRC-FS helper."""
    R = np.asarray(raw_scores, dtype=float)
    sig = np.maximum(np.asarray(cal_sigmas, dtype=float), 1e-8)
    n = len(R)
    s = R / sig
    tau = float(np.std(s)) if np.std(s) > 1e-12 else 1.0

    def loss_at(lam: float) -> np.ndarray:
        return np.minimum(1.0, np.maximum(0.0, s - lam) / tau)

    s_sorted = np.sort(s)
    q_idx = min(max(int(np.ceil((n + 1) * (1.0 - alpha))) - 1, 0), n - 1)
    quantile_s = float(s_sorted[q_idx])
    lambda_hat = quantile_s
    return {
        "lambda_hat": lambda_hat,
        "risk": float(np.mean(loss_at(lambda_hat))),
        "tau": tau,
        "cal_coverage": float(np.mean(s <= lambda_hat)),
        "quantile_s": quantile_s,
        "alpha": alpha,
        "n_cal": n,
        "mode": mode,
        "method": "CRC-FS-L",
    }


def predict_interval_crc_fs_l(
    probs: torch.Tensor,
    q_hat: float,
    sigma: float,
    median_sigma: float,
    target_class: int,
    metric_fn: Callable,
) -> Tuple[float, float]:
    """Legacy single-sample perturbation helper."""
    beta = min(float(q_hat) * float(max(sigma, 1e-8)), 20.0)
    m_neg = compute_metric_for_b(probs, -beta, target_class, metric_fn)
    m_pos = compute_metric_for_b(probs, beta, target_class, metric_fn)
    return float(min(m_neg, m_pos)), float(max(m_neg, m_pos))
