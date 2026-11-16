"""
src/conformal/compass_j.py

COMPASS-J: Jacobian-based Feature Perturbation
(Cheung et al., arXiv:2509.22240, ICLR 2026 - Section 2.2)

Y tuong cot loi:
    COMPASS-L cong deu hang so beta vao toan bo logit → khong hieu qua,
    lam interval phinh to (over-coverage).

    COMPASS-J dung dao ham (Jacobian) cua metric Volume m
    theo logit truoc softmax:

        dV/d_logit[target, x] = p_target(x) * (1 - p_target(x)) * voxel_vol

    Day chinh la variance cua Bernoulli distribution tai moi voxel.
    Cac voxel gan ranh gioi quyet dinh (p_target ≈ 0.5) co gradient
    lon nhat → thuong keo theo su thay doi lon nhat trong Volume.

    Perturbation chi tap trung vao vung ranh gioi (boundary) thay
    vi toan bo anh → cung mot beta nhung Width hep hon nhieu.

    Formula:
        logit'(x) = logit(x) + beta * J_normalized(x)
        p'(x) = softmax(logit'(x))

Complexity: O(n_cal * steps) per fold
"""

import numpy as np
import torch
from typing import Callable, Tuple, Optional
from sklearn.decomposition import TruncatedSVD
from src.conformal.compass import compute_metric_for_b, perturb_probabilities


# ─────────────────────────────────────────────
#  1.  Tinh Jacobian dV/d_logit (Boundary-weighted direction)
# ─────────────────────────────────────────────

def compute_volume_jacobian(
    probs: torch.Tensor,
    target_class: int,
    voxel_vol: float,
) -> torch.Tensor:
    """
    Tinh dao ham cua Volume theo LOGIT (truoc softmax).

    Dung soft-argmax approximation:
        V(p) ≈ voxel_vol * sum_x p_target(x)

    Dao ham theo logit[target, x] (chain rule qua softmax):
        dV / d_logit[target, x] = p_target(x) * (1 - p_target(x)) * voxel_vol

    Day la phuong sai Bernoulli tai moi voxel:
        - p ≈ 0.5 (ranh gioi quyet dinh) → gradient LON → thay doi Volume nhieu
        - p ≈ 0   hoac p ≈ 1 (vung chac chan) → gradient nho → it anh huong

    Ket qua: huong perturbation chi tap trung vao RANH GIOI can organ,
    giup interval hep hon so voi COMPASS-L (cong deu toan bo).

    Args:
        probs        : (C, H, W, D) softmax probabilities tren GPU.
        target_class : class index cua co quan muc tieu (e.g., 3 = LV).
        voxel_vol    : the tich mot voxel (mL).

    Returns:
        J : Jacobian tensor, shape = (C, H, W, D), cung device voi probs.
            Chi khac 0 o channel target_class.
    """
    eps = 1e-7
    p_target = torch.clamp(probs[target_class], eps, 1.0 - eps)  # (H, W, D)

    # Variance of Bernoulli: p * (1 - p) - day la Jacobian dV/d_logit[target]
    jacobian_target = p_target * (1.0 - p_target) * voxel_vol    # (H, W, D)

    J = torch.zeros_like(probs)  # (C, H, W, D)
    J[target_class] = jacobian_target
    return J


def compute_pooled_jacobian(
    probs: torch.Tensor,
    target_class: int,
    voxel_vol: float,
) -> np.ndarray:
    """
    Tinh Jacobian RUT GON bang cach SUM theo spatial dimensions.
    Ket qua: vector [C] — LUON cung shape cho moi sample!
    Dung de PCA huong channel-wise semantic sensitivity.

    Giong nhu COMPASS paper (ICLR 2026, Section 4.1):
    \"Summing the spatial dimensions isolates the channel-wise
    semantic sensitivity, which remains consistent across samples.\"
    """
    J_full = compute_volume_jacobian(probs, target_class, voxel_vol)
    # Pool spatial dims (H, W, D) -> shape [C]
    J_pooled = J_full.sum(dim=[1, 2, 3])
    return J_pooled.cpu().numpy()  # [C] — always same shape!


# ─────────────────────────────────────────────
#  2.  PCA de tim dominant subspace V_L
# ─────────────────────────────────────────────

def compute_pca_subspace(
    jacobians: np.ndarray,
    n_components: int = 1,
) -> np.ndarray:
    """
    Tinh ma tran khong gian con (dominant subspace) tu tap Jacobians
    cua tap calibration.

    Moi Jacobian duoc flatten thanh vector roi dua vao SVD.
    Ket qua la top-k left singular vectors (principal components).

    Args:
        jacobians    : shape (n_cal, D_flat) - moi hang la 1 Jacobian da flatten.
        n_components : so chieu cua subspace.

    Returns:
        V_L : shape (n_components, D_flat) - cac principal directions.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(jacobians)
    return svd.components_   # (n_components, D_flat)


def project_jacobian_to_subspace(
    jacobian: torch.Tensor,
    V_L: np.ndarray,
) -> torch.Tensor:
    """
    Project 1 Jacobian len shared PCA subspace → perturbation direction.

    direction = V_L^T @ V_L @ J_flat
    (project len subspace roi reconstruct)

    Args:
        jacobian : (C, H, W, D) Jacobian tensor tren GPU.
        V_L      : (n_components, D_flat) PCA components tu calibration.

    Returns:
        direction: (C, H, W, D) tensor, cung device voi jacobian.
    """
    shape = jacobian.shape
    J_flat = jacobian.flatten().cpu().numpy()

    # Project: coeffs @ V_L = (V_L @ J_flat) @ V_L
    coeffs = V_L @ J_flat                       # (n_components,)
    recon = coeffs @ V_L                        # (D_flat,) — projected

    direction = torch.tensor(
        recon.reshape(shape),
        dtype=jacobian.dtype,
        device=jacobian.device,
    )
    return direction


def compute_multiclass_logit_jacobian(
    probs: torch.Tensor,
    target_class: int,
    voxel_vol: float,
) -> torch.Tensor:
    """
    Exact multi-class Jacobian of soft volume V_soft(softmax(z)) wrt pre-softmax logits z:
        dV / dz[c, x] = p_target(x) * (1_{c == target} - p_c(x)) * voxel_vol
    Shape: (C, H, W, D).
    """
    eps = 1e-7
    probs_c = torch.clamp(probs, eps, 1.0 - eps)
    p_t = probs_c[target_class]
    J_z = -probs_c * p_t.unsqueeze(0) * float(voxel_vol)
    J_z[target_class] = p_t * (1.0 - p_t) * float(voxel_vol)
    return J_z


def compute_hidden_feature_subspace(
    cal_probs: list,
    target_class: int,
    voxel_vols: list,
    head_weight: np.ndarray,
    n_components: int = 5,
) -> Tuple[np.ndarray, list]:
    """
    True intermediate-feature COMPASS-J subspace (Cheung et al., ICLR 2026):
      1. Compute exact logit Jacobian J_z(x) and map back to penultimate feature
         layer h(x) via linear head W in R^{C x D_h}: J_h = W^T J_z.
      2. Pool spatially -> R^{D_h}, compute top-L PCA basis V_L in R^{L x D_h}.
      3. Project each sample's voxel-wise J_h onto V_L and map back to logit space:
         Delta_z = (W V_L^T V_L W^T) J_z.
    """
    W_t = torch.tensor(head_weight, dtype=torch.float32, device=cal_probs[0].device)  # (C, D_h)
    pooled_jh = []
    jzs = []
    for probs, vvol in zip(cal_probs, voxel_vols):
        J_z = compute_multiclass_logit_jacobian(probs, target_class, float(vvol))
        jzs.append(J_z)
        jz_sum = J_z.reshape(J_z.shape[0], -1).sum(dim=1)  # (C,)
        jh_sum = (W_t.T @ jz_sum).detach().cpu().numpy()   # (D_h,)
        pooled_jh.append(jh_sum)

    X = np.stack(pooled_jh, axis=0)
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1]) if X.shape[0] > 1 else 1
    V_L = compute_pca_subspace(X, n_components=max(1, n_comp))
    V_L_t = torch.tensor(V_L, dtype=torch.float32, device=cal_probs[0].device)
    M_logit = W_t @ (V_L_t.T @ V_L_t) @ W_t.T  # (C, C)

    directions = []
    for J_z in jzs:
        dz = (M_logit @ J_z.reshape(J_z.shape[0], -1)).reshape(J_z.shape)
        abs_max = dz.abs().max()
        if abs_max > 1e-8:
            dz = dz / abs_max
        directions.append(dz)

    return V_L, directions


def hidden_feature_direction_for_sample(
    probs: torch.Tensor,
    target_class: int,
    voxel_vol: float,
    head_weight: np.ndarray,
    V_L: np.ndarray,
) -> torch.Tensor:
    """Compute projected COMPASS-J hidden-feature direction for a single test sample."""
    W_t = torch.tensor(head_weight, dtype=torch.float32, device=probs.device)
    V_L_t = torch.tensor(V_L, dtype=torch.float32, device=probs.device)
    M_logit = W_t @ (V_L_t.T @ V_L_t) @ W_t.T
    J_z = compute_multiclass_logit_jacobian(probs, target_class, float(voxel_vol))
    dz = (M_logit @ J_z.reshape(J_z.shape[0], -1)).reshape(J_z.shape)
    abs_max = dz.abs().max()
    if abs_max > 1e-8:
        dz = dz / abs_max
    return dz


def compute_shared_directions(
    cal_probs: list,
    target_class: int,
    voxel_vols: list,
    n_components: int = 5,
) -> Tuple[Optional[np.ndarray], list]:
    """
    Compute shared channel-wise PCA subspace from multi-class logit Jacobians
    and modulate per-voxel spatial Jacobians (fixing the previous bug that
    replaced spatial Jacobians with torch.ones_like).
    """
    pooled_jacs = []
    jacobians = []
    for i, probs in enumerate(cal_probs):
        J_full = compute_multiclass_logit_jacobian(probs, target_class, float(voxel_vols[i]))
        jacobians.append(J_full)
        pooled_jacs.append(J_full.reshape(J_full.shape[0], -1).sum(dim=1).cpu().numpy())

    X = np.stack(pooled_jacs, axis=0)
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1]) if X.shape[0] > 1 else 1
    V_L = compute_pca_subspace(X, n_components=max(1, n_comp))
    V_L_t = torch.tensor(V_L, dtype=jacobians[0].dtype, device=jacobians[0].device)
    P_c = V_L_t.T @ V_L_t  # (C, C)

    directions = []
    for J in jacobians:
        # Project spatial-voxel Jacobian vectors onto channel subspace (preserves spatial boundary weighting!)
        d = (P_c @ J.reshape(J.shape[0], -1)).reshape(J.shape)
        abs_max = d.abs().max()
        if abs_max > 1e-8:
            d = d / abs_max
        directions.append(d)

    return V_L, directions


# ─────────────────────────────────────────────
#  3.  Perturb theo huong Jacobian (COMPASS-J)
# ─────────────────────────────────────────────

def perturb_jacobian_direction(
    probs: torch.Tensor,
    beta: float,
    direction: torch.Tensor,
) -> torch.Tensor:
    """
    Perturb LOGITS along Jacobian direction and apply softmax:
        logit'(x) = logit(p(x)) + beta * direction(x)
        p'(x) = softmax(logit'(x))
    """
    eps = 1e-10
    probs_safe = torch.clamp(probs, eps, 1.0 - eps)
    log_p = torch.log(probs_safe)
    log_p_perturbed = log_p + float(beta) * direction
    return torch.softmax(log_p_perturbed, dim=0)


# ─────────────────────────────────────────────
#  4.  COMPASS-J Score cho mot sample
# ─────────────────────────────────────────────

def compass_j_score(
    probs: torch.Tensor,
    y_true: float,
    target_class: int,
    metric_fn: Callable,
    direction: torch.Tensor,
    b_max: float = 15.0,
    n_iter: int = 16,
) -> float:
    """
    COMPASS-J score using binary mask + binary search.

    R_i = min{beta >= 0 : y_true in [m(-beta*d), m(+beta*d)]}
    where m = metric_fn(argmax(perturbed_probs)).
    """
    abs_max = direction.abs().max()
    if abs_max > 1e-8:
        direction_n = direction / abs_max
    else:
        direction_n = direction

    m0 = compute_metric_for_b(probs, 0.0, target_class, metric_fn)
    if abs(m0 - y_true) < 1e-9:
        return 0.0

    lo, hi = 0.0, b_max
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        p_pos = perturb_jacobian_direction(probs,  mid, direction_n)
        p_neg = perturb_jacobian_direction(probs, -mid, direction_n)

        mask_pos = (torch.argmax(p_pos, dim=0) == target_class).to(torch.uint8)
        mask_neg = (torch.argmax(p_neg, dim=0) == target_class).to(torch.uint8)

        m_pos = metric_fn(mask_pos)
        m_neg = metric_fn(mask_neg)

        if min(m_neg, m_pos) <= y_true <= max(m_neg, m_pos):
            hi = mid
        else:
            lo = mid
    return float(hi)


# ─────────────────────────────────────────────
#  5.  Calibrate COMPASS-J
# ─────────────────────────────────────────────

def calibrate_compass_j(
    scores: np.ndarray,
    alpha: float,
) -> float:
    """
    Calibrate COMPASS-J dung split conformal quantile.
    Uses the conservative quantile from split_conformal.py.
    """
    from src.conformal.split_conformal import conformal_quantile
    scores = np.asarray(scores, dtype=float)
    return conformal_quantile(scores, alpha)


# ─────────────────────────────────────────────
#  6.  Predict interval với COMPASS-J
# ─────────────────────────────────────────────

def predict_interval_compass_j(
    probs: torch.Tensor,
    beta_hat: float,
    target_class: int,
    metric_fn: Callable,
    direction: torch.Tensor,
) -> Tuple[float, float]:

    abs_max = direction.abs().max()
    if abs_max > 1e-8:
        direction_n = direction / abs_max
    else:
        direction_n = direction

    p_pos = perturb_jacobian_direction(probs,  beta_hat, direction_n)
    p_neg = perturb_jacobian_direction(probs, -beta_hat, direction_n)

    mask_pos = (torch.argmax(p_pos, dim=0) == target_class).to(torch.uint8)
    mask_neg = (torch.argmax(p_neg, dim=0) == target_class).to(torch.uint8)

    m_pos = metric_fn(mask_pos)
    m_neg = metric_fn(mask_neg)

    return float(min(m_pos, m_neg)), float(max(m_pos, m_neg))
