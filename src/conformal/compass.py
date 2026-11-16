import numpy as np
import torch
from typing import Tuple, Callable, Union

def perturb_probabilities(probs: torch.Tensor, b: float, target_class: int = 3) -> torch.Tensor:
    """
    Applies a uniform logit shift (COMPASS-L) to the predicted probabilities.

    logit'(x) = logit(x) + b   (for target_class only, in logit space)
    Equivalent to: probs'[target] = probs[target] * exp(b), then renormalize
    """
    eps = 1e-10
    probs_safe = torch.clamp(probs, eps, 1.0 - eps)

    log_p = torch.log(probs_safe)
    log_p[target_class] = log_p[target_class] + float(b)
    return torch.softmax(log_p, dim=0)


def compute_metric_for_b(probs: torch.Tensor, b: float, target_class: int, metric_fn: Callable) -> float:
    """Applies logit shift b, argmax threshold, and computes the clinical metric."""
    p_perturbed = perturb_probabilities(probs, b, target_class)
    mask = torch.argmax(p_perturbed, dim=0)
    binary_mask = (mask == target_class).to(torch.uint8)
    return metric_fn(binary_mask)


# ═══════════════════════════════════════════════════
#  COMPASS-L: Binary mask + binary search
# ═══════════════════════════════════════════════════

def compass_l_score(probs: torch.Tensor, y_true: float, target_class: int,
                    metric_fn: Callable, b_max: float = 15.0, n_iter: int = 16) -> float:
    """
    COMPASS-L score using binary mask + binary search.

    R_i = min{ beta >= 0 : y_true in [m(-beta), m(+beta)] }
    where m(b) = metric_fn(argmax(perturb(probs, b))).
    """
    m0 = compute_metric_for_b(probs, 0.0, target_class, metric_fn)
    if abs(m0 - y_true) < 1e-9:
        return 0.0

    lo, hi = 0.0, b_max
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        m_neg = compute_metric_for_b(probs, -mid, target_class, metric_fn)
        m_pos = compute_metric_for_b(probs,  mid, target_class, metric_fn)
        if min(m_neg, m_pos) <= y_true <= max(m_neg, m_pos):
            hi = mid
        else:
            lo = mid
    return float(hi)


def signed_compass_score(
    probs: torch.Tensor,
    y_true: float,
    target_class: int,
    metric_fn: Callable,
    direction: Union[torch.Tensor, None] = None,
    b_max: float = 20.0,
    n_iter: int = 18,
) -> float:
    """
    Signed feature-space perturbation score r_i in [-b_max, +b_max].

    Because m(b) is monotonically non-decreasing in b for target_class logit shifts
    (or non-negative target-weighted directions), y_true in [m(b_lo), m(b_hi)]
    is equivalent to b_lo <= r_i <= b_hi up to voxel discretization.
    """
    eps = 1e-10
    log_p = torch.log(torch.clamp(probs, eps, 1.0 - eps))

    def eval_b(b: float) -> float:
        if direction is None:
            lp = log_p.clone()
            lp[target_class] = lp[target_class] + float(b)
        else:
            lp = log_p + float(b) * direction
        mask = (torch.argmax(lp, dim=0) == target_class).to(torch.uint8)
        return float(metric_fn(mask))

    m0 = eval_b(0.0)
    if abs(m0 - y_true) < 1e-9:
        return 0.0

    if y_true > m0:
        lo, hi = 0.0, b_max
        if eval_b(hi) < y_true:
            return float(b_max)
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            if eval_b(mid) >= y_true:
                hi = mid
            else:
                lo = mid
        return float(hi)

    lo, hi = 0.0, b_max
    if eval_b(-hi) > y_true:
        return float(-b_max)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if eval_b(-mid) <= y_true:
            hi = mid
        else:
            lo = mid
    return float(-hi)


def map_interval_to_mask_perturbation(
    probs: torch.Tensor,
    target_lo: float,
    target_hi: float,
    target_class: int,
    metric_fn: Callable,
    direction: Union[torch.Tensor, None] = None,
    b_max: float = 30.0,
    n_iter: int = 18,
) -> Tuple[float, float, float, float]:
    """
    Map target interval [target_lo, target_hi] onto realizable 3D segmentation masks
    [m(b_lo), m(b_hi)] along the feature/logit perturbation manifold, preserving
    enclosure m(b_lo) <= target_lo and m(b_hi) >= target_hi whenever reachable.

    Returns:
        (lo_realized, hi_realized, b_lo, b_hi)
    """
    eps = 1e-10
    log_p = torch.log(torch.clamp(probs, eps, 1.0 - eps))

    def eval_b(b: float) -> float:
        if direction is None:
            lp = log_p.clone()
            lp[target_class] = lp[target_class] + float(b)
        else:
            lp = log_p + float(b) * direction
        mask = (torch.argmax(lp, dim=0) == target_class).to(torch.uint8)
        return float(metric_fn(mask))

    m0 = eval_b(0.0)

    def find_endpoint(v_targ: float, side: str) -> Tuple[float, float]:
        if side == "lo":
            if v_targ <= 0.0:
                return -b_max, 0.0
            if m0 <= v_targ:
                lo, hi = 0.0, b_max
                v_hi = eval_b(hi)
                if v_hi <= v_targ:
                    return hi, v_hi
                for _ in range(n_iter):
                    mid = 0.5 * (lo + hi)
                    if eval_b(mid) <= v_targ:
                        lo = mid
                    else:
                        hi = mid
                return lo, eval_b(lo)
            lo, hi = -b_max, 0.0
            if eval_b(lo) > v_targ:
                return lo, float(v_targ)
            for _ in range(n_iter):
                mid = 0.5 * (lo + hi)
                if eval_b(mid) <= v_targ:
                    lo = mid
                else:
                    hi = mid
            return lo, eval_b(lo)

        if m0 >= v_targ:
            lo, hi = -b_max, 0.0
            v_lo = eval_b(lo)
            if v_lo >= v_targ:
                return lo, v_lo
            for _ in range(n_iter):
                mid = 0.5 * (lo + hi)
                if eval_b(mid) >= v_targ:
                    hi = mid
                else:
                    lo = mid
            return hi, eval_b(hi)
        lo, hi = 0.0, b_max
        if eval_b(hi) < v_targ:
            return hi, float(v_targ)
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            if eval_b(mid) >= v_targ:
                hi = mid
            else:
                lo = mid
        return hi, eval_b(hi)

    b_lo, v_lo = find_endpoint(float(target_lo), "lo")
    b_hi, v_hi = find_endpoint(float(target_hi), "hi")
    if v_lo <= v_hi:
        return v_lo, v_hi, float(b_lo), float(b_hi)
    return v_hi, v_lo, float(b_hi), float(b_lo)


def calibrate_compass(scores: Union[np.ndarray, list], alpha: float) -> float:
    """Calibrate COMPASS scores using split conformal quantile."""
    from src.conformal.split_conformal import conformal_quantile
    scores_arr = np.array(scores)
    return conformal_quantile(scores_arr, alpha)


def predict_interval_compass(probs: torch.Tensor, beta_hat: float,
                              target_class: int, metric_fn: Callable) -> Tuple[float, float]:
    """Prediction interval dung binary mask + metric_fn."""
    m_neg = compute_metric_for_b(probs, -beta_hat, target_class, metric_fn)
    m_pos = compute_metric_for_b(probs,  beta_hat, target_class, metric_fn)
    return float(min(m_neg, m_pos)), float(max(m_neg, m_pos))
