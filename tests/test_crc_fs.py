"""
tests/test_crc_fs.py

Unit tests for CRC-FS (Conformal Risk Control in Feature Space: CRC-FS-L & CRC-FS-J)
mathematical guarantees and invariance properties.
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal.compass import map_interval_to_mask_perturbation, signed_compass_score
from src.conformal.crc_fs import (
    bounded_severity_risk,
    calibrate_and_predict_crc_fs,
    extract_directional_features,
    optimal_asym_shift_and_ratio,
)


def _make_synthetic_samples(n_patients: int = 120, seed: int = 7) -> list:
    rng = np.random.default_rng(seed)
    samples = []
    for p in range(n_patients):
        pid = f"patient{p:03d}"
        grp = ["DCM", "HCM", "MINF", "NOR", "RV"][p % 5]
        probs = np.zeros((4, 6, 16, 16), dtype=np.float32)
        probs[0] = 0.85
        probs[3] = 0.05
        probs[1] = 0.05
        probs[2] = 0.05
        probs[3, 1:5, 4:12, 4:12] = 0.96
        probs[0, 1:5, 4:12, 4:12] = 0.02
        probs[1, 1:5, 4:12, 4:12] = 0.01
        probs[2, 1:5, 4:12, 4:12] = 0.01
        unc_strength = float(rng.uniform(0.1, 0.45))
        probs[3, 0, 4:12, 4:12] = 0.45 + unc_strength * 0.2
        probs[0, 0, 4:12, 4:12] = 1.0 - probs[3, 0, 4:12, 4:12] - 0.02
        probs /= probs.sum(axis=0, keepdims=True)

        vvol = 0.05
        feats = extract_directional_features(probs, target_class=3, voxel_vol=vvol)
        noise = rng.normal(0.0, 0.8 + 0.7 * feats["max_end_unc"])
        if noise < 0:
            noise *= 1.8
        y_true = max(0.0, feats["y_hard"] + 0.8 + noise)

        probs_t = torch.tensor(probs, dtype=torch.float32)
        mfn = lambda m, v=vvol: float(m.sum().item()) * v
        samples.append({
            "patient_id": pid,
            "group": grp,
            "y_true": y_true,
            "probs": probs_t,
            "metric_fn": mfn,
            **feats,
        })
    return samples


def test_extract_directional_features_nonnegative():
    samples = _make_synthetic_samples(10)
    for s in samples:
        for k in ("y_hard", "y_soft", "mass_below", "mass_above", "sens", "sqrt_sens", "max_end_unc", "rim_p0"):
            assert s[k] >= 0.0, f"{k} must be non-negative"
        assert s["y_hard"] > 0.0


def test_signed_score_and_manifold_enclosure():
    s = _make_synthetic_samples(1)[0]
    lo_t = max(0.0, s["y_hard"] - 2.0)
    hi_t = s["y_hard"] + 2.5
    lo_r, hi_r, b_lo, b_hi = map_interval_to_mask_perturbation(
        s["probs"], lo_t, hi_t, target_class=3, metric_fn=s["metric_fn"]
    )
    assert lo_r <= lo_t + 1e-6
    assert hi_r >= hi_t - 1e-6
    assert b_lo <= b_hi

    r_score = signed_compass_score(s["probs"], s["y_hard"] + 1.0, target_class=3, metric_fn=s["metric_fn"])
    assert r_score >= 0.0


def test_crc_fs_coverage_and_risk_guarantee():
    pool = _make_synthetic_samples(500, seed=555)
    covs_l, covs_j, sevs_crc = [], [], []
    alpha = 0.10
    rng = np.random.default_rng(77)
    for s_idx in range(10):
        perm = rng.permutation(len(pool))
        cal = [pool[i] for i in perm[:200]]
        tst = [pool[i] for i in perm[200:]]
        ty = np.array([s["y_true"] for s in tst])

        res_l = calibrate_and_predict_crc_fs(cal, tst, alpha=alpha, mode="crc_fs_l", seed=42 + s_idx)
        covs_l.append(float(np.mean((ty >= res_l["lower"]) & (ty <= res_l["upper"]))))
        assert np.all(res_l["upper"] >= res_l["lower"])

        res_j = calibrate_and_predict_crc_fs(cal, tst, alpha=alpha, mode="crc_fs_j", seed=42 + s_idx)
        covs_j.append(float(np.mean((ty >= res_j["lower"]) & (ty <= res_j["upper"]))))

        res_crc = calibrate_and_predict_crc_fs(cal, tst, alpha=alpha, mode="risk_crc", seed=42 + s_idx)
        sevs_crc.append(bounded_severity_risk(ty, res_crc["lower"], res_crc["upper"], res_crc["tau_scale"]))

    assert np.mean(covs_l) >= 0.89, f"Mean CRC-FS-L coverage too low: {np.mean(covs_l)}"
    assert np.mean(covs_j) >= 0.89, f"Mean CRC-FS-J coverage too low: {np.mean(covs_j)}"
    assert np.mean(sevs_crc) <= alpha + 0.01, f"Mean CRC-FS severity risk too high: {np.mean(sevs_crc)}"


def test_optimal_asym_shift_and_ratio_prefers_skew():
    rng = np.random.default_rng(9)
    res = rng.exponential(scale=3.0, size=100) - 0.5
    sn = np.ones(100)
    sp = np.ones(100)
    d, rho = optimal_asym_shift_and_ratio(res, sn, sp, alpha=0.1)
    assert rho >= 1.0 or d > 0.0, "Should shift/skew toward positive tail"


if __name__ == "__main__":
    test_extract_directional_features_nonnegative()
    test_signed_score_and_manifold_enclosure()
    test_crc_fs_coverage_and_risk_guarantee()
    test_optimal_asym_shift_and_ratio_prefers_skew()
    print("ALL UNIT TESTS PASSED!")
