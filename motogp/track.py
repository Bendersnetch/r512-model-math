# motogp/track.py

import numpy as np
import pandas as pd

from .physics import G, LEAN_MAX_DEG, V_MAX_STRAIGHT_KMH


def load_track(csv_path: str):
    """
    Lit le circuit depuis un CSV au format :
    s_m,x_m,y_m,theta_rad,w_m
    """
    df = pd.read_csv(csv_path)
    s = df["s_m"].to_numpy()
    x = df["x_m"].to_numpy()
    y = df["y_m"].to_numpy()
    theta = df["theta_rad"].to_numpy()
    w = df["w_m"].to_numpy()
    return {"s": s, "x": x, "y": y, "theta": theta, "w": w}


def compute_curvature(s: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Calcule la courbure κ(s) ~ dθ/ds."""
    theta_u = np.unwrap(theta)
    N = len(s)
    kappa = np.zeros(N)

    ds_center = s[2:] - s[:-2]
    dtheta_center = theta_u[2:] - theta_u[:-2]
    kappa[1:-1] = dtheta_center / ds_center

    kappa[0] = (theta_u[1] - theta_u[0]) / (s[1] - s[0])
    kappa[-1] = (theta_u[-1] - theta_u[-2]) / (s[-1] - s[-2])
    return kappa


def compute_vmax_lean(
    kappa: np.ndarray,
    phi_max_deg: float = LEAN_MAX_DEG,
    vmax_straight_kmh: float = V_MAX_STRAIGHT_KMH,
) -> np.ndarray:
    """Vitesse max autorisée par l'angle de prise d'angle."""
    phi_max = np.deg2rad(phi_max_deg)
    vmax_straight = vmax_straight_kmh / 3.6

    k_abs = np.abs(kappa)
    v_max = np.full_like(k_abs, vmax_straight, dtype=float)

    eps = 1e-6
    mask = k_abs > eps
    v_max[mask] = np.sqrt(G * np.tan(phi_max) / k_abs[mask])
    return v_max


def backward_braking_profile(
    s: np.ndarray,
    v_local_max: np.ndarray,
    a_brake_max: float = 15.0,
) -> np.ndarray:
    """Passe arrière pour imposer la limite de freinage."""
    N = len(s)
    v_profile = v_local_max.copy()

    for i in range(N - 2, -1, -1):
        ds = s[i + 1] - s[i]
        v_prev_max_sq = v_profile[i + 1] ** 2 + 2 * a_brake_max * ds
        v_prev_max = np.sqrt(max(v_prev_max_sq, 0.0))
        v_profile[i] = min(v_profile[i], v_prev_max)

    return v_profile


def precompute_track_center(csv_path: str):
    """
    Prépare le track de la ligne centrale :
      - charge CSV
      - calcule la courbure
      - calcule un profil de vitesse max v_profile (utilisable en ref)
    """
    track = load_track(csv_path)
    track["kappa"] = compute_curvature(track["s"], track["theta"])
    v_max_lean = compute_vmax_lean(track["kappa"])
    track["v_max_lean"] = v_max_lean
    track["v_profile"] = backward_braking_profile(
        track["s"],
        v_max_lean,
        a_brake_max=15.0,
    )
    return track
