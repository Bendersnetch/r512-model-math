# motogp/physics.py

import numpy as np

# Gravité
G = 9.81  # m/s²

# Paramètres MotoGP tirés du modèle
P = 220_000        # W - puissance
m = 240            # kg - moto + pilote
k_aero = 0.18      # coef aéro
Fr = 35.32         # N - frottements roul.
amax = 12.0        # m/s² (1.2 g)

# Limites
LEAN_MAX_DEG = 67.0                 # angle max
LEAN_RATE_MAX_DEG_PER_S = 120.0     # roll rate max (°/s)
MU_TIRE = 1.6                       # coefficient de friction
GRIP_TOL = 1e-3                     # tolérance numérique

V_MAX_STRAIGHT_KMH = 340.0
V_MAX_STRAIGHT = V_MAX_STRAIGHT_KMH / 3.6  # m/s


def accel(v: float, throttle: float) -> float:
    """
    Accélération longitudinale moteur (sans frein), selon modèle.

    v        : vitesse (m/s)
    throttle : [0,1]
    """
    if v < 0.1:
        v = 0.1

    a_motor = (P / v - k_aero * v * v - Fr) / m
    a_real = throttle * min(a_motor, amax)
    return max(a_real, 0.0)


def lean_angle_deg(v: float, kappa: float) -> float:
    """
    Angle d'inclinaison (en degrés) pour tenir la force centripète :
        a_lat = v^2 * kappa
        tan(phi) = a_lat / g
    """
    a_lat = (v ** 2) * kappa
    phi = np.arctan2(abs(a_lat), G)
    return np.rad2deg(phi)
