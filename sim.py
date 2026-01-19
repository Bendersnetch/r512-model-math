#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import tkinter as tk
from tkinter import filedialog

# ==========================
#  Constantes globales
# ==========================

G = 9.81  # gravité (m/s²)

# --- Paramètres MotoGP (ton modèle) ---
P = 220_000        # W - puissance
m = 240            # kg - masse moto + pilote
k_aero = 0.18      # coef aéro
Fr = 35.32         # N - frottements roul.
amax = 12.0        # m/s² (1.2 g) - limite d'accel moteur


# ==========================
#  Modèle MotoGP
# ==========================

def accel(v, throttle):
    """
    Retourne l'accélération réelle d'une MotoGP.

    v : vitesse actuelle (m/s)
    throttle : % poignée (0 à 1)
    """
    if v < 0.1:
        v = 0.1

    a_motor = (P / v - k_aero * v * v - Fr) / m
    a_real = throttle * min(a_motor, amax)
    return max(a_real, 0.0)


def update_speed(v, throttle, dt):
    """Retourne la nouvelle vitesse après dt secondes."""
    a = accel(v, throttle)
    return v + a * dt


# ==========================
#  Outils circuit
# ==========================

def load_track(csv_path):
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


def compute_curvature(s, theta):
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


def compute_vmax_lean(kappa, phi_max_deg=67.0, vmax_straight_kmh=340.0):
    """Vitesse max autorisée par l'angle de prise d'angle."""
    phi_max = np.deg2rad(phi_max_deg)
    vmax_straight = vmax_straight_kmh / 3.6

    k_abs = np.abs(kappa)
    v_max = np.full_like(k_abs, vmax_straight, dtype=float)

    eps = 1e-6
    mask = k_abs > eps
    v_max[mask] = np.sqrt(G * np.tan(phi_max) / k_abs[mask])
    return v_max


def backward_braking_profile(s, v_local_max, a_brake_max=15.0):
    """Passe arrière pour imposer la limite de freinage."""
    N = len(s)
    v_profile = v_local_max.copy()

    for i in range(N - 2, -1, -1):
        ds = s[i + 1] - s[i]
        v_prev_max_sq = v_profile[i + 1] ** 2 + 2 * a_brake_max * ds
        v_prev_max = np.sqrt(max(v_prev_max_sq, 0.0))
        v_profile[i] = min(v_profile[i], v_prev_max)

    return v_profile


def add_speed_profile(track,
                      phi_max_deg=67.0,
                      vmax_straight_kmh=340.0,
                      a_brake_max=15.0):
    """
    À partir d'un track avec s, theta, kappa :
    ajoute v_max_lean et v_profile (profil de vitesse max).
    """
    kappa = track["kappa"]
    s = track["s"]

    v_max_lean = compute_vmax_lean(kappa, phi_max_deg, vmax_straight_kmh)
    v_profile = backward_braking_profile(s, v_max_lean, a_brake_max=a_brake_max)

    track["v_max_lean"] = v_max_lean
    track["v_profile"] = v_profile
    return track


def precompute_track_center(csv_path):
    """
    Prépare le track de la ligne centrale :
      - charge CSV
      - calcule la courbure
      - ajoute v_profile centre (utile pour comparer)
    """
    track = load_track(csv_path)
    track["kappa"] = compute_curvature(track["s"], track["theta"])
    track = add_speed_profile(track)
    return track


# ==========================
#  Trajectoire optimisée (prise en compte largeur)
# ==========================

def compute_racing_line_from_center(track_center,
                                    offset_ratio=0.9,
                                    smooth_window=31):
    """
    Génère une 'racing line' à l'intérieur de la largeur du circuit.

    Approche heuristique :
      - dans les virages serrés (|kappa| grand), on se décale vers l'extérieur
      - décalage max = offset_ratio * w/2
      - on lisse le décalage sur s
    """
    s = track_center["s"]
    x_c = track_center["x"]
    y_c = track_center["y"]
    theta_c = track_center["theta"]
    w = track_center["w"]
    kappa_c = track_center["kappa"]

    nx = -np.sin(theta_c)
    ny =  np.cos(theta_c)

    d_max = 0.5 * w * offset_ratio       # en m

    k_abs = np.abs(kappa_c)
    kappa_ref = np.percentile(k_abs, 90)
    if kappa_ref < 1e-6:
        kappa_ref = np.max(k_abs) + 1e-6

    weight = np.clip(k_abs / (kappa_ref + 1e-9), 0.0, 1.0)

    direction_sign = -np.sign(kappa_c)
    d_raw = d_max * weight * direction_sign

    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        d = np.convolve(d_raw, kernel, mode="same")
    else:
        d = d_raw

    x_r = x_c + d * nx
    y_r = y_c + d * ny

    dx_ds = np.gradient(x_r, s)
    dy_ds = np.gradient(y_r, s)
    theta_r = np.arctan2(dy_ds, dx_ds)

    kappa_r = compute_curvature(s, theta_r)

    track_r = {
        "s": s,
        "x": x_r,
        "y": y_r,
        "theta": theta_r,
        "w": w,
        "kappa": kappa_r,
        "x_center": x_c,
        "y_center": y_c,
    }

    track_r = add_speed_profile(track_r)
    return track_r


# ==========================
#  Physique : angle & friction circle
# ==========================

def lean_angle_deg(v, kappa):
    """Angle de prise d’angle en degrés."""
    a_lat = (v ** 2) * kappa
    phi = np.arctan2(abs(a_lat), G)
    return np.rad2deg(phi)


def simulate_lap_with_friction(
    track,
    dt=0.01,
    v0=0.0,
    accel_margin=1.0,
    kp_throttle=0.15,
    brake_decel=15.0,
    phi_max_deg=67.0,
    mu_tire=1.5,
):
    """Simule un tour avec modèle MotoGP + cercle de friction sur 'track'."""

    s_track = track["s"]
    kappa = track["kappa"]
    v_profile = track["v_profile"]

    s_total = s_track[-1]
    v = v0
    s_sim = s_track[0]

    history = []
    t = 0.0

    while s_sim < s_total:
        idx = np.searchsorted(s_track, s_sim, side="right") - 1
        idx = max(0, min(idx, len(s_track) - 1))

        v_target = v_profile[idx]
        k = kappa[idx]

        if v < v_target - accel_margin:
            throttle_cmd = np.clip(kp_throttle * (v_target - v), 0.0, 1.0)
            a_long_desired = accel(v, throttle_cmd)
            brake_level = 0.0
        elif v > v_target + accel_margin:
            throttle_cmd = 0.0
            a_long_desired = -brake_decel
            brake_level = 1.0
        else:
            throttle_cmd = 0.0
            a_long_desired = 0.0
            brake_level = 0.0

        a_lat = v ** 2 * k
        a_lat_abs = abs(a_lat)
        a_total_max = mu_tire * G

        if a_lat_abs >= a_total_max:
            a_long = 0.0
            throttle_eff = 0.0
            brake_eff = 0.0
        else:
            a_long_max_mag = np.sqrt(max(a_total_max ** 2 - a_lat_abs ** 2, 0.0))

            if a_long_desired > 0:
                a_long = min(a_long_desired, a_long_max_mag)
                if a_long_desired > 1e-3:
                    throttle_eff = throttle_cmd * (a_long / a_long_desired)
                else:
                    throttle_eff = 0.0
                brake_eff = 0.0
            elif a_long_desired < 0:
                a_long = max(a_long_desired, -a_long_max_mag, -brake_decel)
                throttle_eff = 0.0
                brake_eff = min(abs(a_long) / brake_decel, 1.0)
            else:
                a_long = 0.0
                throttle_eff = 0.0
                brake_eff = 0.0

        if a_long >= 0:
            v = update_speed(v, throttle_eff, dt)
        else:
            v = max(v + a_long * dt, 0.0)

        v = min(v, v_target * 1.05)

        s_sim += v * dt
        t += dt

        phi_deg = lean_angle_deg(v, k)
        lean_safe = (phi_deg <= phi_max_deg + 1e-3)

        a_long_abs = abs(a_long)
        denom = G * mu_tire
        if denom > 1e-6:
            mu_usage = np.sqrt(a_lat_abs ** 2 + a_long_abs ** 2) / denom
        else:
            mu_usage = 0.0

        history.append({
            "t": t,
            "s": s_sim,
            "idx": idx,
            "v_mps": v,
            "v_kmh": v * 3.6,
            "v_target_mps": v_target,
            "v_target_kmh": v_target * 3.6,
            "throttle_cmd": throttle_cmd,
            "throttle_eff": throttle_eff,
            "brake_cmd": brake_level,
            "brake_eff": brake_eff,
            "a_long": a_long,
            "a_lat": a_lat,
            "mu_usage": mu_usage,
            "kappa": k,
            "lean_deg": phi_deg,
            "lean_safe": lean_safe,
        })

    print(f"Nb de frames simulées : {len(history)} (durée ~ {history[-1]['t']:.2f} s)")
    return history


# ==========================
#  Animation / visualisation
# ==========================

def animate_lap(track, history, interval_ms=20, frame_step=1):
    """
    Prépare l’animation (ne fait PAS plt.show()).
    frame_step : nombre de frames de simulation sautées
                 (1 = tout, 2 = 1 sur 2, 5 = 1 sur 5 → plus rapide)
    """

    x_track = track["x"]
    y_track = track["y"]

    t_arr       = np.array([h["t"] for h in history])
    v_kmh_arr   = np.array([h["v_kmh"] for h in history])
    th_arr      = np.array([h["throttle_eff"] for h in history])
    br_arr      = np.array([h["brake_eff"] for h in history])
    lean_arr    = np.array([h["lean_deg"] for h in history])
    mu_usage    = np.array([h["mu_usage"] for h in history])
    idx_arr     = np.array([h["idx"] for h in history])

    # On construit la liste de frames à afficher (on en saute pour accélérer)
    frames_idx = list(range(0, len(history), max(1, frame_step)))

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])

    ax_track = fig.add_subplot(gs[:, 0])

    if "x_center" in track:
        ax_track.plot(track["x_center"], track["y_center"],
                      linewidth=1, color="lightblue", label="Centre")
        ax_track.plot(x_track, y_track, linewidth=2, color="blue", label="Racing line")
        ax_track.legend(loc="lower left")
    else:
        ax_track.plot(x_track, y_track, linewidth=1)

    ax_track.set_aspect('equal', 'box')
    ax_track.set_title("Moto sur le circuit")
    bike_point, = ax_track.plot([], [], marker='o', markersize=8, color="orange")

    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis('off')
    text_info = ax_info.text(
        0.0, 1.0,
        "",
        transform=ax_info.transAxes,
        va="top",
        fontsize=10,
    )

    ax_bars = fig.add_subplot(gs[1, 1])
    categories = ["Throttle", "Brake", "Grip"]
    values = [0.0, 0.0, 0.0]
    bar_container = ax_bars.bar(categories, values)
    ax_bars.set_ylim(0, 1.0)
    ax_bars.set_ylabel("Pourcentage")
    ax_bars.set_title("Commandes & utilisation du grip")

    margin = 10.0
    ax_track.set_xlim(x_track.min() - margin, x_track.max() + margin)
    ax_track.set_ylim(y_track.min() - margin, y_track.max() + margin)

    def update(frame_i):
        # frame_i est un index dans frames_idx
        idx_hist = frames_idx[frame_i]
        i = idx_hist

        idx = idx_arr[i]
        x = x_track[idx]
        y = y_track[idx]
        bike_point.set_data([x], [y])

        v_kmh   = v_kmh_arr[i]
        th      = th_arr[i]
        br      = br_arr[i]
        lean    = lean_arr[i]
        mu_used = mu_usage[i]

        text_str = (
            f"Temps : {t_arr[i]:6.2f} s\n"
            f"Vitesse : {v_kmh:6.1f} km/h\n"
            f"Accélérateur : {th*100:5.1f} %\n"
            f"Frein : {br*100:5.1f} %\n"
            f"Inclinaison : {lean:5.1f}°\n"
            f"Grip utilisé : {mu_used*100:5.1f} %"
        )
        text_info.set_text(text_str)

        bar_container[0].set_height(th)
        bar_container[1].set_height(br)
        bar_container[2].set_height(mu_used)

        return bike_point, text_info, *bar_container

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames_idx),
        interval=interval_ms,
        blit=False,
        repeat=False
    )

    return fig, anim


# ==========================
#  Main avec Tkinter
# ==========================

if __name__ == "__main__":
    # --- 1) Choix du fichier CSV ---
    root_file = tk.Tk()
    root_file.withdraw()

    csv_path = filedialog.askopenfilename(
        title="Sélectionne le fichier CSV du circuit",
        filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")]
    )

    root_file.destroy()

    if not csv_path:
        print("Aucun fichier sélectionné, fin du programme.")
        raise SystemExit

    print(f"Fichier sélectionné : {csv_path}")

    # --- 2) Petite fenêtre Tkinter pour choisir la vitesse d'animation ---
    speed_choice = {"factor": 1}  # valeur par défaut

    def launch_simulation():
        speed_choice["factor"] = speed_var.get()
        speed_window.destroy()

    speed_window = tk.Tk()
    speed_window.title("Paramètres de simulation")

    tk.Label(speed_window, text="Vitesse d'animation :", font=("Arial", 11, "bold")).pack(pady=5)

    speed_var = tk.IntVar(value=1)
    for text, val in [("x1 (normal)", 1),
                      ("x2", 2),
                      ("x5", 5),
                      ("x10", 10)]:
        tk.Radiobutton(speed_window, text=text, variable=speed_var, value=val)\
            .pack(anchor="w", padx=10)

    tk.Button(speed_window, text="Lancer la simulation", command=launch_simulation)\
        .pack(pady=10)

    speed_window.mainloop()

    speed_factor = speed_choice["factor"]
    print(f"Vitesse d'animation choisie : x{speed_factor}")

    # --- 3) Prétraitement + racing line ---
    print("Prétraitement du circuit (ligne centrale)...")
    track_center = precompute_track_center(csv_path)

    print("Calcul de la racing line en utilisant la largeur du circuit...")
    track_race = compute_racing_line_from_center(track_center,
                                                 offset_ratio=0.9,
                                                 smooth_window=31)

    # --- 4) Simulation ---
    print("Simulation du tour sur la racing line...")
    history = simulate_lap_with_friction(
        track_race,
        dt=0.01,
        v0=0.0,
        accel_margin=1.0,
        kp_throttle=0.15,
        brake_decel=15.0,
        phi_max_deg=67.0,
        mu_tire=1.6,
    )

    # --- 5) Animation (frame_step = speed_factor) ---
    print("Préparation de l'animation...")
    # interval_ms = 10 → ~100 fps théorique
    fig, anim = animate_lap(track_race, history,
                            interval_ms=10,
                            frame_step=speed_factor)

    print("Affichage...")
    plt.show()
