#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vérification de fermeture d'un circuit à partir d'un CSV, via sélection de fichier (tkinter).

Critère principal :
    ∫_0^L cos(theta(s)) ds ≈ 0
    ∫_0^L sin(theta(s)) ds ≈ 0

Colonnes attendues par défaut :
    s_m        : abscisse curviligne (m)
    theta_rad  : angle (radian)
Optionnel :
    x_m, y_m   : pour vérifier la fermeture géométrique
"""

import numpy as np
import pandas as pd
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox


# --- paramètres de tolérance (ajuste comme tu veux) ---
INTEGRAL_TOL = 0.1   # m, sur ∫cosθ ds et ∫sinθ ds
GEOM_TOL = 5.0       # m, distance départ–arrivée max
S_COL = "s_m"
THETA_COL = "theta_rad"


def check_closure_integral(s, theta, tol=0.1):
    """Vérifie les intégrales de cos(theta) et sin(theta) sur [0, L]."""
    # Tri au cas où s ne serait pas croissant
    idx = np.argsort(s)
    s = s[idx]
    theta = theta[idx]

    if np.any(np.diff(s) <= 0):
        raise ValueError("La colonne s_m doit être strictement croissante (ou au moins strictement ordonnée).")

    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    # Intégration numérique (méthode des trapèzes)
    I_x = np.trapz(cos_th, s)
    I_y = np.trapz(sin_th, s)

    ok_x = abs(I_x) <= tol
    ok_y = abs(I_y) <= tol

    return I_x, I_y, ok_x and ok_y


def check_closure_geometry(x, y, tol=1.0):
    """Vérifie la fermeture géométrique à partir de x,y (m)."""
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    err = float(np.hypot(dx, dy))
    return err, (err <= tol)


def main():
    # Fenêtre Tkinter minimale
    root = tk.Tk()
    root.withdraw()

    csv_file = filedialog.askopenfilename(
        title="Choisir un fichier CSV de circuit",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not csv_file:
        messagebox.showinfo("Annulé", "Aucun fichier CSV sélectionné.")
        return

    csv_path = Path(csv_file)

    try:
        df = pd.read_csv(csv_path)

        if S_COL not in df.columns or THETA_COL not in df.columns:
            raise ValueError(
                f"Colonnes requises non trouvées.\n"
                f"Il faut au minimum '{S_COL}' et '{THETA_COL}'.\n"
                f"Colonnes disponibles : {list(df.columns)}"
            )

        s = df[S_COL].to_numpy(dtype=float)
        theta = df[THETA_COL].to_numpy(dtype=float)

        I_x, I_y, ok_int = check_closure_integral(s, theta, tol=INTEGRAL_TOL)

        msg_lines = []
        msg_lines.append(f"Fichier : {csv_path.name}")
        msg_lines.append("")
        msg_lines.append("=== Critère intégral ===")
        msg_lines.append(f"Longueur L ≈ {s.max():.3f} m")
        msg_lines.append(f"∫ cos(theta) ds = {I_x:.6f} m (tolérance {INTEGRAL_TOL} m)")
        msg_lines.append(f"∫ sin(theta) ds = {I_y:.6f} m (tolérance {INTEGRAL_TOL} m)")
        msg_lines.append(f"Circuit fermé (intégral) : {'OUI' if ok_int else 'NON'}")

        if "x_m" in df.columns and "y_m" in df.columns:
            x = df["x_m"].to_numpy(dtype=float)
            y = df["y_m"].to_numpy(dtype=float)
            err_geom, ok_geom = check_closure_geometry(x, y, tol=GEOM_TOL)

            msg_lines.append("")
            msg_lines.append("=== Critère géométrique (x_m, y_m) ===")
            msg_lines.append(f"Erreur de fermeture : {err_geom:.6f} m (tolérance {GEOM_TOL} m)")
            msg_lines.append(f"Circuit fermé (géométrique) : {'OUI' if ok_geom else 'NON'}")
        else:
            msg_lines.append("")
            msg_lines.append("Pas de colonnes x_m / y_m : critère géométrique non vérifié.")

        full_msg = "\n".join(msg_lines)

        if not ok_int:
            messagebox.showerror("Circuit NON fermé", full_msg)
        else:
            messagebox.showinfo("Circuit fermé (intégral OK)", full_msg)

    except Exception as e:
        messagebox.showerror("Erreur", str(e))


if __name__ == "__main__":
    main()
