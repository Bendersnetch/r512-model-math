import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import Transformer

import tkinter as tk
from tkinter import filedialog, messagebox


def read_gpx_points(gpx_path: Path) -> np.ndarray:
    root = ET.parse(gpx_path).getroot()

    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
    pts = []

    for p in root.findall(".//gpx:trkpt", ns) + root.findall(".//trkpt"):
        lat = p.get("lat")
        lon = p.get("lon")
        if lat and lon:
            pts.append((float(lat), float(lon)))

    for p in root.findall(".//gpx:rtept", ns) + root.findall(".//rtept"):
        lat = p.get("lat")
        lon = p.get("lon")
        if lat and lon:
            pts.append((float(lat), float(lon)))

    if not pts:
        raise ValueError("Aucun point GPX trouvé.")

    return np.array(pts, dtype=float)  # (N,2) lat,lon


def utm_epsg_for(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    hemi = 326 if lat >= 0 else 327
    return f"epsg:{hemi}{zone:02d}"


def project_to_meters(latlon: np.ndarray):
    lat_c = float(np.mean(latlon[:, 0]))
    lon_c = float(np.mean(latlon[:, 1]))
    epsg = utm_epsg_for(lon_c, lat_c)

    transformer = Transformer.from_crs("epsg:4326", epsg, always_xy=True)
    x, y = transformer.transform(latlon[:, 1], latlon[:, 0])
    return np.column_stack([x, y]), epsg


def cumulative_s(xy: np.ndarray):
    d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    return np.concatenate([[0.0], np.cumsum(d)])


def resample_polyline(xy: np.ndarray, step: float):
    s = cumulative_s(xy)
    L = float(s[-1])
    s_target = np.arange(0.0, L, step)
    x = np.interp(s_target, s, xy[:, 0])
    y = np.interp(s_target, s, xy[:, 1])
    return np.column_stack([x, y]), s_target, L


def tangent_heading(xy_res: np.ndarray):
    dx = np.gradient(xy_res[:, 0])
    dy = np.gradient(xy_res[:, 1])
    return np.arctan2(dy, dx)


def main():
    root = tk.Tk()
    root.withdraw()

    gpx_file = filedialog.askopenfilename(
        title="Choisir un fichier GPX",
        filetypes=[("GPX files", "*.gpx"), ("All files", "*.*")]
    )

    if not gpx_file:
        messagebox.showinfo("Annulé", "Aucun fichier sélectionné.")
        return

    gpx_path = Path(gpx_file)

    try:
        latlon = read_gpx_points(gpx_path)

        _, idx = np.unique(latlon, axis=0, return_index=True)
        latlon = latlon[np.sort(idx)]

        xy, epsg = project_to_meters(latlon)
        xy_res, s_res, L = resample_polyline(xy, step=5.0)
        theta = tangent_heading(xy_res)

        dx = xy_res[-1, 0] - xy_res[0, 0]
        dy = xy_res[-1, 1] - xy_res[0, 1]
        closure_err = np.hypot(dx, dy)

        if closure_err > 5.0:
            messagebox.showerror(
                "Erreur : tracé non fermé",
                f"Le tracé n'est pas fermé !\nErreur : {closure_err:.2f} m"
            )
            return

        df = pd.DataFrame({
            "s_m": s_res,
            "x_m": xy_res[:, 0],
            "y_m": xy_res[:, 1],
            "theta_rad": theta,
            "w_m": 12.0
        })

        out_path = gpx_path.with_name(gpx_path.stem + "_points_5m.csv")
        df.to_csv(out_path, index=False)

        messagebox.showinfo(
            "Succès",
            f"Conversion terminée !\n\nCSV créé :\n{out_path}"
        )

    except Exception as e:
        messagebox.showerror("Erreur", str(e))


if __name__ == "__main__":
    main()
