# src/possession.py
"""
Posesión por equipo: en cada frame, el robot más cercano al balón (dentro de
un umbral) 'posee'. Se suma el tiempo por equipo.
Requiere el CSV con la columna 'equipo' ya llena, es decir, post-DINO y el CSV del balón.
"""
import numpy as np
import pandas as pd


def calcular_posesion(csv_equipos="data/trayectorias_equipos.csv",
                      csv_balon="data/balon_final.csv",
                      umbral_cm=30.0, fps=30):
    robots = pd.read_csv(csv_equipos)
    balon  = pd.read_csv(csv_balon)

    balon_por_frame = balon.set_index("frame")[["x_campo", "y_campo"]]

    posesion = []          
    for frame in balon_por_frame.index:
        bx, by = balon_por_frame.loc[frame, "x_campo"], balon_por_frame.loc[frame, "y_campo"]
        en_frame = robots[robots["frame"] == frame]
        if len(en_frame) == 0:
            continue

        d = np.hypot(en_frame["x_campo"] - bx, en_frame["y_campo"] - by)
        i_min = d.idxmin()
        if d.loc[i_min] <= umbral_cm:
            posesion.append(int(en_frame.loc[i_min, "equipo"]))

    posesion = pd.Series(posesion)
    total = len(posesion)
    if total == 0:
        print("No hubo frames con posesión clara.")
        return None

    print(f"Frames con posesión definida: {total}")
    print("\n--- POSESIÓN POR EQUIPO ---")
    resultado = {}
    for eq in sorted(posesion.unique()):
        n = (posesion == eq).sum()
        pct = 100 * n / total
        seg = n / fps
        nombre = "verde" if eq == 0 else "oscuro"
        resultado[eq] = pct
        print(f"  Equipo {eq} ({nombre}): {pct:.1f}%  ({seg:.1f} s, {n} frames)")
    return resultado