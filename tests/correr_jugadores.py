"""
correr_jugadores.py — Posesión y mapas de calor por jugador (R1/R2 de cada equipo).
Usa trayectorias_jugadores.csv (con columna 'jugador'). Imprime la posesión por
jugador y genera un mapa de calor por jugador (rejilla 2x2: Verde R1/R2, Oscuro R1/R2).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.ndimage import gaussian_filter

CSV_JUG   = "data/trayectorias_jugadores.csv"
CSV_BALON = "data/balon_final.csv"
PNG_MAPAS = "outputs/mapa_calor_jugadores.png"
LARGO, ANCHO = 243, 182
NOM = {0: "Verde", 1: "Oscuro"}
CMAP = {0: "Greens", 1: "Oranges"}
RG_MIN = 16          


def dibujar_cancha(ax, c="0.4"):
    ax.add_patch(Rectangle((0, 0), ANCHO, LARGO, fill=False, ec=c, lw=1.2, zorder=3))
    ax.plot([0, ANCHO], [121.5, 121.5], color=c, lw=1, zorder=3)
    ax.add_patch(Circle((91, 121.5), 30, fill=False, ec=c, lw=1, zorder=3))
    ax.add_patch(Rectangle((51, 0), 80, 25, fill=False, ec=c, lw=1, zorder=3))
    ax.add_patch(Rectangle((51, 218), 80, 25, fill=False, ec=c, lw=1, zorder=3))
    ax.plot([61, 121], [0, 0], color="#d4b106", lw=5, zorder=4)
    ax.plot([61, 121], [243, 243], color="#1f6fb0", lw=5, zorder=4)
    ax.set_xlim(-5, ANCHO + 5); ax.set_ylim(LARGO + 5, -5)
    ax.set_aspect("equal"); ax.axis("off")


def posesion_por_jugador(tj, b):
    by = b.set_index("frame")[["x_campo", "y_campo"]]
    cont, total = {}, 0
    for fr, g in tj.groupby("frame"):
        if fr not in by.index:
            continue
        d = np.hypot(g.x_campo - by.loc[fr, "x_campo"], g.y_campo - by.loc[fr, "y_campo"])
        i = d.idxmin()
        if d.loc[i] <= 30:
            k = (g.loc[i, "equipo"], g.loc[i, "jugador"]); cont[k] = cont.get(k, 0) + 1; total += 1
    print(f"POSESIÓN POR JUGADOR (sobre {total} frames con posesión):")
    for eq in (0, 1):
        for ju in (1, 2):
            c = cont.get((eq, ju), 0)
            print(f"  {NOM[eq]} R{ju}: {100*c/total:.0f}%")
    return cont, total


def mapas_por_jugador(tj):
    rg = tj.groupby("tracker_id").agg(sx=("x_campo", "std"), sy=("y_campo", "std"))
    rg["rg"] = np.hypot(rg.sx, rg.sy)
    tj = tj[~tj.tracker_id.isin(rg.index[rg.rg < RG_MIN])]

    fig, axes = plt.subplots(2, 2, figsize=(10, 13))
    for eq in (0, 1):
        for ju in (1, 2):
            ax = axes[ju - 1][eq]
            s = tj[(tj.equipo == eq) & (tj.jugador == ju)]
            if len(s):
                H, xe, ye = np.histogram2d(s.x_campo, s.y_campo, bins=(18, 24),
                                           range=[[0, ANCHO], [0, LARGO]])
                H = gaussian_filter(H, 0.8)
                C = H / H.max() if H.max() > 0 else H
                ax.pcolormesh(xe, ye, C.T, cmap=CMAP[eq], vmin=0, vmax=1, zorder=0, shading="auto")
            dibujar_cancha(ax)
            ax.set_title(f"{NOM[eq]} R{ju}", fontsize=12)
    plt.tight_layout()
    plt.savefig(PNG_MAPAS, dpi=130, bbox_inches="tight")
    print(f"Guardado: {PNG_MAPAS}")


def main():
    tj = pd.read_csv(CSV_JUG); b = pd.read_csv(CSV_BALON)
    posesion_por_jugador(tj, b)
    mapas_por_jugador(tj)


if __name__ == "__main__":
    main()