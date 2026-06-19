"""
Control de espacio (Voronoi) por equipo.

Divide el campo en celdas; cada celda pertenece al robot más cercano, y se pinta
del color de su equipo. La suma por equipo = % del campo controlado.

Produce dos cosas:
  1. Métrica AGREGADA: % de campo controlado por cada equipo, promediado SOLO sobre
     los frames con ambos equipos en cancha (si Verde no está, Oscuro controla el
     100% trivialmente, sin información). Se imprime y se grafica como barra.
  2. SNAPSHOT: el control de espacio en un frame concreto, sobre campo de césped.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import ListedColormap

# ==================================================
CSV_ROBOTS = "data/trayectorias_equipos.csv"
CSV_BALON  = "data/balon_final.csv"
SNAP_FRAME = 495                 # frame para el snapshot
SNAP_PNG   = "outputs/voronoi_snapshot.png"
BAR_PNG    = "outputs/control_espacio.png"
RES        = 2.0                 # cm por celda (menor = más fino y más lento)
VERDE, OSCURO = "#37c46a", "#ff7a1a"
GRASS, STRIPE = "#4a8f4e", "#458a49"
LARGO, ANCHO = 243, 182
# =============================================================

xs = np.arange(0, ANCHO, RES); ys = np.arange(0, LARGO, RES)
XX, YY = np.meshgrid(xs, ys)
GX, GY = XX.ravel(), YY.ravel()


def control_celdas(g):
    pts, eqs = g[["x_campo", "y_campo"]].values, g.equipo.values
    D = (GX[:, None] - pts[:, 0]) ** 2 + (GY[:, None] - pts[:, 1]) ** 2
    return eqs[D.argmin(axis=1)]


def agregado(t):
    pv, po, n = [], [], 0
    for _, g in t.groupby("frame"):
        if (g.equipo == 0).any() and (g.equipo == 1).any():
            near = control_celdas(g)
            pv.append(100 * (near == 0).mean()); po.append(100 * (near == 1).mean()); n += 1
    return np.mean(pv), np.mean(po), n


def dibujar_lineas(ax, c="white"):
    ax.add_patch(Rectangle((0, 0), ANCHO, LARGO, fill=False, ec=c, lw=2, zorder=3))
    ax.plot([0, ANCHO], [121.5, 121.5], color=c, lw=2, zorder=3)
    ax.add_patch(Circle((91, 121.5), 30, fill=False, ec=c, lw=2, zorder=3))
    ax.add_patch(Rectangle((51, 0), 80, 25, fill=False, ec=c, lw=2, zorder=3))
    ax.add_patch(Rectangle((51, 218), 80, 25, fill=False, ec=c, lw=2, zorder=3))
    ax.plot([61, 121], [0, 0], color="#f0cf2a", lw=7, zorder=4)
    ax.plot([61, 121], [243, 243], color="#2f8fd6", lw=7, zorder=4)
    ax.set_xlim(-6, ANCHO + 6); ax.set_ylim(LARGO + 6, -6)
    ax.set_aspect("equal"); ax.axis("off")


def snapshot(t, b, frame):
    g = t[t.frame == frame]; bb = b[b.frame == frame]
    near = control_celdas(g).reshape(XX.shape)
    pv = 100 * (near == 0).mean(); po = 100 * (near == 1).mean()
    fig, ax = plt.subplots(figsize=(5.8, 7.2))
    ax.imshow(near, extent=[0, ANCHO, LARGO, 0], origin="upper",
              cmap=ListedColormap([VERDE, OSCURO]), alpha=0.45, zorder=0, aspect="equal")
    dibujar_lineas(ax)
    for _, r in g.iterrows():
        ax.scatter([r.x_campo], [r.y_campo], s=240,
                   color=VERDE if r.equipo == 0 else OSCURO, edgecolor="white", lw=2.5, zorder=6)
    if len(bb):
        ax.scatter([bb.x_campo.iloc[0]], [bb.y_campo.iloc[0]], s=90,
                   color="white", edgecolor="black", lw=1.5, zorder=7)
    ax.set_title(f"Control de espacio — frame {frame}\nVerde {pv:.0f}%  ·  Oscuro {po:.0f}% del campo",
                 fontsize=11, pad=10, color="#222")
    plt.tight_layout(); plt.savefig(SNAP_PNG, dpi=140, bbox_inches="tight"); plt.close()


def barra(pv, po, n):
    fig, ax = plt.subplots(figsize=(8, 1.8))
    ax.barh([0], [pv], color=VERDE); ax.barh([0], [po], left=[pv], color=OSCURO)
    ax.text(pv / 2, 0, f"Verde\n{pv:.0f}%", ha="center", va="center", color="white", fontweight="bold")
    ax.text(pv + po / 2, 0, f"Oscuro\n{po:.0f}%", ha="center", va="center", color="white", fontweight="bold")
    ax.set_xlim(0, 100); ax.axis("off")
    ax.set_title(f"Control de espacio promedio  (sobre {n} frames con ambos equipos)", fontsize=12)
    plt.tight_layout(); plt.savefig(BAR_PNG, dpi=140, bbox_inches="tight"); plt.close()


def main():
    t = pd.read_csv(CSV_ROBOTS); b = pd.read_csv(CSV_BALON)
    pv, po, n = agregado(t)
    print(f"Control de espacio: Verde {pv:.0f}% / Oscuro {po:.0f}%  (n={n} frames)")
    barra(pv, po, n)
    snapshot(t, b, SNAP_FRAME)
    print(f"Guardado: {BAR_PNG} y {SNAP_PNG}")


if __name__ == "__main__":
    main()