import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.cancha import dibujar_cancha, LARGO, ANCHO

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------- parámetros ----------------
CSV_ENTRADA = "data/trayectorias_equipos.csv"
SALIDA_PNG  = "outputs/mapa_calor_equipos2.png"
PARKED_RG   = 16.0         
NORMALIZE   = "per_team"    # "per_team" (relativo por equipo) | "shared" (absoluto comparable)
SIGMA       = 0.8           # suavizado gaussiano (en celdas); 0 para desactivar
BINS        = (18, 24)      # celdas ~10x10 cm sobre 182x243

NOMBRES = {0: "Equipo Verde", 1: "Equipo Oscuro"}
CMAP    = {0: "Greens", 1: "Oranges"}


def filtrar_parados(df, umbral_rg=PARKED_RG):
    """Quita robots casi estáticos (retirados/parados en la orilla)."""
    rg = df.groupby("tracker_id").agg(sx=("x_campo", "std"), sy=("y_campo", "std"))
    rg["rg"] = np.hypot(rg["sx"], rg["sy"])
    parados = rg.index[rg["rg"] < umbral_rg].tolist()
    if parados:
        detalle = [(int(t), round(float(rg.loc[t, "rg"]), 1)) for t in parados]
        print(f"Robots excluidos por estar parados (radio de giro < {umbral_rg} cm): {detalle}")
    else:
        print("No se excluyó ningún robot por estar parado.")
    return df[~df["tracker_id"].isin(parados)]


def histograma_equipo(sub):
    H, xe, ye = np.histogram2d(
        sub["x_campo"], sub["y_campo"], bins=BINS, range=[[0, ANCHO], [0, LARGO]]
    )
    if HAS_SCIPY and SIGMA:
        H = gaussian_filter(H, SIGMA)
    return H, xe, ye


def main():
    df = pd.read_csv(CSV_ENTRADA)
    df = filtrar_parados(df)

    hists = {eq: histograma_equipo(df[df["equipo"] == eq]) for eq in (0, 1)}
    vmax_shared = max(H.max() for H, _, _ in hists.values()) or 1.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    for ax, eq in zip(axes, [0, 1]):
        H, xe, ye = hists[eq]
        if NORMALIZE == "per_team":
            C = H / H.max() if H.max() > 0 else H
            vmin, vmax, etiqueta = 0, 1, "densidad relativa"
        else:  # "shared"
            C = H
            vmin, vmax, etiqueta = 0, vmax_shared, "frames (recuento)"

        malla = ax.pcolormesh(xe, ye, C.T, cmap=CMAP[eq], vmin=vmin, vmax=vmax,
                              zorder=0, shading="auto")
        dibujar_cancha(ax)  # líneas de la cancha encima del mapa
        fig.colorbar(malla, ax=ax, fraction=0.046, pad=0.04, label=etiqueta)
        ax.set_title(f"{NOMBRES[eq]} — mapa de calor", fontsize=12)

    plt.tight_layout()
    plt.savefig(SALIDA_PNG, dpi=130, bbox_inches="tight")
    print(f"Guardado: {SALIDA_PNG}  (modo={NORMALIZE}, scipy={HAS_SCIPY})")


if __name__ == "__main__":
    main()