"""
Mapa de disparos y goles por equipo, sobre campo de fútbol.

Define un DISPARO (tiro a gol) como un evento en que el balón acelera por encima de un umbral
dirigido hacia una portería (un golpe/remate), y marca el punto desde donde sale.
Esto es lo que produce un verdadero shot map (puntos repartidos por posición),
a diferencia de "el balón entró al área", que solo da 1-2 puntos.

Filtra los rebotes post-gol (el balón saliendo de la red) excluyendo disparos que
caen dentro de un episodio de gol ya detectado.

Dirección de ataque (validada con los datos): Verde -> amarilla, Oscuro -> azul.
El campo se dibuja aquí mismo (no usa src/cancha.py) para tener el fondo de césped.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from src.events import detectar_goles, consolidar_goles

CSV_BALON  = "data/balon_final.csv"
SALIDA_PNG = "outputs/shot_map2.png"
UMBRAL_CMF = 4.0     
SALTO_MAX  = 3       

# Colores de equipo (versiones vivas para resaltar sobre césped)
VERDE, OSCURO = "#37c46a", "#ff7a1a"
# Campo
GRASS, STRIPE, LINE = "#4a8f4e", "#458a49", "white"
AMARILLA_C, AZUL_C = "#f0cf2a", "#2f8fd6"

# Quién ataca cada portería (validado por la dirección media del balón en posesión)
ATACA  = {"amarilla": 0, "azul": 1}      # 0=Verde, 1=Oscuro
NOMBRE = {0: "Verde", 1: "Oscuro"}
COLOR  = {0: VERDE, 1: OSCURO}
LARGO, ANCHO = 243, 182


def dibujar_campo(ax):
    ax.add_patch(Rectangle((0, 0), ANCHO, LARGO, color=GRASS, zorder=0))
    for k in range(0, LARGO, 27):
        if (k // 27) % 2 == 0:
            ax.add_patch(Rectangle((0, k), ANCHO, 27, color=STRIPE, zorder=0))
    ax.add_patch(Rectangle((0, 0), ANCHO, LARGO, fill=False, ec=LINE, lw=2, zorder=3))
    ax.plot([0, ANCHO], [121.5, 121.5], color=LINE, lw=2, zorder=3)
    ax.add_patch(Circle((91, 121.5), 30, fill=False, ec=LINE, lw=2, zorder=3))
    ax.add_patch(Circle((91, 121.5), 1.5, color=LINE, zorder=3))
    ax.add_patch(Rectangle((51, 0), 80, 25, fill=False, ec=LINE, lw=2, zorder=3))
    ax.add_patch(Rectangle((51, 218), 80, 25, fill=False, ec=LINE, lw=2, zorder=3))
    ax.plot([61, 121], [0, 0], color=AMARILLA_C, lw=7, zorder=4, solid_capstyle="butt")
    ax.plot([61, 121], [243, 243], color=AZUL_C, lw=7, zorder=4, solid_capstyle="butt")
    ax.set_xlim(-6, ANCHO + 6); ax.set_ylim(LARGO + 6, -6)
    ax.set_aspect("equal"); ax.axis("off")


def detectar_disparos(df_balon, umbral=UMBRAL_CMF, salto_max=SALTO_MAX):
    """Disparo = transición de balón lento a rápido hacia una portería."""
    df = df_balon.sort_values("frame").reset_index(drop=True)
    f, x, y = df.frame.values, df.x_campo.values, df.y_campo.values

    def vel(i):
        if i < 0 or i >= len(df) - 1 or f[i + 1] - f[i] > salto_max:
            return None
        return np.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) / (f[i + 1] - f[i])

    disparos, i = [], 1
    while i < len(df) - 1:
        s, sp = vel(i), vel(i - 1)
        if s is not None and sp is not None and sp < umbral <= s:
            porteria = "azul" if (y[i + 1] - y[i]) > 0 else "amarilla"
            pico, j = s, i
            while j < len(df) - 1 and (v := vel(j)) is not None and v >= umbral:
                pico = max(pico, v); j += 1
            equipo = ATACA[porteria]
            disparos.append({"frame": int(f[i]), "x": round(float(x[i]), 1),
                             "y": round(float(y[i]), 1), "porteria": porteria,
                             "equipo": equipo, "equipo_nombre": NOMBRE[equipo],
                             "vel_ms": round(pico * 30 / 100, 1)})
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(disparos)


def quitar_rebotes_post_gol(disparos, goles, margen=30):
    """Excluye 'disparos' que caen dentro/justo después de un episodio de gol."""
    if len(disparos) == 0 or len(goles) == 0:
        return disparos
    malos = [(g["frame"] - 5, g["frame"] + 260 + margen) for _, g in goles.iterrows()]
    keep = [not any(a <= r.frame <= b for a, b in malos) for _, r in disparos.iterrows()]
    return disparos[keep].reset_index(drop=True)


def main():
    balon = pd.read_csv(CSV_BALON)
    goles = consolidar_goles(detectar_goles(balon))
    disparos = quitar_rebotes_post_gol(detectar_disparos(balon), goles)

    fig, ax = plt.subplots(figsize=(5.8, 7.2))
    dibujar_campo(ax)
    blanco = [pe.withStroke(linewidth=3, foreground="white")]

    for _, s in disparos.iterrows():    
        ax.scatter([s.x], [s.y], s=200, facecolor="white",
                   edgecolor=COLOR[s.equipo], lw=3, zorder=6)
    for _, g in goles.iterrows():        
        eq = ATACA[g.porteria]
        gy = 238 if g.porteria == "azul" else 5
        ax.scatter([g.x], [gy], s=420, marker="*", color=COLOR[eq],
                   edgecolor="white", lw=2, zorder=7)

    g_o = (goles.porteria == "azul").sum() if len(goles) else 0
    g_v = (goles.porteria == "amarilla").sum() if len(goles) else 0
    n_v = (disparos.equipo == 0).sum() if len(disparos) else 0
    n_o = (disparos.equipo == 1).sum() if len(disparos) else 0
    ax.set_title(f"Mapa de disparos y goles\nVerde {g_v} gol(es), {n_v} disparo(s)  ·  "
                 f"Oscuro {g_o} gol(es), {n_o} disparo(s)", fontsize=11, pad=10, color="#222")
    leg = [Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                  markeredgecolor="#888", markeredgewidth=2, markersize=11, label="Disparo"),
           Line2D([0], [0], marker="*", color="none", markerfacecolor="#888",
                  markersize=15, label="Gol")]
    ax.legend(handles=leg, loc="upper left", fontsize=8, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(SALIDA_PNG, dpi=140, bbox_inches="tight")
    print(f"Disparos: {len(disparos)} | Goles: {len(goles)}")
    print(f"Guardado: {SALIDA_PNG}")


if __name__ == "__main__":
    main()