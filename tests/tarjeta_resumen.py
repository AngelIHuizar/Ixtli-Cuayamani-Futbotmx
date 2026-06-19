"""
correr_tarjeta_resumen.py — Tarjeta-resumen del partido (estilo deck Centro).

Lienzo de 100x100: cada elemento se coloca por coordenadas (x, y) de 0 a 100.
TODO lo editable está en el bloque DATOS y en COLORES. Para mover algo, cambia
sus coordenadas; para cambiar un número o texto, edítalo en DATOS.

"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ==================================================
SALIDA_PNG = "outputs/tarjeta_resumen.png"

# --- Colores ---
BG     = "#0d3b2e"   # fondo (verde deck)
VERDE  = "#2f9e5e"   # equipo Verde
OSCURO = "#d4691e"   # equipo Oscuro
TX     = "#f2f2ee"   # texto principal
SUB    = "#9fc3b0"   # texto secundario / etiquetas

# --- Datos del partido ---
TITULO    = "Resumen del partido"
SUBTITULO = "Recorte 2 min · cámara superior · FutBotMX"
GOL_VERDE, GOL_OSCURO = 0, 1
# barras de posesión: (etiqueta, % verde, % oscuro)
POSESIONES = [
    ("Posesión total (505 frames con balón)", 29, 71),
    ("Posesión cara a cara (ambos en cancha, 218 fr.)", 67, 33),
]
# stats sueltas: (etiqueta, valor, columna 'izq'/'der', color_valor)
STATS = [
    ("Llegadas al área",    "2",      "izq", TX),
    ("Disparos a portería", "1",      "izq", TX),
    ("Balón detectado",     "48%",    "der", TX),
    ("Más peligroso",       "Oscuro", "der", OSCURO),
]
NOTA = ("Nota: posesión sobre frames con balón visible. La acción se concentró en la mitad de la "
        "portería\nazul; Verde no llegó a generar tiro. Goles atribuidos por la dirección de ataque "
        "de cada equipo.")
# =============================================================


def barra_posesion(ax, y, label, v, o):
    ax.text(40, y + 5, label, color=SUB, fontsize=10, va="center")
    x0, w = 40, 52
    ax.add_patch(Rectangle((x0, y - 3), w * v / 100, 6, color=VERDE))
    ax.add_patch(Rectangle((x0 + w * v / 100, y - 3), w * o / 100, 6, color=OSCURO))
    ax.text(x0 + 1, y, f"{v}%", color="white", fontsize=9, va="center", fontweight="bold")
    ax.text(x0 + w - 1, y, f"{o}%", color="white", fontsize=9, va="center",
            ha="right", fontweight="bold")


def main():
    fig = plt.figure(figsize=(9, 5.2)); fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off"); ax.set_facecolor(BG)

    # Título
    ax.text(6, 92, TITULO, color=TX, fontsize=20, fontweight="bold", va="top")
    ax.text(6, 85.5, SUBTITULO, color=SUB, fontsize=10, va="top")

    # Marcador
    ax.text(6, 72, "GOL", color=SUB, fontsize=11, va="center")
    ax.text(6, 62, str(GOL_VERDE), color=VERDE, fontsize=44, fontweight="bold", va="center")
    ax.text(15, 62, "–", color=TX, fontsize=30, va="center")
    ax.text(22, 62, str(GOL_OSCURO), color=OSCURO, fontsize=44, fontweight="bold", va="center")
    ax.text(6, 50, "Verde", color=VERDE, fontsize=11, va="center", fontweight="bold")
    ax.text(22, 50, "Oscuro", color=OSCURO, fontsize=11, va="center", fontweight="bold")

    for i, (label, v, o) in enumerate(POSESIONES):
        barra_posesion(ax, 70 - i * 16, label, v, o)

    ax.add_patch(Rectangle((40, 32), 52, 0.3, color=SUB, alpha=0.4))
    fila = {"izq": 0, "der": 0}
    for label, val, col, c in STATS:
        y = 25 - fila[col] * 7
        xl, xv = (40, 64) if col == "izq" else (72, 94)
        ha = "center" if col == "izq" else "right"
        ax.text(xl, y, label, color=SUB, fontsize=10, va="center")
        ax.text(xv, y, val, color=c, fontsize=13, fontweight="bold", va="center", ha=ha)
        fila[col] += 1

    ax.text(6, 8, NOTA, color=SUB, fontsize=7.5, va="center", style="italic")
    plt.savefig(SALIDA_PNG, dpi=140, facecolor=BG, bbox_inches="tight")
    print(f"Guardado: {SALIDA_PNG}")


if __name__ == "__main__":
    main()