# src/cancha.py
"""Dibuja la cancha de FutBotMX a escala real (cm) sobre un eje de matplotlib."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc

LARGO, ANCHO = 243.0, 182.0   # y, x en cm


def dibujar_cancha(ax=None, color_linea="white", color_campo="#1b6b3a"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 7))
    ax.set_facecolor(color_campo)
    lw = 2
    # Borde
    ax.add_patch(Rectangle((0, 0), ANCHO, LARGO, fill=False, ec=color_linea, lw=lw))
    # Línea central
    ax.plot([0, ANCHO], [LARGO/2, LARGO/2], color=color_linea, lw=lw)
    # Círculo central (60 cm diámetro = 30 radio)
    ax.add_patch(Circle((ANCHO/2, LARGO/2), 30, fill=False, ec=color_linea, lw=lw))
    # Áreas de penalti (80 ancho x 25 prof), centradas en x
    ax.add_patch(Rectangle((ANCHO/2 - 40, 0), 80, 25, fill=False, ec=color_linea, lw=lw))
    ax.add_patch(Rectangle((ANCHO/2 - 40, LARGO-25), 80, 25, fill=False, ec=color_linea, lw=lw))
    # Porterías (60 ancho), como líneas gruesas de color
    ax.plot([ANCHO/2-30, ANCHO/2+30], [0, 0], color="#e6c200", lw=5)        # amarilla
    ax.plot([ANCHO/2-30, ANCHO/2+30], [LARGO, LARGO], color="#2a6fb0", lw=5) # azul

    ax.set_xlim(-12, ANCHO+12)
    ax.set_ylim(LARGO+12, -12)     
    ax.set_aspect("equal")
    ax.axis("off")
    return ax