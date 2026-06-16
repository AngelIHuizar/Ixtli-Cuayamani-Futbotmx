# src/events.py
"""
Detección de goles desde la trayectoria del balón (cm), con atribución de equipo.
- Equipo verde (0): anota en la portería AMARILLA (y=0).
- Equipo oscuro (1): anota en la portería AZUL (y=243).
Criterio: el balón debe CRUZAR la línea y SOSTENER el cruce varios frames.
"""
import numpy as np
import pandas as pd

PORTERIA_X_MIN = 61.0
PORTERIA_X_MAX = 121.0
LINEA_AMARILLA = 0.0
LINEA_AZUL     = 243.0
# Profundidad del área de penalti (cm), desde la línea de gol
AREA_PROF = 25.0

ANOTA_EN_AMARILLA = 0   # equipo verde
ANOTA_EN_AZUL     = 1   # equipo oscuro
NOMBRE = {0: "Verde", 1: "Oscuro"}


def _en_ancho(x):
    return PORTERIA_X_MIN <= x <= PORTERIA_X_MAX

def detectar_tiros(df_balon, fps=30, espera_seg=2.0):
    """
    Un TIRO A GOL = el balón ENTRA al área de penalti (25 cm frente a la portería,
    dentro del ancho 61-121) acercándose a la línea de gol.
    Devuelve DataFrame: frame, porteria, equipo, equipo_nombre, x, y.
    """
    df = df_balon.sort_values("frame").reset_index(drop=True)
    tiros, ultimo, dentro_de = [], -1e9, None

    for i in range(len(df)):
        x, y, frame = df.x_campo[i], df.y_campo[i], df.frame[i]

        # ¿En el área de alguna portería?
        zona = None
        if _en_ancho(x):
            if y <= LINEA_AMARILLA + AREA_PROF:      # área amarilla (0–25)
                zona = "amarilla"
            elif y >= LINEA_AZUL - AREA_PROF:        # área azul (218–243)
                zona = "azul"

        # Contar un tiro al ENTRAR a la zona (no mientras siga dentro)
        es_nuevo = zona is not None and zona != dentro_de
        if es_nuevo and (frame - ultimo) / fps >= espera_seg:
            equipo = ANOTA_EN_AMARILLA if zona == "amarilla" else ANOTA_EN_AZUL
            tiros.append({"frame": int(frame), "porteria": zona,
                          "equipo": equipo, "equipo_nombre": NOMBRE[equipo],
                          "x": round(float(x), 1), "y": round(float(y), 1)})
            ultimo = frame
        dentro_de = zona

    return pd.DataFrame(tiros)

def detectar_goles(df_balon, fps=30, espera_seg=3.0,
                   puntos_confirmacion=2, salto_max_frames=15):
    """
    GOL si el balón cruza la línea de gol Y se mantiene del lado de adentro
    durante al menos 'puntos_confirmacion' detecciones seguidas (sin huecos
    mayores a 'salto_max_frames'). Esto evita los toques aislados (falsos cruces).
    Devuelve DataFrame con: frame, porteria, equipo, equipo_nombre, x, y.
    """
    df = df_balon.sort_values("frame").reset_index(drop=True)
    goles, ultimo_gol = [], -1e9

    for i in range(len(df)):
        x, y, frame = df.x_campo[i], df.y_campo[i], df.frame[i]
        if (frame - ultimo_gol) / fps < espera_seg:
            continue

        # ¿Está más allá de una línea de gol, dentro del ancho de portería?
        zona = None
        if _en_ancho(x):
            if y <= LINEA_AMARILLA:
                zona = "amarilla"
            elif y >= LINEA_AZUL:
                zona = "azul"
        if zona is None:
            continue

        # Confirmar: los siguientes puntos (consecutivos, sin huecos grandes)
        # también están del lado de adentro -> el balón ENTRÓ, no solo tocó.
        confirmados = 1
        f_prev = frame
        for j in range(i + 1, len(df)):
            if df.frame[j] - f_prev > salto_max_frames:
                break   # hueco grande: se perdió el balón, no podemos confirmar
            yj, xj = df.y_campo[j], df.x_campo[j]
            dentro = (zona == "amarilla" and yj <= LINEA_AMARILLA) or \
                     (zona == "azul" and yj >= LINEA_AZUL)
            if dentro:
                confirmados += 1
                f_prev = df.frame[j]
                if confirmados >= puntos_confirmacion:
                    break
            else:
                break   # salió: era un rebote/roce, no gol

        if confirmados >= puntos_confirmacion:
            equipo = ANOTA_EN_AMARILLA if zona == "amarilla" else ANOTA_EN_AZUL
            goles.append({"frame": int(frame), "porteria": zona,
                          "equipo": equipo, "equipo_nombre": NOMBRE[equipo],
                          "x": round(float(x), 1), "y": round(float(y), 1)})
            ultimo_gol = frame

    return pd.DataFrame(goles)

def consolidar_goles(goles, fps=30, ventana_seg=10):
    """Une detecciones cercanas en la misma portería como un solo gol."""
    if len(goles) == 0:
        return goles
    goles = goles.sort_values("frame").reset_index(drop=True)
    keep, ultimo_frame, ultima_porteria = [], -1e9, None
    for _, g in goles.iterrows():
        if not (g["porteria"] == ultima_porteria and
                (g["frame"] - ultimo_frame) / fps < ventana_seg):
            keep.append(g)
            ultima_porteria = g["porteria"]
        ultimo_frame = g["frame"]
    return pd.DataFrame(keep).reset_index(drop=True)