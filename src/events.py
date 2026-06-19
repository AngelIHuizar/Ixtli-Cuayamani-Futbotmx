# src/events.py
import numpy as np
import pandas as pd

# --- Geometría de la cancha (cm) ---
PORTERIA_X_MIN = 61.0
PORTERIA_X_MAX = 121.0
LINEA_AMARILLA = 0.0
LINEA_AZUL     = 243.0
AREA_PROF = 25.0 # Profundidad del área de penalti (cm), desde cada línea de gol

# --- Atribución de equipo por portería de destino ---
ANOTA_EN_AMARILLA = 0   # equipo B (verde)
ANOTA_EN_AZUL     = 1   # equipo A (oscuro)
NOMBRE = {0: "Verde", 1: "Oscuro"}
MIN_FRAMES_GOL = 30


def _en_ancho(x: float) -> bool:
    """¿La coordenada x cae dentro del ancho de la portería?"""
    return PORTERIA_X_MIN <= x <= PORTERIA_X_MAX


def _zona_gol(x: float, y: float):
    if not _en_ancho(x):
        return None
    if y <= LINEA_AMARILLA:
        return "amarilla"
    if y >= LINEA_AZUL:
        return "azul"
    return None


def detectar_llegadas_area(df_balon, fps=30, espera_seg=2.0):
    df = df_balon.sort_values("frame").reset_index(drop=True)
    llegadas, ultimo, dentro_de = [], -1e9, None

    for i in range(len(df)):
        x, y, frame = df.x_campo[i], df.y_campo[i], df.frame[i]

        zona = None
        if _en_ancho(x):
            if y <= LINEA_AMARILLA + AREA_PROF:      # área amarilla (0-25)
                zona = "amarilla"
            elif y >= LINEA_AZUL - AREA_PROF:        # área azul (218-243)
                zona = "azul"

        es_nueva = zona is not None and zona != dentro_de
        if es_nueva and (frame - ultimo) / fps >= espera_seg:
            equipo = ANOTA_EN_AMARILLA if zona == "amarilla" else ANOTA_EN_AZUL
            llegadas.append({"frame": int(frame), "porteria": zona,
                             "equipo": equipo, "equipo_nombre": NOMBRE[equipo],
                             "x": round(float(x), 1), "y": round(float(y), 1)})
            ultimo = frame
        dentro_de = zona

    return pd.DataFrame(llegadas)


detectar_tiros = detectar_llegadas_area


def detectar_goles(df_balon, fps=30, espera_seg=3.0,
                   min_frames_gol=MIN_FRAMES_GOL, salto_max_frames=15):

    df = df_balon.sort_values("frame").reset_index(drop=True)
    goles, ultimo_gol = [], -1e9
    i = 0
    n = len(df)

    while i < n:
        x, y, frame = df.x_campo[i], df.y_campo[i], df.frame[i]

        if (frame - ultimo_gol) / fps < espera_seg:
            i += 1
            continue

        zona = _zona_gol(x, y)
        if zona is None:
            i += 1
            continue

        primer_frame = frame
        ultimo_atras = frame
        f_prev = frame
        j = i + 1
        while j < n:
            fj = df.frame[j]
            if fj - f_prev > salto_max_frames:
                break  
            dentro = _zona_gol(df.x_campo[j], df.y_campo[j]) == zona
            if not dentro:
                break  
            ultimo_atras = fj
            f_prev = fj
            j += 1

        duracion = ultimo_atras - primer_frame

        if duracion >= min_frames_gol:
            equipo = ANOTA_EN_AMARILLA if zona == "amarilla" else ANOTA_EN_AZUL
            goles.append({"frame": int(primer_frame), "porteria": zona,
                          "equipo": equipo, "equipo_nombre": NOMBRE[equipo],
                          "x": round(float(x), 1), "y": round(float(y), 1),
                          "frames_sostenido": int(duracion)})
            ultimo_gol = ultimo_atras

        i = max(j, i + 1)

    return pd.DataFrame(goles)


def consolidar_goles(goles, fps=30, ventana_seg=10):
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