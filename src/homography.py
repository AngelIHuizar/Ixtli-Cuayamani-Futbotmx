# src/homography.py
"""
Homografía: convierte coordenadas de imagen (píxeles) a coordenadas reales
de la cancha (cm). Basado en las dimensiones oficiales del Reglamento
Copa FutBotMX 2026 (sección 7): campo de 243 × 182 cm.
"""
import cv2
import numpy as np

# Dimensiones reales del campo (cm)
CAMPO_LARGO = 243.0   # eje portería-portería 
CAMPO_ANCHO = 182.0   # eje banda-banda 


def calcular_homografia(esquinas_px):
    """
    esquinas_px: las 4 esquinas del campo en la imagen, en este orden:
        1) superior-izquierda 
        2) superior-derecha
        3) inferior-derecha   
        4) inferior-izquierda
    Devuelve la matriz H que mapea píxeles -> cm de cancha.
    """
    src = np.float32(esquinas_px)
    dst = np.float32([
        [0,           0],            # 1 -> (0, 0)
        [CAMPO_ANCHO, 0],            # 2 -> (182, 0)
        [CAMPO_ANCHO, CAMPO_LARGO],  # 3 -> (182, 243)
        [0,           CAMPO_LARGO],  # 4 -> (0, 243)
    ])
    return cv2.getPerspectiveTransform(src, dst)

def calcular_homografia_multi(puntos_px, puntos_cm):
    """Homografía robusta con N>=4 correspondencias (mínimos cuadrados)."""
    src = np.float32(puntos_px)
    dst = np.float32(puntos_cm)
    H, _ = cv2.findHomography(src, dst, method=0)   # method=0: usa TODOS los puntos
    return H

def proyectar(puntos_px, H):
    """Proyecta puntos (N,2) de píxeles de imagen a cm de cancha."""
    pts = np.asarray(puntos_px, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def guardar_H(H, ruta="data/homografia.npy"):
    np.save(ruta, H)
    print(f"Homografía guardada en {ruta}")


def cargar_H(ruta="data/homografia.npy"):
    return np.load(ruta)