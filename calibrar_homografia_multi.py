import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.homography import (calcular_homografia_multi, guardar_H, proyectar,
                            CAMPO_ANCHO, CAMPO_LARGO)

RUTA_VIDEO = r"dataset/camara_superior/IMG_9933.mov"
FRAME = 0

# Puntos de referencia (nombre, coordenada real en cm). Click EN ESTE ORDEN:
REFERENCIAS = [
    ("Esquina superior-izquierda",  (0.0,   0.0)),
    ("Poste IZQ porteria amarilla", (61.0,  0.0)),
    ("Poste DER porteria amarilla", (121.0, 0.0)),
    ("Esquina inferior-derecha",    (182.0, 243.0)),
    ("Poste DER porteria azul",     (121.0, 243.0)),
    ("Poste IZQ porteria azul",     (61.0,  243.0)),
    ("Esquina inferior-izquierda",  (0.0,   243.0)),
]

cap = cv2.VideoCapture(RUTA_VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
ok, frame = cap.read()
cap.release()
assert ok
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

puntos_px = []
idx = [0]
fig, ax = plt.subplots(figsize=(7, 11))
ax.imshow(frame_rgb)
ax.axis("off")
ax.set_title(f"Clic en: {REFERENCIAS[0][0]}")

def on_click(e):
    if e.inaxes != ax or len(puntos_px) >= len(REFERENCIAS):
        return
    puntos_px.append((e.xdata, e.ydata))
    ax.scatter(e.xdata, e.ydata, c="yellow", s=80, zorder=5)
    ax.text(e.xdata + 12, e.ydata, str(len(puntos_px)),
            color="yellow", fontsize=13, fontweight="bold")
    idx[0] += 1
    if idx[0] < len(REFERENCIAS):
        ax.set_title(f"Clic en: {REFERENCIAS[idx[0]][0]}")
    fig.canvas.draw()

fig.canvas.mpl_connect("button_press_event", on_click)
plt.show()

if len(puntos_px) != len(REFERENCIAS):
    raise SystemExit("Faltaron puntos. Vuelve a correrlo.")

puntos_cm = [c for _, c in REFERENCIAS]
H = calcular_homografia_multi(puntos_px, puntos_cm)
guardar_H(H)

# --- Error de reproyección sobre los puntos usados ---
print("\n--- Error de reproyeccion por punto ---")
errs = []
for (nombre, esperado), px in zip(REFERENCIAS, puntos_px):
    proy = proyectar([px], H)[0]
    err = np.hypot(proy[0] - esperado[0], proy[1] - esperado[1])
    errs.append(err)
    print(f"{nombre:30s} -> ({proy[0]:6.1f},{proy[1]:6.1f})  err {err:.1f} cm")
print(f"\nError de reproyeccion promedio: {np.mean(errs):.1f} cm")

# --- Verificación visual (vista cenital) ---
ESCALA = 3.0
S = np.array([[ESCALA, 0, 0], [0, ESCALA, 0], [0, 0, 1]])
H_vis = S @ H                                  # px -> cm -> px canónico
canvas = (int(CAMPO_ANCHO * ESCALA), int(CAMPO_LARGO * ESCALA))
warp = cv2.warpPerspective(frame, H_vis, canvas)
cv2.imwrite("outputs/verificacion_homografia2.jpg", warp)
print("Guardado: outputs/verificacion_homografia2.jpg")