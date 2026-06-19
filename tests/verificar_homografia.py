# verificar_homografia.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.homography import cargar_H, proyectar, CAMPO_ANCHO, CAMPO_LARGO

RUTA_VIDEO = r"dataset/camara_superior/IMG_9933.mov"
FRAME = 0

H = cargar_H()

cap = cv2.VideoCapture(RUTA_VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
ok, frame = cap.read()
cap.release()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

REFERENCIAS = [
    ("Centro del circulo central",        (91.0, 121.5)),
    ("Poste IZQ de la porteria amarilla", (61.0,   0.0)),
    ("Poste IZQ de la porteria azul",     (61.0, 243.0)),
]

puntos_px = []
idx = [0]

fig, ax = plt.subplots(figsize=(7, 11))
ax.imshow(frame_rgb)
ax.set_title(f"Clic en: {REFERENCIAS[0][0]}")
ax.axis("off")

def on_click(event):
    if event.inaxes != ax or len(puntos_px) >= len(REFERENCIAS):
        return
    puntos_px.append((event.xdata, event.ydata))
    ax.scatter(event.xdata, event.ydata, c="yellow", s=90, zorder=5)
    fig.canvas.draw()
    idx[0] += 1
    if idx[0] < len(REFERENCIAS):
        ax.set_title(f"Clic en: {REFERENCIAS[idx[0]][0]}")
        fig.canvas.draw()

fig.canvas.mpl_connect("button_press_event", on_click)
plt.show()

print("\n--- Verificación de la homografía ---")
errores = []
for (nombre, esperado), px in zip(REFERENCIAS, puntos_px):
    proy = proyectar([px], H)[0]
    err = np.hypot(proy[0] - esperado[0], proy[1] - esperado[1])
    errores.append(err)
    print(f"{nombre}")
    print(f"   esperado: ({esperado[0]:.1f}, {esperado[1]:.1f}) cm")
    print(f"   proyectado: ({proy[0]:.1f}, {proy[1]:.1f}) cm")
    print(f"   error: {err:.1f} cm\n")

print(f"Error promedio: {np.mean(errores):.1f} cm")