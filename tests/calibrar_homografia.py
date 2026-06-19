# calibrar_homografia como no no se ve la esquina superior derecha tomaré el de la porteria amarilla 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.homography import guardar_H, CAMPO_ANCHO, CAMPO_LARGO

RUTA_VIDEO = r"dataset/camara_superior/IMG_9933.mov"
FRAME = 0

cap = cv2.VideoCapture(RUTA_VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
ok, frame = cap.read()
cap.release()
assert ok
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Coordenadas REALES (cm) de los 4 puntos (en caso de que sean claramente vistos)
DESTINO_CM = np.float32([
    [0,           0],            # 1: esquina superior-izquierda (amarilla)
    [0,           CAMPO_LARGO],  # 2: esquina inferior-izquierda (azul)
    [CAMPO_ANCHO, CAMPO_LARGO],  # 3: esquina inferior-derecha (azul)
    [121,         0],            # 4: poste DERECHO de la portería amarilla
])

puntos = []
fig, ax = plt.subplots(figsize=(7, 11))
ax.imshow(frame_rgb)
ax.set_title("Clic EN ESTE ORDEN:\n"
             "1) sup-izq   2) inf-izq   3) inf-der\n"
             "4) poste DERECHO de la porteria amarilla (arriba)")
ax.axis("off")

def on_click(event):
    if event.inaxes != ax or len(puntos) >= 4:
        return
    puntos.append((event.xdata, event.ydata))
    ax.scatter(event.xdata, event.ydata, c="yellow", s=90, zorder=5)
    ax.text(event.xdata + 14, event.ydata, str(len(puntos)),
            color="yellow", fontsize=15, fontweight="bold")
    fig.canvas.draw()

fig.canvas.mpl_connect("button_press_event", on_click)
plt.show()

if len(puntos) != 4:
    raise SystemExit("Necesitas marcar 4 puntos.")

H = cv2.getPerspectiveTransform(np.float32(puntos), DESTINO_CM)
guardar_H(H)

# Verificación
ESCALA = 3.0
dst_vis = (DESTINO_CM * ESCALA).astype(np.float32)
H_vis = cv2.getPerspectiveTransform(np.float32(puntos), dst_vis)
canvas = (int(CAMPO_ANCHO*ESCALA), int(CAMPO_LARGO*ESCALA))
warp = cv2.warpPerspective(frame, H_vis, canvas)
cv2.imwrite("outputs/verificacion_homografia.jpg", warp)
print("Guardado: outputs/verificacion_homografia.jpg")