import cv2
import supervision as sv
from src.segmentation import segmentar_robots

RUTA = r"dataset/camara_superior/IMG_9933.mov"      
FRAME_OBJETIVO = 1939           

cap = cv2.VideoCapture(RUTA)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_OBJETIVO)
ok, frame = cap.read()
cap.release()
print("Frame leído:", ok)

sin_filtro = segmentar_robots(frame, filtrar_manos=False)
con_filtro = segmentar_robots(frame, filtrar_manos=True)
print(f"Sin filtro: {len(sin_filtro)} robots")
print(f"Con filtro: {len(con_filtro)} robots")

anotado = sv.MaskAnnotator(opacity=0.6, color_lookup=sv.ColorLookup.INDEX).annotate(
    scene=frame.copy(), detections=con_filtro
)
cv2.imwrite("outputs/prueba_filtro_1939.jpg", anotado)
print("Guardado: outputs/prueba_filtro_1939.jpg")