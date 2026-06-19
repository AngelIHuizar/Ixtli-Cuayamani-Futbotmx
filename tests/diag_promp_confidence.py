import cv2
import supervision as sv
from src.segmentation import segmentar_con_texto

cap = cv2.VideoCapture("dataset/camara_superior/IMG_9933.mov")
ok, frame = cap.read()
cap.release()
print("Frame leído:", ok, frame.shape if ok else "—")

for concepto in ["robot", "small robot", "orange ball"]:
    det = segmentar_con_texto(frame, concepto, umbral=0.25)
    anotado = sv.MaskAnnotator(opacity=0.6, color_lookup=sv.ColorLookup.INDEX).annotate(
        scene=frame.copy(), detections=det
    )
    nombre = concepto.replace(" ", "_")
    cv2.imwrite(f"outputs/diag_{nombre}.jpg", anotado)
    print(f"{concepto}: {len(det)} -> outputs/diag_{nombre}.jpg")