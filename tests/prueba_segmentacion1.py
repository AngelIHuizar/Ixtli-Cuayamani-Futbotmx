import cv2
from src.segmentation import segmentar_con_texto

cap = cv2.VideoCapture("dataset/camara_superior/recorte_2min.mov")
ok, frame = cap.read()
cap.release()

det_robots = segmentar_con_texto(frame, "robot")
det_balon  = segmentar_con_texto(frame, "mini orange ball")
print(f"Robots: {len(det_robots)} | Balón: {len(det_balon)}")