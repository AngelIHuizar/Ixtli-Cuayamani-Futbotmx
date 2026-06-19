import cv2, numpy as np
from src.segmentation import segmentar_con_texto
from src.homography import cargar_H, proyectar

H = cargar_H()
FRAMES = [167, 863, 2760, 5000, 15000] 

cap = cv2.VideoCapture(r"dataset/camara_superior/IMG_9933.mov")
for f in FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, frame = cap.read()
    if not ok:
        continue
    det = segmentar_con_texto(frame, "orange ball", umbral=0.2)

    vis = frame.copy()
    info = []
    for i in range(len(det)):
        x1, y1, x2, y2 = det.xyxy[i]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cm = proyectar([(cx, cy)], H)[0]
        conf = det.confidence[i] if det.confidence is not None else 0
        info.append(f"({cm[0]:.0f},{cm[1]:.0f}) conf={conf:.2f}")
        cv2.circle(vis, (int(cx), int(cy)), 16, (0, 0, 255), 3)
    cv2.putText(vis, f"f{f}: {len(det)} balones", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imwrite(f"outputs/sam_balon_{f}.jpg", vis)
    print(f"frame {f}: {len(det)} detecciones -> {info}")
cap.release()