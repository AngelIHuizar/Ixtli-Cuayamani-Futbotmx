import cv2
from src.ball import detectar_balon, mascara_balon
from src.homography import cargar_H

H = cargar_H()
RUTA = r"dataset/camara_superior/IMG_9933.mov"
FRAMES = [0, 5000, 10000, 15000, 20000]

cap = cv2.VideoCapture(RUTA)
for f in FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, frame = cap.read()
    if not ok:
        continue
    centro = detectar_balon(frame, H=H)        # <-- ahora con homografía
    mask   = mascara_balon(frame)
    vis = frame.copy()
    if centro is not None:
        cv2.circle(vis, (int(centro[0]), int(centro[1])), 18, (0, 0, 255), 3)
        estado = f"balon en ({centro[0]:.0f}, {centro[1]:.0f})"
    else:
        estado = "NO detectado"
    cv2.putText(vis, f"frame {f}: {estado}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imwrite(f"outputs/balon_{f:05d}.jpg", vis)
    cv2.imwrite(f"outputs/balon_mask_{f:05d}.jpg", mask)
    print(f"frame {f}: {estado}")
cap.release()