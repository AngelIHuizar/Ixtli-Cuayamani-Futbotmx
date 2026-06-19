import cv2
import supervision as sv
from src.segmentation import segmentar_con_texto

RUTA = "dataset/camara_superior/IMG_9933.mov"          
PROMPT = "small robot"
UMBRAL = 0.25
N_MUESTRAS = 12                       # cuántos frames revisar a lo largo del video

cap = cv2.VideoCapture(RUTA)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total} frames, {fps:.1f} fps, ~{total/fps:.0f} s")

indices = [int(total * i / N_MUESTRAS) for i in range(N_MUESTRAS)]

conteos = []
for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)   
    ok, frame = cap.read()
    if not ok:
        print(f"Frame {idx}: no se pudo leer")
        continue

    det = segmentar_con_texto(frame, PROMPT, umbral=UMBRAL)
    conteos.append((idx, len(det)))

    anotado = sv.MaskAnnotator(opacity=0.6, color_lookup=sv.ColorLookup.INDEX).annotate(
        scene=frame.copy(), detections=det
    )
    cv2.putText(anotado, f"frame {idx}: {len(det)} robots", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.imwrite(f"outputs/val_{idx:05d}.jpg", anotado)

cap.release()

print("\nResumen de detecciones por frame:")
for idx, n in conteos:
    marca = "  <-- revisar" if n != 4 else ""
    print(f"  frame {idx:6d}: {n} robots{marca}")