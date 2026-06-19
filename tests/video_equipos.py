import cv2
import pandas as pd
import numpy as np
from collections import defaultdict, deque

CSV_ROBOTS = "data/trayectorias_equipos.csv"
CSV_BALON  = "data/balon_final.csv"
VIDEO      = r"dataset/camara_superior/recorte_2min.mov"  
SALIDA     = "outputs/video_equipos.mp4"

COLORES = {0: (0, 200, 0),     # equipo verde
           1: (0, 140, 255)}   # equipo oscuro -> naranja
NOMBRES = {0: "Verde", 1: "Oscuro"}
COLOR_BALON = (0, 0, 255)      # rojo (BGR), para que contraste con el balón naranja

df      = pd.read_csv(CSV_ROBOTS)
balon   = pd.read_csv(CSV_BALON)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(SALIDA, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

robots_por_frame = {f: g for f, g in df.groupby("frame")}
balon_por_frame  = balon.set_index("frame")[["x_px", "y_px"]].to_dict("index")

estelas = defaultdict(lambda: deque(maxlen=25))
estela_balon = deque(maxlen=20)

idx = 0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while True:
    ok, frame = cap.read()
    if not ok:
        break

    # ---------- ROBOTS ----------
    if idx in robots_por_frame:
        for _, r in robots_por_frame[idx].iterrows():
            if pd.isna(r["equipo"]):
                continue
            eq = int(r["equipo"])
            color = COLORES.get(eq, (200, 200, 200))
            x, y = int(r["x_px"]), int(r["y_px"])
            tid = int(r["tracker_id"])

            estelas[tid].append((x, y))
            pts = list(estelas[tid])
            for k in range(1, len(pts)):
                cv2.line(frame, pts[k-1], pts[k], color, 2)

            cv2.circle(frame, (x, y), 26, color, 3)
            cv2.putText(frame, NOMBRES[eq], (x - 30, y - 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ---------- BALÓN ----------
    if idx in balon_por_frame:
        bx = int(balon_por_frame[idx]["x_px"])
        by = int(balon_por_frame[idx]["y_px"])
        estela_balon.append((bx, by))
        pts = list(estela_balon)
        for k in range(1, len(pts)):
            cv2.line(frame, pts[k-1], pts[k], COLOR_BALON, 2)
        cv2.circle(frame, (bx, by), 12, COLOR_BALON, 3)
        cv2.putText(frame, "balon", (bx + 14, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BALON, 2)

    writer.write(frame)
    idx += 1
    if idx % 200 == 0:
        print(f"  frame {idx}/{total}")

cap.release()
writer.release()
print(f"Guardado: {SALIDA}")