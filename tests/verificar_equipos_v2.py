# verificar_equipos_v2.py
import cv2
import pandas as pd
import numpy as np
from src.team_id_mascara import _fraccion_verde_en_mascara
from src.segmentation import segmentar_robots

df = pd.read_csv("data/trayectorias_equipos.csv")
cap = cv2.VideoCapture(r"dataset/camara_superior/recorte_2min.mov")

for tid in sorted(df["tracker_id"].unique()):
    sub = df[df["tracker_id"] == tid]
    eq = int(sub["equipo"].iloc[0])
    nombre_eq = "verde" if eq == 0 else "oscuro"
    # 3 muestras repartidas en la vida del track
    muestras = sub.iloc[np.linspace(0, len(sub)-1, 3, dtype=int)]
    for k, (_, r) in enumerate(muestras.iterrows()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
        ok, frame = cap.read()
        if not ok:
            continue
        x, y = int(r["x_px"]), int(r["y_px"])
        # recorte amplio para ver el cuerpo completo
        x1, y1 = max(0, x-90), max(0, y-90)
        x2, y2 = x+90, y+90
        rec = frame[y1:y2, x1:x2]
        if rec.size > 0:
            cv2.imwrite(f"outputs/check_eq{eq}_{nombre_eq}_tid{tid}_m{k}.jpg", rec)
    print(f"robot {tid} -> equipo {eq} ({nombre_eq})")
cap.release()
print("\nListo. Revisa outputs/check_eq*_*.jpg")