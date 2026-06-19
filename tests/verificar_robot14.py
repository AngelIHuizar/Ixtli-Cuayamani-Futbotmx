# verificar_robot14.py
import cv2
import pandas as pd
import numpy as np

TID = 14
df = pd.read_csv("data/trayectorias_equipos.csv")
sub = df[df["tracker_id"] == TID].reset_index(drop=True)
print(f"Robot {TID}: {len(sub)} frames, equipo actual = {int(sub['equipo'].iloc[0])}")

cap = cv2.VideoCapture(r"dataset/camara_superior/recorte_2min.mov")
# 6 muestras repartidas en toda la vida del track
muestras = sub.iloc[np.linspace(0, len(sub)-1, 6, dtype=int)]
for k, (_, r) in enumerate(muestras.iterrows()):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
    ok, frame = cap.read()
    if not ok:
        continue
    x, y = int(r["x_px"]), int(r["y_px"])
    x1, y1 = max(0, x-100), max(0, y-100)
    rec = frame[y1:y+100, x1:x+100]
    if rec.size > 0:
        cv2.imwrite(f"outputs/robot14_muestra{k}_f{int(r['frame'])}.jpg", rec)
        print(f"  muestra {k}: frame {int(r['frame'])}")
cap.release()
print("Listo. Revisa outputs/robot14_muestra*.jpg")