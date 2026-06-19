# verificar_equipos.py
import cv2
import pandas as pd
from src.team_id import _recorte_marcador

df = pd.read_csv("data/trayectorias_equipos.csv")
cap = cv2.VideoCapture(r"dataset/camara_superior/recorte_2min.mov")  

for tid in sorted(df["tracker_id"].unique()):
    sub = df[df["tracker_id"] == tid]
    r = sub.iloc[len(sub) // 2]           
    equipo = int(r["equipo"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
    ok, frame = cap.read()
    if ok:
        rec = _recorte_marcador(frame, r["x_px"], r["y_px"], lado=110)
        if rec.size > 0:
            cv2.imwrite(f"outputs/eq{equipo}_robot{tid}.jpg", rec)
            print(f"robot {tid} -> equipo {equipo}")
cap.release()
print("Listo. Revisa outputs/eq0_*.jpg y outputs/eq1_*.jpg")