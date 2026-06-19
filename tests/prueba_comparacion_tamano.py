import cv2, pandas as pd
from src.team_id import _recorte_marcador

df = pd.read_csv("data/trayectorias_limpias.csv")
cap = cv2.VideoCapture(r"dataset/camara_superior/recorte_2min.mov")

for tid in [9, 14, 0, 1]:       
    sub = df[df["tracker_id"] == tid]
    r = sub.iloc[len(sub)//2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
    ok, frame = cap.read()
    if not ok:
        continue
    for lado in [70, 110, 130, 160]:
        rec = _recorte_marcador(frame, r["x_px"], r["y_px"], lado)
        if rec.size > 0:
            cv2.imwrite(f"outputs/recorte_tid{tid}_lado{lado}.jpg", rec)
cap.release()
print("Listo. Compara los recortes de distintos tamaños.")