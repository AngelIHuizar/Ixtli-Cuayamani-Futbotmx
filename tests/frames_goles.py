import cv2
import pandas as pd
from src.events import detectar_goles

balon = pd.read_csv("data/balon_final.csv")
goles = detectar_goles(balon, modo="estricto")

cap = cv2.VideoCapture(r"dataset/camara_superior/recorte_2min.mov")  # tu ruta
for _, g in goles.iterrows():
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(g["frame"]))
    ok, frame = cap.read()
    if ok:
        cv2.imwrite(f"outputs/gol_f{int(g['frame'])}_{g['equipo_nombre']}.jpg", frame)
        print(f"Frame {g['frame']}: gol de {g['equipo_nombre']} en portería {g['porteria']}")
cap.release()