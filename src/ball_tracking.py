import cv2
import numpy as np
import pandas as pd
from src.ball import candidatos_balon
from src.homography import cargar_H, proyectar

def limpiar_trayectoria(df, max_vel_cm_s=400, fps=30):
    if len(df) < 3:
        return df
    df = df.sort_values("frame").reset_index(drop=True)
    keep = [True] * len(df)
    for i in range(1, len(df)):
        dt = (df.frame[i] - df.frame[i-1]) / fps
        if dt == 0:
            continue
        d = np.hypot(df.x_campo[i] - df.x_campo[i-1],
                     df.y_campo[i] - df.y_campo[i-1])
        if d / dt > max_vel_cm_s:      
            keep[i] = False
    return df[keep].reset_index(drop=True)

def rastrear_balon(ruta_video, salida_csv="data/balon.csv",
                   frame_inicio=0, frame_fin=None,
                   max_salto_px=130, max_perdidos=45):
    H = cargar_H()
    cap = cv2.VideoCapture(ruta_video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_fin is None:
        frame_fin = total
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)

    filas, ult, perdidos, idx = [], None, 0, frame_inicio
    while idx < frame_fin:
        ok, frame = cap.read()
        if not ok:
            break
        cands = candidatos_balon(frame, H=H)
        elegido = None
        if cands:
            if ult is None:
                elegido = max(cands, key=lambda c: c[3])[:2]
            else:
                d, c = min(((np.hypot(c[0]-ult[0], c[1]-ult[1]), c) for c in cands),
                           key=lambda z: z[0])
                if d <= max_salto_px:
                    elegido = c[:2]

        if elegido is not None:
            ult, perdidos = elegido, 0
            filas.append({"frame": idx, "x_px": elegido[0], "y_px": elegido[1]})
        else:
            perdidos += 1
            if perdidos > max_perdidos:   
                ult = None
        idx += 1
        if idx % 2000 == 0:
            print(f"  frame {idx}/{frame_fin} | balon detectado en {len(filas)} frames")
    cap.release()

    df = pd.DataFrame(filas)
    if len(df):
        cm = proyectar(df[["x_px", "y_px"]].values, H)
        df["x_campo"], df["y_campo"] = cm[:, 0], cm[:, 1]

    df = limpiar_trayectoria(df)

    df.to_csv(salida_csv, index=False)
    print(f"Guardado: {salida_csv} ({len(df)} frames con balon)")
    return df