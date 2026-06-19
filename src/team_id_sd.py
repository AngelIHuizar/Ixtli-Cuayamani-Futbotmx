import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from src.segmentation import segmentar_robots

VERDE_BAJO = np.array([35, 60, 50])
VERDE_ALTO = np.array([90, 255, 255])
AREA_MIN_ROBOT = 800    


def _verde_en_mascara(frame_bgr, mask):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, VERDE_BAJO, VERDE_ALTO) > 0
    n = mask.sum()
    return float((verde & mask).sum() / n) if n else 0.0


def clasificar(ruta_video, csv_robots="data/trayectorias_limpias.csv",
               salida_csv="data/trayectorias_equipos.csv",
               muestras=25, max_dist=45, min_tasa_robot=0.5):
    df = pd.read_csv(csv_robots)
    cap = cv2.VideoCapture(ruta_video)

    stats = {}   # tid -> (tasa_robot, verde_medio, n_muestras_validas)
    for tid in sorted(df["tracker_id"].unique()):
        filas = df[df["tracker_id"] == tid]
        muestra = filas.iloc[np.linspace(0, len(filas)-1,
                                         min(muestras, len(filas)), dtype=int)]
        verdes, con_robot, total = [], 0, 0
        for _, r in muestra.iterrows():
            total += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
            ok, frame = cap.read()
            if not ok:
                continue
            robots = segmentar_robots(frame, filtrar_manos=False)
            if robots.mask is None or len(robots) == 0:
                continue

            mejor, mejor_d = None, 1e9
            for i in range(len(robots)):
                ys, xs = np.where(robots.mask[i])
                if len(xs) == 0:
                    continue
                d = np.hypot(xs.mean()-r["x_px"], ys.mean()-r["y_px"])
                if d < mejor_d:
                    mejor_d, mejor = d, robots.mask[i]

            if mejor is not None and mejor_d <= max_dist and mejor.sum() >= AREA_MIN_ROBOT:
                con_robot += 1
                verdes.append(_verde_en_mascara(frame, mejor))
        tasa = con_robot / total if total else 0
        vmed = np.mean(verdes) if verdes else 0
        stats[tid] = (tasa, vmed, len(verdes))
        print(f"  robot {tid}: tasa_robot={tasa:.2f}, verde={vmed:.3f}, n={len(verdes)}")
    cap.release()

    reales = [t for t, (tasa, v, n) in stats.items()
              if tasa >= min_tasa_robot and n >= 3]
    print(f"\nTracks reales (con robot): {reales}")
    descartados = [t for t in stats if t not in reales]
    print(f"Descartados (basura/pasto): {descartados}")

    valores = np.array([[stats[t][1]] for t in reales])
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(valores)
    c0 = valores[km.labels_ == 0].mean()
    c1 = valores[km.labels_ == 1].mean()
    verde_es = 0 if c0 > c1 else 1
    equipo_de = {t: (0 if lbl == verde_es else 1) for t, lbl in zip(reales, km.labels_)}

    print("\nAsignación (0=verde, 1=oscuro):")
    for t in reales:
        print(f"  robot {t} -> equipo {equipo_de[t]} (verde={stats[t][1]:.3f})")

    df["equipo"] = df["tracker_id"].map(lambda t: equipo_de.get(t, -1))
    df = df[df["equipo"] != -1].reset_index(drop=True)   
    df.to_csv(salida_csv, index=False)
    print(f"\nGuardado: {salida_csv} ({df['tracker_id'].nunique()} robots reales)")
    return df, equipo_de