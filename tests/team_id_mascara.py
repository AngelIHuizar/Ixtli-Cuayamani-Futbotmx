import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from src.segmentation import segmentar_robots

VERDE_BAJO = np.array([35, 40, 40])
VERDE_ALTO = np.array([90, 255, 255])


def _fraccion_verde_en_mascara(frame_bgr, mask):
    """Proporción de píxeles verdes DENTRO de la máscara (0 a 1)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, VERDE_BAJO, VERDE_ALTO) > 0
    pix_robot = mask.sum()
    if pix_robot == 0:
        return 0.0
    return float((verde & mask).sum() / pix_robot)


def _centroide(mask):
    ys, xs = np.where(mask)
    return (xs.mean(), ys.mean()) if len(xs) else (None, None)


def identificar_equipos_mascara(ruta_video, csv_robots="data/trayectorias_limpio.csv",
                                salida_csv="data/trayectorias_equipos.csv",
                                muestras_por_robot=20, max_dist=50):
    df = pd.read_csv(csv_robots)
    cap = cv2.VideoCapture(ruta_video)

    verde_por_robot = defaultdict(list)
    for tid in sorted(df["tracker_id"].unique()):
        filas = df[df["tracker_id"] == tid]
        muestras = filas.iloc[np.linspace(0, len(filas)-1,
                                          min(muestras_por_robot, len(filas)),
                                          dtype=int)]
        for _, r in muestras.iterrows():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
            ok, frame = cap.read()
            if not ok:
                continue

            robots = segmentar_robots(frame, filtrar_manos=False)
            if robots.mask is None or len(robots) == 0:
                continue
            mejor, mejor_d = None, 1e9
            for i in range(len(robots)):
                cx, cy = _centroide(robots.mask[i])
                if cx is None:
                    continue
                d = np.hypot(cx - r["x_px"], cy - r["y_px"])
                if d < mejor_d:
                    mejor_d, mejor = d, robots.mask[i]
            if mejor is not None and mejor_d <= max_dist:
                verde_por_robot[tid].append(_fraccion_verde_en_mascara(frame, mejor))
        nverde = np.mean(verde_por_robot[tid]) if verde_por_robot[tid] else 0
        print(f"  robot {tid}: verde_medio={nverde:.3f} ({len(verde_por_robot[tid])} muestras)")
    cap.release()

    ids = sorted(verde_por_robot.keys())
    valores = np.array([[np.mean(verde_por_robot[t])] for t in ids])

    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(valores)
    centro0 = valores[km.labels_ == 0].mean()
    centro1 = valores[km.labels_ == 1].mean()
    verde_es = 0 if centro0 > centro1 else 1
    equipo_de = {t: (0 if lbl == verde_es else 1) for t, lbl in zip(ids, km.labels_)}

    print("\nAsignación (0=verde, 1=oscuro):")
    for t in ids:
        print(f"  robot {t} -> equipo {equipo_de[t]} (verde={np.mean(verde_por_robot[t]):.3f})")

    df["equipo"] = df["tracker_id"].map(equipo_de)
    df.to_csv(salida_csv, index=False)
    print(f"\nGuardado: {salida_csv}")
    return df, equipo_de