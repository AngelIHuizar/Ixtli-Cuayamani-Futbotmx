# src/team_id.py
"""
Identificación de equipos por el marcador superior de cada robot, usando
embeddings de DINOv3 + clustering K-Means en 2 grupos.
Asigna cada tracker_id a un equipo (0 o 1) de forma estable.
"""
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from transformers import AutoImageProcessor, AutoModel
from sklearn.cluster import KMeans

device = "cuda" if torch.cuda.is_available() else "cpu"
MODELO_ID = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
procesador = AutoImageProcessor.from_pretrained(MODELO_ID)
modelo = AutoModel.from_pretrained(MODELO_ID).to(device)
modelo.train(False)


def _embedding(recorte_bgr) -> np.ndarray:
    """Embedding normalizado de un recorte BGR."""
    img = Image.fromarray(cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB))
    inps = procesador(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = modelo(**inps).pooler_output
    emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy()[0]


def _recorte_marcador(frame, x_px, y_px, lado=70):
    """Recorta un cuadro alrededor del centro del robot."""
    h, w = frame.shape[:2]
    x, y = int(x_px), int(y_px)
    x1, y1 = max(0, x - lado // 2), max(0, y - lado // 2)
    x2, y2 = min(w, x + lado // 2), min(h, y + lado // 2)
    return frame[y1:y2, x1:x2]


def identificar_equipos(ruta_video, csv_robots="data/trayectorias_limpias.csv",
                        salida_csv="data/trayectorias_equipos.csv",
                        muestras_por_robot=15, lado=70):
    """
    Para cada tracker_id: recolecta varios recortes, promedia sus embeddings,
    y clasifica el robot completo en equipo 0 o 1 (estable, sin parpadeos).
    """
    df = pd.read_csv(csv_robots)
    cap = cv2.VideoCapture(ruta_video)

    embs_por_robot = defaultdict(list)
    for tid in sorted(df["tracker_id"].unique()):
        filas = df[df["tracker_id"] == tid]
        muestras = filas.iloc[np.linspace(0, len(filas) - 1,
                                          min(muestras_por_robot, len(filas)),
                                          dtype=int)]
        for _, r in muestras.iterrows():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
            ok, frame = cap.read()
            if not ok:
                continue
            recorte = _recorte_marcador(frame, r["x_px"], r["y_px"], lado)
            if recorte.size > 0:
                embs_por_robot[tid].append(_embedding(recorte))
        print(f"  robot {tid}: {len(embs_por_robot[tid])} muestras")
    cap.release()

    ids = sorted(embs_por_robot.keys())
    matriz = np.array([np.mean(embs_por_robot[t], axis=0) for t in ids])

    # ----- agrupar en 2 equipos -----
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(matriz)
    equipo_de = {tid: int(lbl) for tid, lbl in zip(ids, km.labels_)}
    print("\nAsignación de equipos:")
    for tid in ids:
        print(f"  robot {tid} -> equipo {equipo_de[tid]}")

    # Escribir la columna 'equipo' en el CSV
    df["equipo"] = df["tracker_id"].map(equipo_de)
    df.to_csv(salida_csv, index=False)
    print(f"\nGuardado: {salida_csv}")
    return df, equipo_de