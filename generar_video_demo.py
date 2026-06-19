"""
Dos modos (variable MODO):
  - "sam": vuelve a correr SAM 3 frame a frame y dibuja la MÁSCARA real de cada
           robot y del balón.
  - "csv": rápido, sin máscaras (círculos + estelas desde el CSV). Para previsualizar.

Indicadores visuales:
  - Máscara semitransparente con el color del equipo (solo modo "sam").
  - Contorno de la máscara + caja del detector.
  - Etiqueta "A·R2  ID15  .91"  =  Equipo A, Robot 2, tracker_id 15, confianza.
  - Estela de movimiento por robot y del balón.
  - HUD con leyenda (SAM / ByteTrack / DINOv3+KMeans), clave de color y contador.

Equipos:  Equipo A = Oscuro,  Equipo B = Verde.
Colores consistentes con las figuras:
  Verde  #37c46a   Oscuro #ff7a1a   (convertidos a BGR para OpenCV).
"""
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, deque

# ==================================================
MODO        = "sam"
CSV_ROBOTS  = "data/trayectorias_jugadores.csv"
CSV_BALON   = "data/balon_final.csv"
VIDEO       = r"dataset/camara_superior/recorte_2min.mov"
SALIDA      = "outputs/video_demo.mp4"   

FRAME_FIN     = None          
PASO          = 1
UMBRAL_ROBOT  = 0.25
UMBRAL_BALON  = 0.35
FILTRAR_MANOS = False
MATCH_PX      = 45
# =============================================================

def _hex_a_bgr(h):
    h = h.lstrip("#"); r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b, g, r)
COLOR_BGR   = {0: _hex_a_bgr("#37c46a"), 1: _hex_a_bgr("#ff7a1a")}   # 0=Verde, 1=Oscuro
COLOR_BALON = (0, 0, 255)
COLOR_GRIS  = (170, 170, 170)
LETRA       = {0: "B", 1: "A"}      # Verde=B, Oscuro=A
NOMBRE      = {0: "Verde", 1: "Oscuro"}

df    = pd.read_csv(CSV_ROBOTS)
balon = pd.read_csv(CSV_BALON)

equipo_de  = df.dropna(subset=["equipo"]).groupby("tracker_id")["equipo"].first().astype(int).to_dict()
jugador_de = df.dropna(subset=["jugador"]).groupby("tracker_id")["jugador"].first().astype(int).to_dict()

robots_por_frame = {f: g[["x_px","y_px","tracker_id","confianza"]].values
                    for f, g in df.groupby("frame")}
balon_por_frame  = balon.set_index("frame")[["x_px","y_px"]].to_dict("index")


def etiqueta(tid, conf=None):
    eq = equipo_de.get(tid)
    if eq is None:
        return f"ID{tid}", COLOR_GRIS
    jug = jugador_de.get(tid, "?")
    txt = f"{LETRA[eq]}-R{jug}  ID{tid}"
    if conf is not None and not np.isnan(conf):
        txt += f"  .{int(conf*100):02d}"
    return txt, COLOR_BGR[eq]


def emparejar(cx, cy, idx):
    arr = robots_por_frame.get(idx)
    if arr is None or len(arr) == 0:
        return None, None
    d = np.hypot(arr[:,0]-cx, arr[:,1]-cy)
    j = int(np.argmin(d))
    if d[j] > MATCH_PX:
        return None, None
    return int(arr[j,2]), (float(arr[j,3]) if not np.isnan(arr[j,3]) else None)


def dibujar_mascara(frame, mask, color, alpha=0.45):
    overlay = frame.copy()
    overlay[mask] = color
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, cnts, -1, color, 2)


#def poner_etiqueta(frame, x, y, texto, color):
   # (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    #cv2.rectangle(frame, (x, y-th-8), (x+tw+8, y), color, -1)
   # cv2.putText(frame, texto, (x+4, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)

def poner_etiqueta(frame, x, y, texto, color):
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x, y-th-8), (x+tw+8, y), (30, 30, 30), -1)
    cv2.putText(frame, texto, (x+4, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                color, 2, cv2.LINE_AA)


def dibujar_hud(frame, idx, total, modo):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0,0), (w,34), (0,0,0), -1)
    txt = ("SAM 3: segmentacion  |  ByteTrack: tracking  |  DINOv3+KMeans: equipos"
           if modo=="sam" else "Tracking (CSV)")
    cv2.putText(frame, txt, (10,23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    x0 = w - 230
    for eq, dx in ((1,0), (0,110)):
        cv2.rectangle(frame, (x0+dx,8), (x0+dx+18,26), COLOR_BGR[eq], -1)
        cv2.putText(frame, f"Eq.{LETRA[eq]} {NOMBRE[eq]}", (x0+dx+22,23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"frame {idx}/{total}", (10,h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)


if MODO == "sam":
    from src.segmentation import segmentar_robots, segmentar_con_texto


def detectar_balon_det(frame, idx):
    det = segmentar_con_texto(frame, "mini orange ball", umbral=UMBRAL_BALON)
    if len(det) == 0:
        return None, None
    cents = [((det.xyxy[i][0]+det.xyxy[i][2])/2, (det.xyxy[i][1]+det.xyxy[i][3])/2)
             for i in range(len(det))]
    ref = balon_por_frame.get(idx)
    if ref is not None:
        d = [np.hypot(cx-ref["x_px"], cy-ref["y_px"]) for cx,cy in cents]
        i = int(np.argmin(d))
    else:
        i = int(np.argmax(det.confidence)) if det.confidence is not None else 0
    m = det.mask[i] if det.mask is not None else None
    return m, cents[i]


def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir el video: {VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(SALIDA, cv2.VideoWriter_fourcc(*"mp4v"), fps/PASO, (w,h))

    estelas = defaultdict(lambda: deque(maxlen=25))
    estela_balon = deque(maxlen=20)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if FRAME_FIN is not None and idx >= FRAME_FIN:
            break
        if idx % PASO != 0:
            idx += 1
            continue

        if MODO == "sam":
            dets = segmentar_robots(frame, umbral=UMBRAL_ROBOT, filtrar_manos=FILTRAR_MANOS)
            for i in range(len(dets)):
                mask = dets.mask[i]
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    continue
                cx, cy = float(xs.mean()), float(ys.mean())
                tid, conf = emparejar(cx, cy, idx)
                if tid is not None and equipo_de.get(tid) is not None:
                    eq = equipo_de[tid]
                    color = COLOR_BGR[eq]
                    txt, _ = etiqueta(tid, conf)
                    estelas[tid].append((int(cx), int(cy)))
                else:
                    color, txt = COLOR_GRIS, "sin track"
                dibujar_mascara(frame, mask, color)
                x1,y1,x2,y2 = dets.xyxy[i].astype(int)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)
                poner_etiqueta(frame, x1, max(y1,44), txt, color)
        else:
            arr = robots_por_frame.get(idx, [])
            for x,y,tid,conf in arr:
                tid = int(tid)
                if equipo_de.get(tid) is None:
                    continue
                color = COLOR_BGR[equipo_de[tid]]
                x,y = int(x), int(y)
                estelas[tid].append((x,y))
                txt,_ = etiqueta(tid, conf)
                cv2.circle(frame, (x,y), 26, color, 3)
                poner_etiqueta(frame, x-30, max(y-30,44), txt, color)

        for tid, pts in estelas.items():
            eq = equipo_de.get(tid)
            color = COLOR_BGR[eq] if eq is not None else COLOR_GRIS
            p = list(pts)
            for k in range(1, len(p)):
                cv2.line(frame, p[k-1], p[k], color, 2)

        if MODO == "sam":
            mb, cen = detectar_balon_det(frame, idx)
            if cen is not None:
                bx, by = int(cen[0]), int(cen[1])
                if mb is not None:
                    dibujar_mascara(frame, mb, COLOR_BALON, alpha=0.5)
                estela_balon.append((bx,by))
                cv2.circle(frame, (bx,by), 10, COLOR_BALON, 2)
                poner_etiqueta(frame, bx+12, by, "balon", COLOR_BALON)
        else:
            if idx in balon_por_frame:
                bx = int(balon_por_frame[idx]["x_px"]); by = int(balon_por_frame[idx]["y_px"])
                estela_balon.append((bx,by))
                cv2.circle(frame, (bx,by), 12, COLOR_BALON, 3)
                poner_etiqueta(frame, bx+14, by, "balon", COLOR_BALON)

        p = list(estela_balon)
        for k in range(1, len(p)):
            cv2.line(frame, p[k-1], p[k], COLOR_BALON, 2)

        dibujar_hud(frame, idx, total, MODO)
        writer.write(frame)
        idx += 1
        if idx % 50 == 0:
            print(f"  frame {idx}")

    cap.release()
    writer.release()
    print(f"Guardado: {SALIDA}  (modo={MODO}, hasta frame {FRAME_FIN})")


if __name__ == "__main__":
    main()