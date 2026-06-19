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
MODO        = "csv"          # "sam" (máscaras reales) | "csv" (sin máscaras)
CSV_ROBOTS  = "data/trayectorias_equipos.csv"
CSV_BALON   = "data/balon_final.csv"
VIDEO       = r"dataset/camara_superior/recorte_2min.mov"
SALIDA      = "outputs/video_equipos2.mp4"

PASO          = 1            # render 1 de cada PASO frames 
UMBRAL_ROBOT  = 0.25         # umbral SAM robots
UMBRAL_BALON  = 0.35         # umbral SAM balón
FILTRAR_MANOS = False        # True = + lento 
MATCH_PX      = 45           # distancia máx. (px) para asociar máscara SAM <-> track del CSV
# =============================================================

def _hex_a_bgr(h):
    h = h.lstrip("#"); r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)
COLOR_BGR   = {0: _hex_a_bgr("#37c46a"), 1: _hex_a_bgr("#ff7a1a")}   # 0=Verde, 1=Oscuro
COLOR_BALON = (0, 0, 255)            # rojo (contrasta con el balón naranja)
COLOR_GRIS  = (170, 170, 170)        # detecciones sin track (segmentación sin ID)
LETRA       = {0: "B", 1: "A"}       # Verde=B, Oscuro=A
NOMBRE      = {0: "Verde", 1: "Oscuro"}


# ---------------- precómputos desde el CSV ----------------
df    = pd.read_csv(CSV_ROBOTS)
balon = pd.read_csv(CSV_BALON)

equipo_de = df.dropna(subset=["equipo"]).groupby("tracker_id")["equipo"].first().astype(int).to_dict()


def construir_slots(d):
    counts = d.tracker_id.value_counts()
    anclas = list(counts.index[:2])
    if len(anclas) < 2:
        return {t: 1 for t in counts.index}
    primer = {t: d[d.tracker_id == t].frame.min() for t in anclas}
    anclas.sort(key=lambda t: primer[t])          
    slot = {anclas[0]: 1, anclas[1]: 2}
    pos = {a: d[d.tracker_id == a].set_index("frame")[["x_px", "y_px"]] for a in anclas}
    for tid in counts.index:
        if tid in slot:
            continue
        rows = d[d.tracker_id == tid]
        dist = {}
        for a in anclas:
            pa = pos[a]
            ds = []
            for _, r in rows.iterrows():
                if len(pa) == 0:
                    continue
                f = pa.index[np.argmin(np.abs(pa.index.values - r.frame))]
                ds.append(np.hypot(r.x_px - pa.loc[f, "x_px"], r.y_px - pa.loc[f, "y_px"]))
            dist[a] = np.median(ds) if ds else 1e9
        slot[tid] = slot[min(dist, key=dist.get)]
    return slot

slot_de = {}
for eq in sorted(df.equipo.dropna().unique()):
    slot_de.update(construir_slots(df[df.equipo == eq].copy()))

frames_de = df.tracker_id.value_counts().to_dict()

robots_por_frame = {f: g[["x_px", "y_px", "tracker_id", "confianza"]].values
                    for f, g in df.groupby("frame")}
balon_por_frame  = balon.set_index("frame")[["x_px", "y_px"]].to_dict("index")


def slots_del_frame(tids):
    out = {}
    por_eq = defaultdict(list)
    for t in tids:
        eq = equipo_de.get(int(t))
        if eq is not None:
            por_eq[eq].append(int(t))
    for eq, lista in por_eq.items():
        lista.sort(key=lambda t: frames_de.get(t, 0), reverse=True)
        libres = {1, 2}
        for t in lista:
            pref = slot_de.get(t)
            if pref in libres:
                out[t] = pref; libres.discard(pref)
            elif libres:
                s = libres.pop(); out[t] = s
            else:
                out[t] = None       
    return out


def etiqueta(tid, slot):
    eq = equipo_de.get(tid)
    if eq is None:
        return f"ID{tid}", COLOR_GRIS
    r = f"R{slot}" if slot else "extra"
    return f"{LETRA[eq]}-{r}  ID{tid}", COLOR_BGR[eq]


def emparejar(cx, cy, idx):
    arr = robots_por_frame.get(idx)
    if arr is None or len(arr) == 0:
        return None, None
    d = np.hypot(arr[:, 0] - cx, arr[:, 1] - cy)
    j = int(np.argmin(d))
    if d[j] > MATCH_PX:
        return None, None
    return int(arr[j, 2]), (float(arr[j, 3]) if not np.isnan(arr[j, 3]) else None)


def dibujar_mascara(frame, mask, color, alpha=0.45):
    overlay = frame.copy()
    overlay[mask] = color
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    m8 = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, cnts, -1, color, 2)


def poner_etiqueta(frame, x, y, texto, color):
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y), color, -1)
    cv2.putText(frame, texto, (x + 4, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 2, cv2.LINE_AA)


def dibujar_hud(frame, idx, total, modo):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    txt = ("SAM 3: segmentacion  |  ByteTrack: tracking  |  DINOv3+KMeans: equipos"
           if modo == "sam" else "Tracking (CSV)  |  ByteTrack + DINOv3+KMeans")
    cv2.putText(frame, txt, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    # clave de color (A=Oscuro primero)
    x0 = w - 230
    for eq, dx in ((1, 0), (0, 110)):
        cv2.rectangle(frame, (x0 + dx, 8), (x0 + dx + 18, 26), COLOR_BGR[eq], -1)
        cv2.putText(frame, f"Eq.{LETRA[eq]} {NOMBRE[eq]}", (x0 + dx + 22, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"frame {idx}/{total}", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


if MODO == "sam":
    from src.segmentation import segmentar_robots, segmentar_con_texto


def detectar_balon_det(frame, idx):
    """Devuelve (mask, (cx,cy)) del balón emparejado al CSV, o (None,None)."""
    det = segmentar_con_texto(frame, "orange ball", umbral=UMBRAL_BALON)
    if len(det) == 0:
        return None, None
    cents = [((det.xyxy[i][0] + det.xyxy[i][2]) / 2, (det.xyxy[i][1] + det.xyxy[i][3]) / 2)
             for i in range(len(det))]
    ref = balon_por_frame.get(idx)
    if ref is not None:
        d = [np.hypot(cx - ref["x_px"], cy - ref["y_px"]) for cx, cy in cents]
        i = int(np.argmin(d))
    else:
        i = int(np.argmax(det.confidence)) if det.confidence is not None else 0
    m = det.mask[i] if det.mask is not None else None
    return m, cents[i]


def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir el video: {VIDEO}")
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(SALIDA, cv2.VideoWriter_fourcc(*"mp4v"), fps / PASO, (w, h))

    estelas = defaultdict(lambda: deque(maxlen=25))
    estela_balon = deque(maxlen=20)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % PASO != 0:
            idx += 1
            continue

        # ---------- ROBOTS ----------
        if MODO == "sam":
            dets = segmentar_robots(frame, umbral=UMBRAL_ROBOT, filtrar_manos=FILTRAR_MANOS)
            # 1) emparejar todas las detecciones a su track
            empar = []   # (mask, box, tid, conf)
            for i in range(len(dets)):
                mask = dets.mask[i]
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    continue
                cx, cy = float(xs.mean()), float(ys.mean())
                tid, conf = emparejar(cx, cy, idx)
                empar.append((mask, dets.xyxy[i].astype(int), cx, cy, tid, conf))
            # 2) resolver slots por frame (sin colisiones)
            slots = slots_del_frame([e[4] for e in empar if e[4] is not None])
            # 3) dibujar
            for mask, box, cx, cy, tid, conf in empar:
                if tid is not None:
                    eq = equipo_de.get(tid)
                    color = COLOR_BGR[eq] if eq is not None else COLOR_GRIS
                    txt, _ = etiqueta(tid, slots.get(tid))
                    if conf is not None:
                        txt += f"  .{int(conf*100):02d}"
                    estelas[tid].append((int(cx), int(cy)))
                else:
                    color, txt = COLOR_GRIS, "sin track"
                dibujar_mascara(frame, mask, color)
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                poner_etiqueta(frame, x1, max(y1, 44), txt, color)
        else:  # modo CSV: círculos desde el track, sin máscara
            arr = robots_por_frame.get(idx, [])
            slots = slots_del_frame([int(r[2]) for r in arr]) if len(arr) else {}
            for x, y, tid, conf in arr:
                tid = int(tid)
                eq = equipo_de.get(tid)
                if eq is None:
                    continue
                color = COLOR_BGR[eq]
                x, y = int(x), int(y)
                estelas[tid].append((x, y))
                txt, _ = etiqueta(tid, slots.get(tid))
                if not np.isnan(conf):
                    txt += f"  .{int(conf*100):02d}"
                cv2.circle(frame, (x, y), 26, color, 3)
                poner_etiqueta(frame, x - 30, max(y - 30, 44), txt, color)

        # estelas
        for tid, pts in estelas.items():
            eq = equipo_de.get(tid)
            color = COLOR_BGR[eq] if eq is not None else COLOR_GRIS
            p = list(pts)
            for k in range(1, len(p)):
                cv2.line(frame, p[k-1], p[k], color, 2)

        # ---------- BALÓN ----------
        if MODO == "sam":
            mb, cen = detectar_balon_det(frame, idx)
            if cen is not None:
                bx, by = int(cen[0]), int(cen[1])
                if mb is not None:
                    dibujar_mascara(frame, mb, COLOR_BALON, alpha=0.5)
                estela_balon.append((bx, by))
                cv2.circle(frame, (bx, by), 10, COLOR_BALON, 2)
                poner_etiqueta(frame, bx + 12, by, "balon", COLOR_BALON)
        else:
            if idx in balon_por_frame:
                bx = int(balon_por_frame[idx]["x_px"]); by = int(balon_por_frame[idx]["y_px"])
                estela_balon.append((bx, by))
                cv2.circle(frame, (bx, by), 12, COLOR_BALON, 3)
                poner_etiqueta(frame, bx + 14, by, "balon", COLOR_BALON)

        p = list(estela_balon)
        for k in range(1, len(p)):
            cv2.line(frame, p[k-1], p[k], COLOR_BALON, 2)

        dibujar_hud(frame, idx, total, MODO)
        writer.write(frame)

        idx += 1
        if idx % 100 == 0:
            print(f"  frame {idx}/{total}")

    cap.release()
    writer.release()
    print(f"Guardado: {SALIDA}  (modo={MODO})")


if __name__ == "__main__":
    main()
