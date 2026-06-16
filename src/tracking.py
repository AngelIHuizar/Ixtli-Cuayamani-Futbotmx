# src/tracking.py
import cv2
import numpy as np
import pandas as pd
import supervision as sv
from trackers import ByteTrackTracker
from src.segmentation import segmentar_robots, detectar_balon_sam
from src.homography import cargar_H, proyectar


def _centroide(det, i):
    if det.mask is not None:
        ys, xs = np.where(det.mask[i])
        if len(xs) > 0:
            return float(xs.mean()), float(ys.mean())
    x1, y1, x2, y2 = det.xyxy[i]
    return float((x1 + x2) / 2), float((y1 + y2) / 2)


def rastrear_video(ruta_video,
                   salida_robots="data/trayectorias_finaltrack2min.csv",
                   salida_balon="data/balonfinaltrack2min.csv",
                   salida_video=None,                 
                   frame_inicio=0, frame_fin=None, paso=1,
                   umbral=0.25, filtrar_manos=False, min_frames_track=5,
                   umbral_balon=0.35, max_salto_balon_px=120, max_perdidos_balon=300):
    H = cargar_H()
    cap = cv2.VideoCapture(ruta_video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_fin is None:
        frame_fin = total
    print(f"Video: {total} frames, {fps:.1f} fps. Procesando {frame_inicio}–{frame_fin} (paso {paso})")

    # Writer del video anotado
    writer = None
    if salida_video:
        writer = cv2.VideoWriter(salida_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / paso, (w, h))

    # Anotadores
    box_ann   = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
    label_ann = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
    trace_ann = sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK)

    tracker = ByteTrackTracker()
    filas_robots, filas_balon = [], []
    ult_balon, perdidos_balon = None, 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)
    idx = frame_inicio
    while idx < frame_fin:
        ok, frame = cap.read()
        if not ok:
            break

        # ---------- ROBOTS ----------
        robots  = segmentar_robots(frame, umbral=umbral, filtrar_manos=filtrar_manos)
        tracked = tracker.update(robots)
        if tracked.tracker_id is not None:
            for i in range(len(tracked)):
                x, y = _centroide(tracked, i)
                filas_robots.append({
                    "frame": idx, "tracker_id": int(tracked.tracker_id[i]),
                    "x_px": x, "y_px": y, "equipo": "",
                    "confianza": float(tracked.confidence[i]) if tracked.confidence is not None else None,
                })

        # ---------- BALÓN ----------
        balon = detectar_balon_sam(frame, H, ultimo=ult_balon,
                                   umbral=umbral_balon, max_salto_px=max_salto_balon_px)
        if balon is not None:
            ult_balon, perdidos_balon = balon, 0
            filas_balon.append({"frame": idx, "x_px": balon[0], "y_px": balon[1]})
        else:
            perdidos_balon += 1
            if perdidos_balon > max_perdidos_balon:
                ult_balon = None

        # ---------- VIDEO ANOTADO ----------
        if writer is not None:
            anot = frame.copy()
            if tracked.tracker_id is not None and len(tracked) > 0:
                labels = [f"ID {int(t)}" for t in tracked.tracker_id]
                anot = box_ann.annotate(anot, tracked)
                anot = label_ann.annotate(anot, tracked, labels=labels)
                anot = trace_ann.annotate(anot, tracked)
            if balon is not None:
                cv2.circle(anot, (int(balon[0]), int(balon[1])), 14, (0, 0, 255), 3)
                cv2.putText(anot, "balon", (int(balon[0]) + 16, int(balon[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            writer.write(anot)

        idx += paso
        if paso > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        if (idx - frame_inicio) % 50 == 0:
            print(f"  frame {idx}/{frame_fin} | robots: {len(tracked)} | balón: {'sí' if balon else 'no'}")

    cap.release()
    if writer is not None:
        writer.release()

    # ----- Guardar robots -----
    dr = pd.DataFrame(filas_robots)
    dr = dr[dr["tracker_id"] != -1]
    if min_frames_track and len(dr):
        conteo  = dr["tracker_id"].value_counts()
        validos = conteo[conteo >= min_frames_track].index
        dr = dr[dr["tracker_id"].isin(validos)]
    if len(dr):
        cm = proyectar(dr[["x_px", "y_px"]].values, H)
        dr["x_campo"], dr["y_campo"] = cm[:, 0], cm[:, 1]
    dr.to_csv(salida_robots, index=False)

    # ----- Guardar balón -----
    db = pd.DataFrame(filas_balon)
    if len(db):
        cm = proyectar(db[["x_px", "y_px"]].values, H)
        db["x_campo"], db["y_campo"] = cm[:, 0], cm[:, 1]
    db.to_csv(salida_balon, index=False)

    print(f"\nGuardado: {salida_robots} ({len(dr)} filas, {dr['tracker_id'].nunique() if len(dr) else 0} robots)")
    print(f"Guardado: {salida_balon} ({len(db)} frames con balón)")
    if salida_video:
        print(f"Guardado: {salida_video}")
    return dr, db