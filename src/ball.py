import cv2
import numpy as np

HSV_BAJO = np.array([5, 150, 120])
HSV_ALTO = np.array([18, 255, 255])


def mascara_balon(frame_bgr, hsv_bajo=HSV_BAJO, hsv_alto=HSV_ALTO):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, hsv_bajo, hsv_alto)


def candidatos_balon(frame_bgr, H=None, hsv_bajo=HSV_BAJO, hsv_alto=HSV_ALTO,
                     min_area=8, max_area=400, min_circ=0.55, margen_cm=10):
    mask = mascara_balon(frame_bgr, hsv_bajo, hsv_alto)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:      
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circ = 4 * np.pi * area / (perim * perim)
        if circ < min_circ:
            continue
        (x, y), _ = cv2.minEnclosingCircle(cnt)
        if H is not None:
            cm = cv2.perspectiveTransform(np.float32([[[x, y]]]), H)[0][0]
            if not (-margen_cm <= cm[0] <= 182 + margen_cm and
                    -margen_cm <= cm[1] <= 243 + margen_cm):
                continue
        out.append((float(x), float(y), float(area), float(circ)))
    return out


def detectar_balon(frame_bgr, H=None, **kw):
    cands = candidatos_balon(frame_bgr, H=H, **kw)
    return None if not cands else max(cands, key=lambda c: c[3])[:2]