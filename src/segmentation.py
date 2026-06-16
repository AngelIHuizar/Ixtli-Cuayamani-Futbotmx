# src/segmentation.py
"""
Segmentación de robots y balón con SAM 3.
Validado sobre los videos de cámara superior de la Copa FutBotMX. 
By: Cristina, Ángel, Miguel - INAOE
"""
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from src.homography import proyectar   
from transformers import Sam3Processor, Sam3Model

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
model.eval()

PROMPT_ROBOTS = "small robot"
PROMPT_BALON  = "orange ball"
PROMPT_MANOS  = "hand"
UMBRAL_DEFECTO = 0.25


def sam3_a_detections(results: dict) -> sv.Detections:
    """Convierte el output nativo de SAM 3 a sv.Detections."""
    masks  = results["masks"].cpu().numpy().astype(bool)
    xyxy   = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    return sv.Detections(xyxy=xyxy, mask=masks, confidence=scores)


def segmentar_con_texto(frame_bgr, concepto, umbral=UMBRAL_DEFECTO) -> sv.Detections:
    """Segmenta un concepto (vocabulario abierto) en un frame BGR de OpenCV."""
    image_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image_pil, text=concepto, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=umbral, mask_threshold=0.5,
        target_sizes=[image_pil.size[::-1]],
    )[0]
    return sam3_a_detections(results)


def _iou(caja_a, caja_b) -> float:
    """Intersección sobre unión de dos cajas [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = caja_a
    bx1, by1, bx2, by2 = caja_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def segmentar_robots(frame_bgr, umbral=UMBRAL_DEFECTO, filtrar_manos=True,
                     iou_mano=0.3) -> sv.Detections:
    """
    Segmenta los robots del campo. Si filtrar_manos=True, elimina detecciones
    que se solapen con manos de participantes (principal fuente de falsos positivos).
    """
    robots = segmentar_con_texto(frame_bgr, PROMPT_ROBOTS, umbral)
    if not filtrar_manos or len(robots) == 0:
        return robots

    manos = segmentar_con_texto(frame_bgr, PROMPT_MANOS, umbral)
    if len(manos) == 0:
        return robots

    keep = [
        not any(_iou(robots.xyxy[i], manos.xyxy[j]) > iou_mano
                for j in range(len(manos)))
        for i in range(len(robots))
    ]
    return robots[np.array(keep)]

def detectar_balon_sam(frame_bgr, H, ultimo=None, umbral=0.35,
                       max_salto_px=120):
    """
    Balón con SAM 3: mayor confianza, dentro del campo, cerca del anterior.
    Devuelve (x, y) en píxeles o None.
    """
    det = segmentar_con_texto(frame_bgr, "orange ball", umbral=umbral)
    if len(det) == 0:
        return None
    mejor, mejor_conf = None, 0.0
    for i in range(len(det)):
        x1, y1, x2, y2 = det.xyxy[i]
        cx, cy = float((x1+x2)/2), float((y1+y2)/2)
        conf = float(det.confidence[i]) if det.confidence is not None else 0
        # dentro del campo
        cm = proyectar([(cx, cy)], H)[0]
        if not (-15 <= cm[0] <= 197 and -15 <= cm[1] <= 258):
            continue
        # cerca del anterior (si hay)
        if ultimo is not None and np.hypot(cx-ultimo[0], cy-ultimo[1]) > max_salto_px:
            continue
        if conf > mejor_conf:
            mejor, mejor_conf = (cx, cy), conf
    return mejor