import os
import cv2
import torch
import numpy as np
from sam3.model_builder import build_sam3_video_predictor

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# =========================
# CONFIGURACIÓN
# =========================
VIDEO_ORIGINAL = r"C:\Users\angel\sam3\futbotmx\videos\video-297_singular_display.mov"
VIDEO_LIGERO   = r"C:\Users\angel\sam3\futbotmx\videos\video_ligero_sam3.mp4"
VIDEO_SALIDA   = r"C:\Users\angel\sam3\futbotmx\videos\video_sam3_multiclass.mp4"

clases = [
    ("small black robot", "robot_equipo1"),
    ("small ball",        "pelota"),
    ("green field with white stripes","campo"),
    ("hands " ,"manos")

]

COLORES = {
    "robot_equipo1": (0,   255,   0),   # verde
    "pelota":        (0,   165, 255),   # naranja
    "campo":         (255,   0,   0),   # azul
    "manos":  (255,   255,   0),
}

MAX_FRAMES  = 80
FPS_SALIDA  = 10
ANCHO_SALIDA = 640

# =========================
# CREAR VIDEO LIGERO
# =========================
def crear_video_ligero(input_path, output_path, max_frames=80, fps_salida=10, ancho_salida=640):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {input_path}")

    fps_original = cap.get(cv2.CAP_PROP_FPS) or 30
    salto = max(1, int(round(fps_original / fps_salida)))

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("No se pudo leer el primer frame.")

    h, w = frame.shape[:2]
    nuevo_w = ancho_salida
    nuevo_h = int(h * ancho_salida / w)
    if nuevo_h % 2 != 0:
        nuevo_h += 1

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_salida, (nuevo_w, nuevo_h))

    frame_id = guardados = 0
    while ret and guardados < max_frames:
        if frame_id % salto == 0:
            out.write(cv2.resize(frame, (nuevo_w, nuevo_h)))
            guardados += 1
        ret, frame = cap.read()
        frame_id += 1

    cap.release()
    out.release()
    print(f"Video ligero: {output_path} | {guardados} frames | {nuevo_w}x{nuevo_h}")

# =========================
# DIBUJAR UNA CLASE
# =========================
def dibujar_clase(frame_out, outputs, color, nombre):
    if outputs is None:
        return frame_out

    H, W = frame_out.shape[:2]
    boxes    = outputs.get("out_boxes_xywh",   [])
    masks    = outputs.get("out_binary_masks", [])
    probs    = outputs.get("out_probs",        [])
    obj_ids  = outputs.get("out_obj_ids",      [])

    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        x1 = max(0, int((x - w/2) * W))
        y1 = max(0, int((y - h/2) * H))
        x2 = min(W-1, int((x + w/2) * W))
        y2 = min(H-1, int((y + h/2) * H))

        # Máscara semitransparente
        mask_r = cv2.resize((masks[i].astype(np.uint8)*255), (W, H), interpolation=cv2.INTER_NEAREST)
        overlay = frame_out.copy()
        overlay[mask_r > 0] = color
        frame_out = cv2.addWeighted(overlay, 0.35, frame_out, 0.65, 0)

        # Bounding box
        cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)

        # Etiqueta
        prob   = float(probs[i])   if i < len(probs)   else 0.0
        obj_id = int(obj_ids[i])   if i < len(obj_ids) else i
        label  = f"{nombre} {obj_id} | {prob:.2f}"
        cv2.putText(frame_out, label, (x1, max(20, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame_out

# =========================
# EXPORTAR VIDEO MULTICLASE
# =========================
def exportar_video_multiclase(video_path, outputs_por_clase, output_path, fps=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {video_path}")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("No se pudo leer el primer frame.")

    H, W = frame.shape[:2]
    out  = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    frame_idx = 0
    while ret:
        frame_draw = frame.copy()

        # Dibujar cada clase encima del mismo frame
        for nombre, frames_clase in outputs_por_clase.items():
            outputs = frames_clase.get(frame_idx, None)
            color   = COLORES.get(nombre, (255, 255, 255))
            frame_draw = dibujar_clase(frame_draw, outputs, color, nombre)

        cv2.putText(frame_draw, f"Frame {frame_idx}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame_draw)
        ret, frame = cap.read()
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Video guardado: {output_path}")

# =========================
# MAIN
# =========================
def main():
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    crear_video_ligero(VIDEO_ORIGINAL, VIDEO_LIGERO, MAX_FRAMES, FPS_SALIDA, ANCHO_SALIDA)
    torch.cuda.empty_cache()

    print("\nCargando modelo...")
    video_predictor = build_sam3_video_predictor()

    outputs_por_clase = {}

    for prompt, nombre in clases:
        print(f"\nProcesando: {nombre} | prompt: '{prompt}'")

        resp = video_predictor.handle_request(
            request={"type": "start_session", "resource_path": VIDEO_LIGERO}
        )
        sid = resp["session_id"]

        resp = video_predictor.handle_request(
            request={
                "type":        "add_prompt",
                "session_id":  sid,
                "frame_index": 0,
                "text":        prompt,
            }
        )
        n = len(resp.get("outputs", {}).get("out_obj_ids", []))
        print(f"  Frame 0: {n} objetos detectados")

        frames_clase = {}
        if n > 0:
            for tr in video_predictor.handle_stream_request(
                request={"type": "propagate_in_video", "session_id": sid}
            ):
                frames_clase[tr["frame_index"]] = tr["outputs"]

        outputs_por_clase[nombre] = frames_clase

        video_predictor.handle_request(
            request={"type": "close_session", "session_id": sid}
        )
        print(f"  Completado — {len(frames_clase)} frames")

    print("\nExportando video multiclase...")
    exportar_video_multiclase(VIDEO_LIGERO, outputs_por_clase, VIDEO_SALIDA, FPS_SALIDA)
    print("¡Listo!")

if __name__ == "__main__":
    main()
