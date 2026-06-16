from src.tracking import rastrear_video

dr, db = rastrear_video(
    ruta_video=r"dataset/camara_superior/recorte_2min.mov",
    salida_robots="data/trayectorias_final.csv",
    salida_balon="data/balon_final.csv",
    salida_video="outputs/demo_anotado.mp4",     # <-- el video anotado
    frame_inicio=0,
    frame_fin=None,        # None = todo el recorte de 3 min
    paso=1,
)
print("\nRobots:", dr.shape, "| Balón:", db.shape)