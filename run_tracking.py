from src.tracking import rastrear_video

dr, db = rastrear_video(
    ruta_video=r"dataset/camara_superior/recorte_2min.mov",
    salida_robots="data/trayectorias_final.csv",
    salida_balon="data/balon_final.csv",
    salida_video="outputs/demo_mascaras.mp4",     # video con máscaras de SAM
    frame_inicio=0,
    frame_fin=None, #None para todo el vídeo
    paso=1,
)
print("\nRobots:", dr.shape, "| Balón:", db.shape)