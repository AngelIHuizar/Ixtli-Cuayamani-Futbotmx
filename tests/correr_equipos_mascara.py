from src.team_id_mascara import identificar_equipos_mascara

df, equipos = identificar_equipos_mascara(
    ruta_video=r"dataset/camara_superior/recorte_2min.mov",
    csv_robots="data/trayectorias_limpias.csv",
    salida_csv="data/trayectorias_equipos.csv",
)