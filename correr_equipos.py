from src.team_id import identificar_equipos

df, equipos = identificar_equipos(
    ruta_video=r"dataset/camara_superior/recorte_2min.mov",
    csv_robots="data/trayectorias_limpias.csv",
    salida_csv="data/trayectorias_equipos.csv",
)
print("\nResumen:", df.groupby("equipo")["tracker_id"].nunique(), "robots por equipo")