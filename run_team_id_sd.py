from src.team_id_sd import clasificar
df, eq = clasificar(ruta_video=r"dataset/camara_superior/recorte_2min.mov")
print(df.groupby("equipo")["tracker_id"].unique())