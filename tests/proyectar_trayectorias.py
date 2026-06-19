import pandas as pd
from src.homography import cargar_H, proyectar

H = cargar_H()
df = pd.read_csv("data/trayectorias.csv")

df = df[df["tracker_id"] != -1].reset_index(drop=True)

cm = proyectar(df[["x_px", "y_px"]].values, H)
df["x_campo"] = cm[:, 0]
df["y_campo"] = cm[:, 1]

df.to_csv("data/trayectorias_campo.csv", index=False)
print(f"Guardado: data/trayectorias_campo.csv ({len(df)} filas)")
print(df[["frame", "tracker_id", "x_campo", "y_campo"]].head(10))

fuera = df[(df.x_campo < -20) | (df.x_campo > 202) |
           (df.y_campo < -20) | (df.y_campo > 263)]
print(f"\nPuntos claramente fuera del campo: {len(fuera)} de {len(df)}")