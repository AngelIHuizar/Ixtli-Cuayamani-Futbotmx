import pandas as pd

dr = pd.read_csv("data/trayectorias_final.csv")
db = pd.read_csv("data/balon_final.csv")

print("=== ROBOTS ===")
print("Filas:", len(dr), "| Robots únicos:", dr["tracker_id"].nunique())
print("Frames por robot:\n", dr["tracker_id"].value_counts())
print("Rango x_campo:", dr["x_campo"].min().round(1), "a", dr["x_campo"].max().round(1))
print("Rango y_campo:", dr["y_campo"].min().round(1), "a", dr["y_campo"].max().round(1))

print("\n=== BALÓN ===")
print("Frames con balón:", len(db), "de ~3600")
print("Rango x_campo:", db["x_campo"].min().round(1), "a", db["x_campo"].max().round(1))
print("Rango y_campo:", db["y_campo"].min().round(1), "a", db["y_campo"].max().round(1))