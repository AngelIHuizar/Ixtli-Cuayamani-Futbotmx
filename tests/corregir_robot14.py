import pandas as pd

df = pd.read_csv("data/trayectorias_equipos.csv")
df.loc[df["tracker_id"] == 14, "equipo"] = 0   # robot 14 es verde (validado en video)
df.to_csv("data/trayectorias_equipos.csv", index=False)

print("Corregido. Robots por equipo:")
print(df.groupby("equipo")["tracker_id"].unique())