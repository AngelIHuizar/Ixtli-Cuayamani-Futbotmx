import numpy as np
import pandas as pd

robots = pd.read_csv("data/trayectorias_equipos.csv")
balon  = pd.read_csv("data/balon_final.csv")
balon_por_frame = balon.set_index("frame")[["x_campo", "y_campo"]]

conteo_id = {}
for frame in balon_por_frame.index:
    bx, by = balon_por_frame.loc[frame, "x_campo"], balon_por_frame.loc[frame, "y_campo"]
    en_frame = robots[robots["frame"] == frame]
    if len(en_frame) == 0:
        continue
    d = np.hypot(en_frame["x_campo"] - bx, en_frame["y_campo"] - by)
    i_min = d.idxmin()
    if d.loc[i_min] <= 30:
        tid = int(en_frame.loc[i_min, "tracker_id"])
        conteo_id[tid] = conteo_id.get(tid, 0) + 1

print("Frames de posesión por ID:")
for tid in sorted(conteo_id, key=lambda k: -conteo_id[k]):
    eq = robots[robots["tracker_id"] == tid]["equipo"].iloc[0]
    print(f"  robot {tid} (equipo {eq}): {conteo_id[tid]} frames")