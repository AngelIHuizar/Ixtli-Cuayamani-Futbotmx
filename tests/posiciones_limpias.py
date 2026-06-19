import pandas as pd
import matplotlib.pyplot as plt

dr = pd.read_csv("data/trayectorias_limpias.csv")
plt.figure(figsize=(5, 7))
ax = plt.gca()
ax.set_facecolor("#1b4332")
ax.add_patch(plt.Rectangle((0, 0), 182, 243, fill=False, ec="white", lw=2))
for tid in sorted(dr["tracker_id"].unique()):
    sub = dr[dr["tracker_id"] == tid]
    ax.scatter(sub["x_campo"], sub["y_campo"], s=3, label=f"ID {tid}")
ax.set_xlim(-25, 207); ax.set_ylim(268, -25)
ax.set_title("Posiciones limpias por ID")
ax.legend(markerscale=3, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
plt.savefig("outputs/posiciones_limpias.png", dpi=110, bbox_inches="tight")
print("Guardado: outputs/posiciones_limpias.png")