from src.ball_tracking import rastrear_balon
import matplotlib.pyplot as plt

df = rastrear_balon(
    r"dataset/camara_superior/IMG_9933.mov",
    frame_fin=3000,
    max_salto_px=90,     
    max_perdidos=300,     
)
print(df.head())

plt.figure(figsize=(5, 7))
ax = plt.gca()
ax.set_facecolor("#1b4332")
ax.add_patch(plt.Rectangle((0, 0), 182, 243, fill=False, ec="white", lw=2))
ax.scatter(df["x_campo"], df["y_campo"], s=6, color="orange")   # solo puntos
ax.set_xlim(-10, 192); ax.set_ylim(253, -10)
ax.set_title("Trayectoria del balon (cm)")
plt.savefig("outputs/trayectoria_balon.png", dpi=110, bbox_inches="tight")
print("Guardado: outputs/trayectoria_balon.png")