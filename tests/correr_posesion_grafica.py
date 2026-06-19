import matplotlib.pyplot as plt
from src.possesion import calcular_posesion

res = calcular_posesion()  
nombres = {0: "B", 1: "A"}
colores = {0: "#2e8b57", 1: "#d2691e"}

fig, ax = plt.subplots(figsize=(7, 2.2))
inicio = 0
for eq, pct in sorted(res.items()):
    ax.barh(0, pct, left=inicio, color=colores[eq],
            label=f"{nombres[eq]} {pct:.0f}%")
    ax.text(inicio + pct/2, 0, f"{nombres[eq]}\n{pct:.0f}%",
            ha="center", va="center", color="white", fontweight="bold")
    inicio += pct
ax.set_xlim(0, 100); ax.set_ylim(-0.5, 0.5)
ax.axis("off")
ax.set_title("Posesión del balón por equipo", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/posesion_grafica.png", dpi=130, bbox_inches="tight")
print("Guardado: outputs/posesion.png")