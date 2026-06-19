import pandas as pd
from src.events import detectar_goles

balon = pd.read_csv("data/balon_final.csv")
goles = detectar_goles(balon)

print(f"{'='*50}\nGOLES DETECTADOS: {len(goles)}\n{'='*50}")
if len(goles):
    print(goles.to_string(index=False))
    print("\nMarcador:")
    for eq, n in goles["equipo_nombre"].value_counts().items():
        print(f"  {eq}: {n}")
else:
    print("  (ninguno)")