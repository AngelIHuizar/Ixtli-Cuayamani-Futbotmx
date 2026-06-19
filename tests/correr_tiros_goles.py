import pandas as pd
from src.events import detectar_tiros, detectar_goles, consolidar_goles

balon = pd.read_csv("data/balon_final.csv")

tiros = detectar_tiros(balon)
goles = consolidar_goles(detectar_goles(balon))

print(f"{'='*55}")
print(f"TIROS A GOL: {len(tiros)}   |   GOLES: {len(goles)}")
print(f"{'='*55}")

print("\n--- TIROS ---")
print(tiros.to_string(index=False) if len(tiros) else "  (ninguno)")
print("\n--- GOLES ---")
print(goles.to_string(index=False) if len(goles) else "  (ninguno)")

print(f"\n{'='*55}\nRESUMEN POR EQUIPO\n{'='*55}")
for eq, nombre in [(0, "B"), (1, "A")]:
    t = (tiros["equipo"] == eq).sum() if len(tiros) else 0
    g = (goles["equipo"] == eq).sum() if len(goles) else 0
    efec = f"{100*g/t:.0f}%" if t > 0 else "—"
    print(f"  {nombre}: {t} tiros, {g} goles  (efectividad {efec})")