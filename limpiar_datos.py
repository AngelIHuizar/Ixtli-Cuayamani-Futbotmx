import pandas as pd

dr = pd.read_csv("data/trayectorias_final.csv")
n_inicial = len(dr)
ids_inicial = dr["tracker_id"].nunique()
print(f"Antes: {n_inicial} filas, {ids_inicial} IDs únicos\n")

# --- 1. Limpieza espacial: quitar puntos fuera de la cancha (margen 20 cm) ---
dentro = ((dr.x_campo >= -20) & (dr.x_campo <= 202) &
          (dr.y_campo >= -20) & (dr.y_campo <= 263))
n_fuera = (~dentro).sum()
dr = dr[dentro]
print(f"1. Puntos fuera de cancha eliminados: {n_fuera}")

# --- 2. Descartar IDs cortos (fragmentos de switch / destellos) ---
MIN_FRAMES = 100
conteo = dr["tracker_id"].value_counts()
ids_validos = conteo[conteo >= MIN_FRAMES].index
ids_descartados = conteo[conteo < MIN_FRAMES]
dr = dr[dr["tracker_id"].isin(ids_validos)]
print(f"2. IDs descartados por tener <{MIN_FRAMES} frames: {len(ids_descartados)}")
print(f"   (eran: {sorted(ids_descartados.index.tolist())})")

# --- Guardar ---
dr.to_csv("data/trayectorias_limpias.csv", index=False)
print(f"\nDespués: {len(dr)} filas, {dr['tracker_id'].nunique()} IDs conservados")
print(f"IDs conservados: {sorted(dr['tracker_id'].unique())}")
print("\nFrames por ID conservado:")
print(dr["tracker_id"].value_counts())