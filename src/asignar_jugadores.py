# src/asignar_jugadores.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def _frames_de(d, tid):
    return set(d[d.tracker_id == tid].frame.values)


def _construir_slots(d):
    counts = d.tracker_id.value_counts()
    anclas = list(counts.index[:2])
    if len(anclas) < 2:
        return {t: 1 for t in counts.index}

    primer = {t: d[d.tracker_id == t].frame.min() for t in anclas}
    anclas.sort(key=lambda t: primer[t])
    slot = {anclas[0]: 1, anclas[1]: 2}
    frames_ancla = {a: _frames_de(d, a) for a in anclas}
    pos = {a: d[d.tracker_id == a].set_index("frame")[["x_px", "y_px"]] for a in anclas}

    for tid in counts.index:
        if tid in slot:
            continue
        rows = d[d.tracker_id == tid]
        fr_tid = set(rows.frame.values)

        coexiste = {a: len(fr_tid & frames_ancla[a]) > 0 for a in anclas}
        candidatas = [a for a in anclas if not coexiste[a]]
        if len(candidatas) == 1:
            slot[tid] = slot[candidatas[0]]          
            continue
        if len(candidatas) == 0:
            candidatas = anclas                       

        dist = {}
        for a in candidatas:
            pa = pos[a]
            ds = []
            for _, r in rows.iterrows():
                if len(pa) == 0:
                    continue
                f = pa.index[np.argmin(np.abs(pa.index.values - r.frame))]
                ds.append(np.hypot(r.x_px - pa.loc[f, "x_px"], r.y_px - pa.loc[f, "y_px"]))
            dist[a] = np.median(ds) if ds else 1e9
        slot[tid] = slot[min(dist, key=dist.get)]
    return slot


def asignar_jugadores(csv_equipos="data/trayectorias_equipos.csv",
                      salida_csv="data/trayectorias_jugadores.csv"):
    df = pd.read_csv(csv_equipos)
    df = df.dropna(subset=["equipo"]).copy()
    df["equipo"] = df["equipo"].astype(int)

    slot_de = {}
    for eq in sorted(df.equipo.unique()):
        slot_de.update(_construir_slots(df[df.equipo == eq].copy()))

    df["jugador"] = df["tracker_id"].map(slot_de)
    df.to_csv(salida_csv, index=False)

    print("Jugadores por equipo:")
    for eq in sorted(df.equipo.unique()):
        nombre = "Verde (B)" if eq == 0 else "Oscuro (A)"
        sub = df[df.equipo == eq]
        for jug in sorted(sub.jugador.unique()):
            tids = sorted(sub[sub.jugador == jug].tracker_id.unique())
            n = len(sub[sub.jugador == jug])
            print(f"  {nombre} · Jugador {jug}: tracks {tids} ({n} frames)")
    print(f"\nGuardado: {salida_csv}")
    return df