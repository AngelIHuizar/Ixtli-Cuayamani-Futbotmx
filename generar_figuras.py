"""
Genera las 6 figuras del entregable en un archivo.
Las imágenes salen en la subcarpeta ./outputs
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

# ========================================================
CSV_EQUIPOS   = "data/trayectorias_equipos.csv"
CSV_JUGADORES = "data/trayectorias_jugadores.csv"
CSV_BALON     = "data/balon_final.csv"
OUT           = "figuras"          
DPI           = 220                
FMT           = "png"              # "png" | "pdf" | "svg" 
TOTAL_FRAMES_VIDEO = 3607          

# Parámetros Mapas de Calor
PARKED_RG = 16.0         
BINS      = (18, 24)     # celdas ~10x10 cm
SIGMA     = 0.8          # suavizado gaussiano
# ===================================================================

os.makedirs(OUT, exist_ok=True)

# Colores consistentes
VERDE, VERDE_CLARO   = "#2f9e5e", "#37c46a"     # Equipo B
OSCURO, OSCURO_CLARO = "#d4691e", "#ff7a1a"     # Equipo A
AMARILLA, AZUL       = "#e9c11a", "#2f6fd0"
BALON                = "#ffd400"
FONDO_CANCHA         = "#0e2a1d"
LINEAS               = "#cfe8d8"

NOMBRE    = {0: "Equipo B", 1: "Equipo A"}      # 0=Verde, 1=Oscuro
COLOR_NOM = {0: "Verde", 1: "Oscuro"}
COL       = {0: VERDE, 1: OSCURO}

XMAX, YMAX = 182, 243        # x = ancho, y = largo
GX0, GX1   = 61, 121         # boca de portería

# ================= CARGA  =================
rob_eq = pd.read_csv(CSV_EQUIPOS)      
rob_ju = pd.read_csv(CSV_JUGADORES)    
bal    = pd.read_csv(CSV_BALON)

rob_eq['y_campo'] = YMAX - rob_eq['y_campo']
rob_ju['y_campo'] = YMAX - rob_ju['y_campo']
bal['y_campo']    = YMAX - bal['y_campo']
# ===============================================================

# ----------------------- Eventos (inline) -----------------------
def _en_ancho(x):
    return GX0 <= x <= GX1

def _zona_gol(x, y):
    if not _en_ancho(x):
        return None
    if y <= 0:
        return "azul"    
    if y >= YMAX:
        return "amarilla"        
    return None

def detectar_llegadas_area(df, fps=30, espera_seg=2.0, prof=25.0):
    df = df.sort_values("frame").reset_index(drop=True)
    out, ultimo, dentro = [], -1e9, None
    for i in range(len(df)):
        x, y, fr = df.x_campo[i], df.y_campo[i], df.frame[i]
        zona = None
        if _en_ancho(x):
            if y <= prof:
                zona = "azul"            
            elif y >= YMAX - prof:
                zona = "amarilla"        
        if zona is not None and zona != dentro and (fr - ultimo) / fps >= espera_seg:
            eq = 1 if zona == "azul" else 0   
            out.append({"frame": int(fr), "porteria": zona, "equipo": eq,
                        "x": round(float(x), 1), "y": round(float(y), 1)})
            ultimo = fr
        dentro = zona
    return pd.DataFrame(out)

def detectar_goles(df, fps=30, espera_seg=3.0, min_frames_gol=30, salto_max=15):
    df = df.sort_values("frame").reset_index(drop=True)
    out, ultimo_gol, i, n = [], -1e9, 0, len(df)
    while i < n:
        x, y, fr = df.x_campo[i], df.y_campo[i], df.frame[i]
        if (fr - ultimo_gol) / fps < espera_seg:
            i += 1; continue
        zona = _zona_gol(x, y)
        if zona is None:
            i += 1; continue
        primer, ultimo_atras, fprev, j = i, fr, fr, i + 1
        while j < n:
            if df.frame[j] - fprev > salto_max:
                break
            if _zona_gol(df.x_campo[j], df.y_campo[j]) != zona:
                break
            ultimo_atras = df.frame[j]; fprev = df.frame[j]; j += 1
        dur = ultimo_atras - df.frame[primer]
        if dur >= min_frames_gol:
            eq = 1 if zona == "azul" else 0   
            out.append({"frame": int(df.frame[primer]), "porteria": zona, "equipo": eq,
                        "x": round(float(x), 1), "y": round(float(y), 1),
                        "frames_sostenido": int(dur)})
            ultimo_gol = ultimo_atras
        i = max(j, i + 1)
    return pd.DataFrame(out)

def tiros_a_gol(df, umbral_v=2.5, vmax_min=3.5, corr=(40, 140), merge_gap=15):
    d = df.sort_values("frame").reset_index(drop=True)
    fr = d.frame.values; x = d.x_campo.values; y = d.y_campo.values
    dfr = np.diff(fr); dx = np.diff(x); dy = np.diff(y); ok = dfr <= 3
    vx = np.where(ok, dx / dfr, np.nan)
    vy = np.where(ok, dy / dfr, np.nan)
    sp = np.hypot(vx, vy)
    fr2, x2, y2 = fr[1:], x[1:], y[1:]
    
    m = (sp >= umbral_v) & (vy < 0) & (x2 > corr[0]) & (x2 < corr[1]) & (y2 > 0) & (y2 < YMAX/2)
    idx = np.where(m)[0]
    eps = []
    if len(idx):
        s = p = idx[0]
        for k in idx[1:]:
            if fr2[k] - fr2[p] > merge_gap:
                eps.append((s, p)); s = k
            p = k
        eps.append((s, p))
    out = []
    for a, b in eps:
        vmax = np.nanmax(sp[a:b + 1])
        if vmax >= vmax_min and (y2[b] - y2[a]) <= 0: 
            k = a + int(np.argmax(sp[a:b + 1]))
            out.append({"frame": int(fr2[k]), "x": float(x2[k]), "y": float(y2[k]),
                        "vmax": float(vmax), "equipo": 1})   
    return pd.DataFrame(out)

# ----------------------------------------------
def dibujar_cancha(ax, campo=FONDO_CANCHA, linea=LINEAS, top=12, net=False):
    ax.add_patch(Rectangle((0, 0), XMAX, YMAX, color=campo, zorder=0))
    if net:
        ax.add_patch(Rectangle((GX0, -17), GX1-GX0, 17, color="#16352a",
                               ec=linea, lw=0.8, zorder=1))
    ax.add_patch(Rectangle((0, 0), XMAX, YMAX, fill=False, ec=linea, lw=1.6, zorder=3))
    ax.plot([0, XMAX], [YMAX/2, YMAX/2], color=linea, lw=1.2, zorder=3)
    th = np.linspace(0, 2*np.pi, 80)
    ax.plot(XMAX/2 + 22*np.cos(th), YMAX/2 + 22*np.sin(th), color=linea, lw=1.2, zorder=3)
    
    ax.add_patch(Rectangle((GX0, -4), GX1-GX0, 4, color=AZUL, zorder=4))
    ax.add_patch(Rectangle((GX0, YMAX), GX1-GX0, 4, color=AMARILLA, zorder=4))
    
    ax.add_patch(Rectangle((GX0-12, 0), GX1-GX0+24, 25, fill=False, ec=linea, lw=1, zorder=3))
    ax.add_patch(Rectangle((GX0-12, YMAX-25), GX1-GX0+24, 25, fill=False, ec=linea, lw=1, zorder=3))
    ax.set_xlim(-6, XMAX+6); ax.set_ylim(-20, YMAX+top) 
    ax.set_aspect("equal"); ax.axis("off")

# ----------------- Funciones para Heatmaps  -----------------
def filtrar_parados(df, umbral_rg=PARKED_RG):
    if "tracker_id" not in df.columns:
        return df
    rg = df.groupby("tracker_id").agg(sx=("x_campo", "std"), sy=("y_campo", "std"))
    rg["rg"] = np.hypot(rg["sx"], rg["sy"])
    parados = rg.index[rg["rg"] < umbral_rg].tolist()
    if parados:
        print(f"  -> Robots excluidos del heatmap (estáticos): {parados}")
        return df[~df["tracker_id"].isin(parados)]
    return df

def heatmap_pcolormesh(ax, sub, cmap_name, bins=BINS, sigma=SIGMA):
    H, xe, ye = np.histogram2d(
        sub["x_campo"], sub["y_campo"], bins=bins, range=[[0, XMAX], [0, YMAX]]
    )
    if sigma > 0:
        H = gaussian_filter(H, sigma)
    
    C = H / H.max() if H.max() > 0 else H
    malla = ax.pcolormesh(xe, ye, C.T, cmap=cmap_name, vmin=0, vmax=1,
                          zorder=1, shading="auto")
    return malla

# --------------------------------------------------------------------------

def frames_ambos(rob):
    return sorted(set(rob[rob.equipo == 0].frame) & set(rob[rob.equipo == 1].frame))

def posesion_por_frame(rob, umbral=30.0):
    bpf = bal.set_index("frame")[["x_campo", "y_campo"]]
    tiene_jug = "jugador" in rob.columns
    reg = []
    for f in bpf.index:
        bx, by = bpf.loc[f, "x_campo"], bpf.loc[f, "y_campo"]
        enf = rob[rob.frame == f]
        if len(enf) == 0:
            continue
        d = np.hypot(enf.x_campo - bx, enf.y_campo - by)
        i = d.idxmin()
        if d.loc[i] <= umbral:
            fila = [f, int(enf.loc[i, "equipo"])]
            if tiene_jug:
                fila.append(int(enf.loc[i, "jugador"]))
            reg.append(fila)
    cols = ["frame", "equipo"] + (["jugador"] if tiene_jug else [])
    return pd.DataFrame(reg, columns=cols)

def guardar(fig, nombre, facecolor="white"):
    ruta = os.path.join(OUT, f"{nombre}.{FMT}")
    fig.savefig(ruta, dpi=DPI, facecolor=facecolor, bbox_inches="tight")
    plt.close(fig)
    print(f"  ok  {ruta}")

def control_espacio(rob, nx=60, ny=80, solo_frame=None):
    gx = np.linspace(0, XMAX, nx); gy = np.linspace(0, YMAX, ny)
    GX, GY = np.meshgrid(gx, gy)
    cells = np.column_stack([GX.ravel(), GY.ravel()])
    def split(frame):
        sub = rob[rob.frame == frame]
        P = sub[["x_campo", "y_campo"]].values
        eq = sub["equipo"].values
        d = np.sqrt(((cells[:, None, :] - P[None, :, :])**2).sum(2))
        near = eq[d.argmin(1)]
        return (near == 0).mean(), (near == 1).mean(), near.reshape(ny, nx)
    if solo_frame is not None:
        b, a, grid = split(solo_frame)
        return 100*b, 100*a, grid
    fr = frames_ambos(rob); sb = sa = 0.0
    for f in fr:
        b, a, _ = split(f); sb += b; sa += a
    return 100*sb/len(fr), 100*sa/len(fr), len(fr)


# =============================== FIGURAS ===============================
def fig_mapa_calor_equipos():
    # Más ancho (10.5) para que quepan las colorbars sin aplastar la cancha
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.0)); fig.patch.set_facecolor("white")
    df_fil = filtrar_parados(rob_eq)
    cmaps = {0: "Greens", 1: "Oranges"}
    
    for ax, eq in zip(axes, [0, 1]):
        # Fondo claro para que resalte pcolormesh
        dibujar_cancha(ax, campo="#f4f6f4", linea="#6b7d72")
        sub = df_fil[df_fil.equipo == eq]
        malla = heatmap_pcolormesh(ax, sub, cmaps[eq])
        
        # Agregamos la colorbar idéntica a tu script viejo
        fig.colorbar(malla, ax=ax, fraction=0.046, pad=0.04, label="Densidad relativa")
        
        ax.set_title(f"{NOMBRE[eq]} ", color=COL[eq],
                     fontsize=14, fontweight="bold", pad=8, loc="center")

    axes[0].text(0.5, -0.06, "ataca → amarilla (arriba)", transform=axes[0].transAxes,
                 ha="center", fontsize=8, color="#555")
    axes[1].text(0.5, -0.06, "ataca → azul (abajo)", transform=axes[1].transAxes,
                 ha="center", fontsize=8, color="#555")
    fig.suptitle("Mapa de calor de ocupación por equipo", fontsize=15, fontweight="bold", y=0.97, x=0.5, ha="center")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    guardar(fig, "mapa_calor_equipos")

def fig_mapa_calor_jugadores():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.6)); fig.patch.set_facecolor("white")
    df_fil = filtrar_parados(rob_ju)
    # combos: (equipo, jugador, color_titulo, cmap)
    combos = [(1, 1, OSCURO, "Oranges"), (0, 1, VERDE, "Greens"), 
              (1, 2, OSCURO_CLARO, "Oranges"), (0, 2, VERDE_CLARO, "Greens")]
    
    for ax, (eq, jug, color_t, cmap_name) in zip(axes.ravel(), combos):
        dibujar_cancha(ax, campo="#f4f6f4", linea="#6b7d72")
        sub = df_fil[(df_fil.equipo == eq) & (df_fil.jugador == jug)]
        malla = heatmap_pcolormesh(ax, sub, cmap_name)
        
        fig.colorbar(malla, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{NOMBRE[eq]} · Robot {jug}  ({len(sub)} fr)",
                     color=color_t, fontsize=12, fontweight="bold", pad=6, loc="center")
                     
    fig.suptitle("Mapa de calor por jugador (R1 arriba · R2 abajo)",
                 fontsize=15, fontweight="bold", y=0.975)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    guardar(fig, "mapa_calor_jugadores")

def fig_posesion():
    pos = posesion_por_frame(rob_ju); tot = len(pos)
    posc = pos[pos.frame.isin(set(frames_ambos(rob_ju)))]; totc = len(posc)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 5.4),
                                   gridspec_kw={"height_ratios": [1.1, 1]})
    fig.patch.set_facecolor("white")
    filas = [(f"Posesión total\n({tot} fr con balón)",
              100*(pos.equipo == 0).sum()/tot, 100*(pos.equipo == 1).sum()/tot),
             (f"Cara a cara\n({totc} fr, ambos en cancha)",
              100*(posc.equipo == 0).sum()/totc, 100*(posc.equipo == 1).sum()/totc)]
    for i, (lab, b, a) in enumerate(filas):
        y = len(filas)-1-i
        ax1.barh(y, b, color=VERDE, edgecolor="white")
        ax1.barh(y, a, left=b, color=OSCURO, edgecolor="white")
        ax1.text(b/2, y, f"{b:.0f}%", ha="center", va="center", color="white", fontweight="bold")
        ax1.text(b+a/2, y, f"{a:.0f}%", ha="center", va="center", color="white", fontweight="bold")
        ax1.text(-1, y, lab, ha="right", va="center", fontsize=9)
    ax1.set_xlim(0, 100); ax1.set_ylim(-0.6, len(filas)-0.4); ax1.axis("off")
    ax1.text(0, len(filas)-0.2, "Equipo B", color=VERDE, fontsize=10, fontweight="bold")
    ax1.text(100, len(filas)-0.2, "Equipo A", color=OSCURO, fontsize=10,
             fontweight="bold", ha="right")
    barras = [("B · R1", 0, 1, VERDE), ("B · R2", 0, 2, VERDE_CLARO),
              ("A · R1", 1, 1, OSCURO), ("A · R2", 1, 2, OSCURO_CLARO)]
    xs = np.arange(len(barras))
    for x, (lab, eq, jug, c) in zip(xs, barras):
        v = 100*((pos.equipo == eq) & (pos.jugador == jug)).sum()/tot
        ax2.bar(x, v, color=c, edgecolor="white")
        ax2.text(x, v+1, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.set_xticks(xs); ax2.set_xticklabels([b[0] for b in barras], fontsize=9)
    ax2.set_ylim(0, 50); ax2.set_ylabel("% del balón")
    ax2.set_title("Posesión por jugador", fontsize=11, fontweight="bold")
    ax2.text(0, 3, "* 5 tracks\nfragmentados", fontsize=7, color="#888", ha="center")
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)
    fig.suptitle("Posesión del balón", fontsize=15, fontweight="bold", y=0.99, x=0.5, ha="center")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    guardar(fig, "posesion")

def fig_voronoi():
    pctB, pctA, nfr = control_espacio(rob_eq)
    set_bal = set(bal.frame); cands = []
    for f in frames_ambos(rob_eq):
        sub = rob_eq[rob_eq.frame == f]
        if (sub.equipo == 0).sum() == 2 and (sub.equipo == 1).sum() == 2 and f in set_bal:
            cands.append(f)
    f = cands[len(cands)//2]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 7.4)); fig.patch.set_facecolor("white")

    # ---- IZQUIERDA: por EQUIPO ----
    b, a, grid_eq = control_espacio(rob_eq, nx=XMAX, ny=YMAX, solo_frame=f)
    dibujar_cancha(axL, campo="#1b4332", linea="white", top=16)
    rgba = np.zeros((YMAX, XMAX, 4))
    rgba[grid_eq == 0] = (*mc.to_rgb(VERDE), 0.5)
    rgba[grid_eq == 1] = (*mc.to_rgb(OSCURO), 0.5)
    axL.imshow(rgba, origin="lower", extent=[0, XMAX, 0, YMAX], zorder=1)
    sub = rob_eq[rob_eq.frame == f]
    for _, r in sub.iterrows():
        axL.scatter(r.x_campo, r.y_campo, s=260, color=COL[int(r.equipo)],
                    edgecolor="white", lw=2.2, zorder=5)
    axL.set_title(f"Por equipo\nB {b:.0f}%  ·  A {a:.0f}%",
                  fontsize=12, fontweight="bold", pad=6)

    # ---- DERECHA: por JUGADOR ----
    sub_ju = rob_ju[rob_ju.frame == f]
    dibujar_cancha(axR, campo="#1b4332", linea="white", top=16)
    color_jug = {(0,1): VERDE, (0,2): VERDE_CLARO, (1,1): OSCURO, (1,2): OSCURO_CLARO}
    etiq_jug  = {(0,1): "B·R1", (0,2): "B·R2", (1,1): "A·R1", (1,2): "A·R2"}
    P, cols_robot, labs = [], [], []
    for _, r in sub_ju.iterrows():
        P.append([r.x_campo, r.y_campo])
        key = (int(r.equipo), int(r.jugador))
        cols_robot.append(color_jug.get(key, "#888"))
        labs.append(etiq_jug.get(key, "?"))
    P = np.array(P)
    if len(P):
        gx = np.arange(XMAX); gy = np.arange(YMAX)
        GX, GY = np.meshgrid(gx, gy)
        cells = np.column_stack([GX.ravel(), GY.ravel()])
        d = np.sqrt(((cells[:,None,:] - P[None,:,:])**2).sum(2))
        near = d.argmin(1).reshape(YMAX, XMAX)
        rgba2 = np.zeros((YMAX, XMAX, 4))
        for k, c in enumerate(cols_robot):
            rgba2[near == k] = (*mc.to_rgb(c), 0.5)
        axR.imshow(rgba2, origin="lower", extent=[0, XMAX, 0, YMAX], zorder=1)
        for (px, py), c, lab in zip(P, cols_robot, labs):
            axR.scatter(px, py, s=260, color=c, edgecolor="white", lw=2.2, zorder=5)
            axR.text(px, py-9, lab, ha="center", fontsize=8, color="white",
                     fontweight="bold", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.15", fc=c, ec="none", alpha=0.9))
    axR.set_title("Por jugador",
                  fontsize=12, fontweight="bold", pad=6)

    br = bal[bal.frame == f]
    for ax in (axL, axR):
        if len(br):
            ax.scatter(br.x_campo, br.y_campo, s=95, color=BALON,
                       edgecolor="black", lw=1.3, zorder=7)

    fig.suptitle(f"Control de espacio (Voronoi) · frame {f} (2v2)  ·  "
                 f"promedio {nfr} fr: B {pctB:.1f}% / A {pctA:.1f}%",
                 fontsize=12.5, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0,0,1,0.95])
    guardar(fig, "voronoi_2v2")

def fig_voronoi_acumulado():
    """Dominancia de espacio ACUMULADA sobre todos los frames 2v2."""
    nx, ny = XMAX, YMAX
    gx = np.linspace(0, XMAX, nx); gy = np.linspace(0, YMAX, ny)
    GX, GY = np.meshgrid(gx, gy)
    cells = np.column_stack([GX.ravel(), GY.ravel()])

    # Acumular: para cada celda, cuántos frames la controló cada equipo
    cuenta_B = np.zeros(nx*ny)
    cuenta_A = np.zeros(nx*ny)
    frames = frames_ambos(rob_eq)
    usados = 0
    for f in frames:
        sub = rob_eq[rob_eq.frame == f]
        P = sub[["x_campo", "y_campo"]].values
        eq = sub["equipo"].values
        if len(P) < 2:
            continue
        d = np.sqrt(((cells[:, None, :] - P[None, :, :])**2).sum(2))
        near = eq[d.argmin(1)]
        cuenta_B += (near == 0)
        cuenta_A += (near == 1)
        usados += 1

    # Fracción de dominancia por celda (-1=todo A, +1=todo B)
    total = cuenta_B + cuenta_A
    total[total == 0] = 1
    domin = (cuenta_B - cuenta_A) / total      # rango [-1, 1]
    domin = domin.reshape(ny, nx)

    fig, ax = plt.subplots(figsize=(5.8, 7.4)); fig.patch.set_facecolor("white")
    dibujar_cancha(ax, campo="#1b4332", linea="white", top=16)

    # Colormap divergente: oscuro (A) -> blanco -> verde (B)
    cmap = mc.LinearSegmentedColormap.from_list(
        "dom", [OSCURO, "#f4f6f4", VERDE])
    ax.imshow(domin, origin="lower", extent=[0, XMAX, 0, YMAX],
              cmap=cmap, vmin=-1, vmax=1, alpha=0.7, zorder=1)

    pctB, pctA, nfr = control_espacio(rob_eq)
    ax.set_title(f"Dominancia de espacio acumulada ({usados} frames 2v2)\n"
                 f"Equipo B {pctB:.0f}%  ·  Equipo A {pctA:.0f}%",
                 fontsize=11.5, fontweight="bold", pad=8)

    # Mini-leyenda de color
    ax.text(0.02, -0.04, "■ zona dominada por A (oscuro)", transform=ax.transAxes,
            fontsize=8, color=OSCURO, va="top")
    ax.text(0.98, -0.04, "zona dominada por B (verde) ■", transform=ax.transAxes,
            fontsize=8, color=VERDE, va="top", ha="right")

    fig.tight_layout()
    guardar(fig, "voronoi_acumulado")

def fig_voronoi_acumulado_jugador():
    """Dominancia de espacio acumulada por JUGADOR (4 robots) sobre frames 2v2."""
    nx, ny = XMAX, YMAX
    gx = np.linspace(0, XMAX, nx); gy = np.linspace(0, YMAX, ny)
    GX, GY = np.meshgrid(gx, gy)
    cells = np.column_stack([GX.ravel(), GY.ravel()])

    # Las 4 identidades de jugador, en orden fijo
    jugadores = [(0,1), (0,2), (1,1), (1,2)]
    color_jug = {(0,1): VERDE, (0,2): VERDE_CLARO, (1,1): OSCURO, (1,2): OSCURO_CLARO}
    etiq_jug  = {(0,1): "B·R1", (0,2): "B·R2", (1,1): "A·R1", (1,2): "A·R2"}

    # Acumular: para cada celda, cuántos frames la controló cada jugador
    cuenta = {j: np.zeros(nx*ny) for j in jugadores}
    usados = 0
    for f in frames_ambos(rob_ju):
        sub = rob_ju[rob_ju.frame == f]
        # necesitamos exactamente los robots presentes con su jugador
        P, ids = [], []
        for _, r in sub.iterrows():
            key = (int(r.equipo), int(r.jugador))
            if key in color_jug:
                P.append([r.x_campo, r.y_campo]); ids.append(key)
        if len(P) < 2:
            continue
        P = np.array(P)
        d = np.sqrt(((cells[:, None, :] - P[None, :, :])**2).sum(2))
        ganador = d.argmin(1)
        for k, key in enumerate(ids):
            cuenta[key] += (ganador == k)
        usados += 1

    # Para cada celda, qué jugador la dominó más
    matriz = np.stack([cuenta[j] for j in jugadores], axis=1)  # (celdas, 4)
    dominante = matriz.argmax(1)
    sin_datos = matriz.sum(1) == 0
    dominante_grid = dominante.reshape(ny, nx)

    fig, ax = plt.subplots(figsize=(6.0, 7.8)); fig.patch.set_facecolor("white")
    dibujar_cancha(ax, campo="#1b4332", linea="white", top=16)

    rgba = np.zeros((ny, nx, 4))
    for k, j in enumerate(jugadores):
        rgba.reshape(-1,4)[(dominante == k) & ~sin_datos] = (*mc.to_rgb(color_jug[j]), 0.6)
    ax.imshow(rgba, origin="lower", extent=[0, XMAX, 0, YMAX], zorder=1)

    # Leyenda
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=color_jug[j], label=etiq_jug[j]) for j in jugadores]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=4, frameon=True, fontsize=9)

    ax.set_title(f"Dominancia de espacio por jugador ({usados} frames 2v2)",
                 fontsize=12.5, fontweight="bold", pad=8)
    fig.tight_layout()
    guardar(fig, "voronoi_acumulado_jugador")

def fig_shot_map():
    goles = detectar_goles(bal); llegadas = detectar_llegadas_area(bal)
    tiros = tiros_a_gol(bal)

    fig = plt.figure(figsize=(7.6, 8.6)); fig.patch.set_facecolor("white")
    # Cancha a la izquierda (ancha), panel de datos a la derecha
    ax  = fig.add_axes([0.02, 0.06, 0.66, 0.86])   # cancha
    axp = fig.add_axes([0.70, 0.06, 0.28, 0.86]); axp.axis("off")  # panel lateral

    dibujar_cancha(ax, campo="#1b4332", linea="white", top=20)

    # Trayectoria muy tenue (contexto). Cambia a False en caso de no querer.
    if True:
        ax.plot(bal.x_campo, bal.y_campo, color="#dde6ec", lw=0.5, alpha=0.4, zorder=2)

    # Flechas de dirección de ataque (sutiles, en los bordes)
    ax.annotate("", xy=(XMAX*0.5, 14), xytext=(XMAX*0.5, 40),
                arrowprops=dict(arrowstyle="-|>", color=OSCURO, lw=2, alpha=0.6), zorder=4)
    ax.text(XMAX*0.5+8, 27, "ataca A", color=OSCURO, fontsize=8, va="center", zorder=4)

    # --- Eventos ---
    for _, r in llegadas.iterrows():
        ax.scatter(r.x, r.y, s=170, marker="^", color=OSCURO,
                   edgecolor="white", lw=1.5, zorder=6)
    for _, t in tiros.iterrows():
        ax.scatter(t.x, t.y, s=260, marker="X", color="#c0392b",
                   edgecolor="white", lw=1.6, zorder=8)
    for _, g in goles.iterrows():
        ax.scatter(g.x, g.y, s=520, marker="*", color="#ffd24a",
                   edgecolor=OSCURO, lw=1.8, zorder=10)
        # halo del gol
        ax.scatter(g.x, g.y, s=1100, marker="*", color=OSCURO,
                   alpha=0.18, zorder=9)

    ax.set_title("Shot map — actividad ofensiva", fontsize=14,
                 fontweight="bold", pad=10)

    # ---------- PANEL LATERAL (estilo card) ----------
    def card(y, titulo, valor, color):
        axp.add_patch(Rectangle((0, y), 1, 0.11, transform=axp.transAxes,
                                fc="#f2f5f3", ec="none"))
        axp.text(0.06, y+0.075, titulo, transform=axp.transAxes,
                 fontsize=8.5, color="#566", va="center")
        axp.text(0.06, y+0.03, valor, transform=axp.transAxes,
                 fontsize=15, color=color, fontweight="bold", va="center")

    n_lleg = len(llegadas); n_tiros = len(tiros); n_goles = len(goles)
    vmax_tiro = tiros.vmax.max() if len(tiros) else 0
    card(0.86, "GOLES (Equipo A)", str(n_goles), OSCURO)
    card(0.72, "Tiros a gol", str(n_tiros), "#c0392b")
    card(0.58, "Llegadas al área", str(n_lleg), "#b5571a")
    if vmax_tiro:
        card(0.44, "Tiro más rápido", f"{vmax_tiro:.1f} cm/fr", "#333")
    card(0.30, "Equipo B", "0 ataques", VERDE)

    # Leyenda de símbolos dentro del panel
    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0], marker="^", color="w", markerfacecolor=OSCURO,
                  markeredgecolor="white", markersize=10, label="Llegada"),
           Line2D([0],[0], marker="X", color="w", markerfacecolor="#c0392b",
                  markeredgecolor="white", markersize=11, label="Tiro a gol"),
           Line2D([0],[0], marker="*", color="w", markerfacecolor="#ffd24a",
                  markeredgecolor=OSCURO, markersize=15, label="Gol")]
    axp.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               frameon=True, fontsize=8.5, title="Símbolos", title_fontsize=9)

    guardar(fig, "shot_map")

def fig_trayectorias():
    """Trayectorias completas de robots (por equipo) y balón sobre la cancha."""
    fig, ax = plt.subplots(figsize=(6.0, 7.8)); fig.patch.set_facecolor("white")
    dibujar_cancha(ax, campo="#1b4332", linea="white", top=16)

    # --- Trayectorias por JUGADOR (4 líneas: B-R1, B-R2, A-R1, A-R2) ---
    estilo_jug = {
        (0, 1): ("#52e88a",       "Equipo B · R1"),
        (0, 2): ("#a0f0c0", "Equipo B · R2"),
        (1, 1): (OSCURO,      "Equipo A · R1"),
        (1, 2): (OSCURO_CLARO,"Equipo A · R2"),
    }
    for (eq, jug), (color, _) in estilo_jug.items():
        t = rob_ju[(rob_ju.equipo == eq) & (rob_ju.jugador == jug)].sort_values("frame")
        if len(t) < 2:
            continue
        ax.plot(t.x_campo, t.y_campo, color=color, lw=1.5, alpha=0.7, zorder=3)

    # --- Trayectoria del balón encima, más marcada ---
    b = bal.sort_values("frame")
    ax.plot(b.x_campo, b.y_campo, color=BALON, lw=1.8, alpha=0.9, zorder=5)
    # marcar inicio y fin del balón
    if len(b):
        ax.scatter(b.x_campo.iloc[0], b.y_campo.iloc[0], s=70, color="white",
                   edgecolor="black", lw=1.3, zorder=6, label="inicio balón")
        ax.scatter(b.x_campo.iloc[-1], b.y_campo.iloc[-1], s=90, color=BALON,
                   edgecolor="black", lw=1.3, zorder=6, marker="s", label="fin balón")

    # --- Leyenda ---
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0],[0], color=VERDE,        lw=2.5, label="B · R1"),
        Line2D([0],[0], color=VERDE_CLARO,  lw=2.5, label="B · R2"),
        Line2D([0],[0], color=OSCURO,       lw=2.5, label="A · R1"),
        Line2D([0],[0], color=OSCURO_CLARO, lw=2.5, label="A · R2"),
        Line2D([0],[0], color=BALON,        lw=2.5, label="Balón"),
    ]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=5, frameon=True, fontsize=8)

    ax.set_title("Trayectorias del partido", fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout()
    guardar(fig, "trayectorias")

def fig_tarjeta():
    pos = posesion_por_frame(rob_ju); tot = len(pos)
    posc = pos[pos.frame.isin(set(frames_ambos(rob_ju)))]; totc = len(posc)
    pB = round(100*(pos.equipo == 0).sum()/tot); pA = 100-pB
    cB = round(100*(posc.equipo == 0).sum()/totc); cA = 100-cB
    vB, vA, nvor = control_espacio(rob_eq); vB = round(vB); vA = 100-vB
    det = 100*len(bal)/TOTAL_FRAMES_VIDEO
    goles = detectar_goles(bal); llegadas = detectar_llegadas_area(bal)
    tiros = tiros_a_gol(bal)
    n_tiros = len(tiros)
    gB = int((goles.equipo == 0).sum()) if len(goles) else 0
    gA = int((goles.equipo == 1).sum()) if len(goles) else 0

    BG, TX, SUB = "#0d3b2e", "#f2f2ee", "#9fc3b0"
    fig = plt.figure(figsize=(9, 5.4)); fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(6, 93, "Resumen del partido", color=TX, fontsize=20, fontweight="bold", va="top")
    ax.text(6, 86.5, "Recorte 2 min · cámara superior · FutBotMX 2026", color=SUB, fontsize=10, va="top")
    ax.text(6, 74, "GOLES", color=SUB, fontsize=11, va="center")
    ax.text(6, 63, str(gB), color=VERDE, fontsize=44, fontweight="bold", va="center")
    ax.text(15, 63, "–", color=TX, fontsize=30, va="center")
    ax.text(22, 63, str(gA), color=OSCURO, fontsize=44, fontweight="bold", va="center")
    ax.text(6, 51, "Equipo B", color=VERDE, fontsize=11, va="center", fontweight="bold")
    ax.text(22, 51, "Equipo A", color=OSCURO, fontsize=11, va="center", fontweight="bold")
    def barra(y, label, b, a):
        ax.text(40, y+5, label, color=SUB, fontsize=9.5, va="center")
        x0, w = 40, 52
        ax.add_patch(Rectangle((x0, y-3), w*b/100, 6, color=VERDE))
        ax.add_patch(Rectangle((x0+w*b/100, y-3), w*a/100, 6, color=OSCURO))
        ax.text(x0+1, y, f"{b}%", color="white", fontsize=9, va="center", fontweight="bold")
        ax.text(x0+w-1, y, f"{a}%", color="white", fontsize=9, va="center", ha="right", fontweight="bold")
    barra(72, f"Posesión total ({tot} fr con balón)", pB, pA)
    barra(58, f"Posesión cara a cara ({totc} fr, ambos)", cB, cA)
    barra(44, f"Control de espacio · Voronoi ({nvor} fr)", vB, vA)
    ax.add_patch(Rectangle((40, 34), 52, 0.3, color=SUB, alpha=0.4))
    stats = [("Llegadas al área", str(len(llegadas)), "izq", TX),
             ("Tiros a gol", str(n_tiros), "izq", TX),
             ("Balón detectado", f"{det:.1f}%", "der", TX),
             ("Más peligroso", "Equipo A", "der", OSCURO)]
    fila = {"izq": 0, "der": 0}
    for label, val, col, c in stats:
        y = 27 - fila[col]*7
        xl, xv = (40, 64) if col == "izq" else (72, 94)
        ha = "center" if col == "izq" else "right"
        ax.text(xl, y, label, color=SUB, fontsize=9.5, va="center")
        ax.text(xv, y, val, color=c, fontsize=13, fontweight="bold", va="center", ha=ha)
        fila[col] += 1
    nota = ("Posesión sobre frames con balón visible, robot más cercano al balón <30 cm. Cara a cara quedó pareja; "
            "Equipo A domina el espacio, hizo el único tiro a gol (~f1877, 9.2 cm/fr)\n"
            "y anotó en la portería azul (f3180). La acción se concentró en la mitad de la "
            "portería azul; Verde no llegó a generar tiro. Goles atribuidos por la dirección de ataque "
            "de cada equipo.")
    ax.text(6, 9, nota, color=SUB, fontsize=7.3, va="center", style="italic")
    guardar(fig, "tarjeta_resumen", facecolor=BG)


if __name__ == "__main__":
    print("Generando figuras en ./%s (DPI=%d, formato=%s)" % (OUT, DPI, FMT))
    fig_mapa_calor_equipos()
    fig_mapa_calor_jugadores()
    fig_voronoi_acumulado()
    fig_posesion()
    fig_voronoi()
    fig_voronoi_acumulado_jugador()
    fig_shot_map()
    fig_trayectorias()
    fig_tarjeta()
    print("Listo.")