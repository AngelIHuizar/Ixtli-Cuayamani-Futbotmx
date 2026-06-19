# Ixtli-Cuayamani · Análisis de Fútbol Robótico con SAM 3

**Copa FutBotMX-Meta 2026 — Capítulo Visión por Computadora — Categoría Profesional**

Sistema de visión para la segmentacion, rastreo y análisis de partidos de fútbol robótico a partir de video de un partido grabado con vista cenital. Integra **SAM 3** para segmentación de vocabulario abierto, **DINOv3** para identificación de equipos y **homografía** para proyectar los datos a coordenadas reales de la cancha. Como resultado, genera analíticas deportivas como: posesión por equipo, mapas de calor, control de espacio, tiros y goles.

**Equipo:** Cristina Pérez Ramos, Ángel Itzcoatl Huizar Bretado y Miguel Galicia Cuamatzi.

## TL;DR

Pipeline end-to-end que toma un video cenital de un partido de fútbol robótico y produce:

- **Segmentación** de robots (`"small robot"`) y balón (`"mini orange ball"`) con **SAM 3**, separando robots en colisión y filtrando manos del árbitro.
- **Seguimiento** con ByteTrack (IDs persistentes) e **identificación de equipos no supervisada** con embeddings **DINOv3** + K-Means (sin etiquetas, robusta a cambios de ID).
- **Homografía** de 7 puntos → coordenadas reales de cancha (cm), con error medio de 3.4 cm.
- **Métricas y eventos**: posesión (total y cara a cara), goles y disparos atribuidos por dirección de ataque, control de espacio (Voronoi).
- **Visualizaciones**: mapa de calor por equipo, gráfica de posesión, shot map, control de espacio y tarjeta-resumen del partido.
- **Transparencia**: cada métrica reporta sus limitaciones y los sesgos detectados se documentan abiertamente (secciones 7 y 10).

> Cumple los entregables de la convocatoria: flujo de procesamiento (§ 3.5.1), visualización, narrativa de datos (§ 3.5.2) y documentación (§ 3.5.4). La innovación sobre SAM 3 corresponde a la **integración con otros modelos** (§ 3.7.3).

## Arquitectura

```
video (cámara superior cenital · recorte ~2 min)
        │
        ▼
[segmentación]   SAM 3 · prompts "small robot" / "mini orange ball" (umbral 0.35)
        │
        ▼
[tracking]       ByteTrack · IDs persistentes por frame (robots y balón)
        │
        ▼
[homografía]     7 puntos · imagen → cancha en cm (error de reproyección ≈ 3.4 cm)
        │
        ▼
[equipos]        verde dentro de la máscara de SAM → Equipo B (Verde) / Equipo A (Oscuro)
        │
        ▼
[métricas+eventos]  posesión · Voronoi · llegadas · tiros a gol · goles  →  figuras
```

## Identificación de equipos (decisión de diseño)

Los equipos se distinguen por la **presencia de verde dentro de la máscara que SAM
entrega para cada robot**: si la máscara contiene suficiente verde, es Equipo B (Verde);
si no, Equipo A (Oscuro). **Ya no se usa DINOv3 + K-Means**, que en este material daba
asignaciones menos estables. El método por máscara es directo, reproducible y no requiere
entrenamiento ni embeddings.

Un caso se corrigió a mano (el track 14, asignado a Verde) y queda documentado para
transparencia.

## Cancha y datos

- Cancha **243 cm (largo, eje y) × 182 cm (ancho, eje x)**. Boca de portería x ∈ [61, 121].
  Porterías en los extremos y = 0 (**amarilla**) e y = 243 (**azul**). Dos robots por equipo.
  La **azul** es el extremo inferior en el video; las figuras se orientan igual (azul abajo).
- **Equipos:** Equipo B = **Verde** (id 0) · Equipo A = **Oscuro** (id 1).
  - Verde (B): tracks 2, 3, 4, 9, 12, 14, 17.
  - Oscuro (A): tracks 0, 1, 15.
- **Jugadores:** Verde R1 = {2, 4, 9, 12, 14}, R2 = {3, 17} · Oscuro R1 = {1}, R2 = {0, 15}.
- **Archivos** (`data/`): `trayectorias_equipos.csv`, `trayectorias_jugadores.csv`
  (con columna `jugador`) y `balon_final.csv` (1513 frames de balón detectados).

## Resultados

| Métrica | Equipo B (Verde) | Equipo A (Oscuro) |
|---|---|---|
| Posesión total (454 fr con balón) | 30 % | 70 % |
| Posesión cara a cara (265 fr, ambos en cancha) | 51 % | 49 % |
| Control de espacio · Voronoi (1791 fr) | 38.7 % | 61.3 % |
| Posesión por jugador (R1 / R2) | 0 % / 30 % | 27 % / 43 % |
| Llegadas al área | 0 | 2 |
| Tiros a gol | 0 | 1 |
| **Goles** | 0 | **1** |

Detección de balón: **41.9 %** (1513 / 3607 frames del recorte).

**Lectura del partido.** Con los datos reprocesados, la posesión **cara a cara quedó
pareja (51 / 49)**: no es cierto que un equipo "tuviera el balón". Lo que separa a los
equipos es el espacio y el peligro. **El Equipo A (Oscuro) controló más cancha (≈61 %)**,
fue el **único en generar peligro** (logrando 2 llegadas y el único tiro a gol, ~f1876
a 9.2 cm/frame) y **anotó el único gol** (frame 3180, portería azul, balón 242 frames en
la red). El gol entró rodando lento, por eso cuenta como gol pero no como tiro. El Equipo
B no registró llegadas ni tiros.

## Figuras

| Figura | Archivo |
|---|---|
| Mapa de calor por equipo | `outputs/mapa_calor_equipos.png` |
| Mapa de calor por jugador (2×2) | `outputs/mapa_calor_jugadores.png` |
| Posesión (total + cara a cara + por jugador) | `outputs/posesion.png` |
| Control de espacio (Voronoi, snapshot 2v2) | `outputs/voronoi_2v2.png` |
| Shot map (tiros, llegadas y gol) | `outputs/shot_map.png` |
| Tarjeta-resumen | `outputs/tarjeta_resumen.png` |

## Reproducir

```bash
# Dependencias
pip install -r requirements.txt          # numpy, pandas, matplotlib, scipy, ...

# Prueba de humo: valida datos, mapeo de equipos y eventos (sin video ni SAM)
python smoke_test.py

# Generar las 6 figuras (lee data/, escribe outputs/)
python generar_figuras.py
```

Parámetros de salida (DPI, formato PNG/PDF/SVG) en el bloque de configuración al inicio
de `generar_figuras.py`. `smoke_test.py` confirma que los CSV cargan, que el mapeo
track→equipo es correcto (Oscuro = {0,1,15}) y que la detección da 1 gol (Equipo A) y
2 llegadas.

Los scripts del pipeline completo (segmentación, tracking, homografía, equipos) están en
`scripts/`, con los módulos reutilizables en `src/`.

## Estructura del repositorio

```
.
├── src/            módulos reutilizables (segmentación, tracking, homografía, equipos, eventos)
├── scripts/        runners del pipeline y de figuras
├── experiments/    variantes y diagnósticos (método DINO previo, pruebas, verificaciones)
├── data/           CSV procesados (gitignored salvo entregables)
├── dataset/        frames y calibración crudos
├── outputs/        figuras generadas
├── docs/           THIRD_PARTY_LICENSES, notas de cumplimiento
├── generar_figuras.py
├── smoke_test.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Definiciones (para reproducibilidad)

- **Posesión:** en cada frame con balón visible, posee el robot **más cercano al balón
  dentro de 30 cm desde el centro del robot**. La total se calcula sobre los frames con balón; la "cara a cara"
  solo sobre los frames con **ambos** equipos presentes.
- **Control de espacio (Voronoi):** cada celda de la cancha se asigna al equipo del robot
  más cercano; se promedia sobre los frames con ambos equipos.
- **Llegada al área:** el balón entra al área de penalti (25 cm frente a la portería,
  dentro del ancho del arco).
- **Tiro a gol:** episodio en que el balón se desplaza rápido (≥ 2.5 cm/frame, pico ≥ 3.5)
  dirigido a la portería azul, por delante de la línea y encarando la boca del arco.
- **Gol:** el balón cruza la línea dentro del ancho del arco y **se sostiene ≥ 30 frames**
  detrás de ella (rechaza cruces breves).

## Limitaciones y caveats

- **Detección de balón 41.9 %.** El balón se ve en 1513 de 3607 frames; los huecos se
  ignoran/interpolan según la métrica. Las líneas largas y rectas del shot map son tramos
  entre detecciones separadas, no movimiento real.
- **Sesgo de detección hacia el Verde** y **recorte con fase sin juego**: pueden afectar
  las cuentas absolutas.
- **Verde R1 = 0 % de posesión** lleva caveat: ese "jugador" agrupa **5 tracks
  fragmentados** ({2,4,9,12,14}); su 0 % puede deberse a *ID switches*, no necesariamente
  a inactividad real.
- **Falso positivo de gol (frame 416):** el balón cruza ~13 frames y vuelve a jugarse; se
  **rechaza** con el criterio de sostenimiento ≥ 30 frames. El gol real (3180, 242 frames)
  se conserva.
- **Tiros a gol** depende del umbral de velocidad; con el criterio reportado da 1 (el
  remate de f1876 aparece con cualquier umbral razonable).
- **Geometría (confirmada con el video):** la azul está en y=243 (parte inferior del
  video) y la amarilla en y=0. El gol y las llegadas ocurren en la azul, atacada y anotada
  por el Equipo A (Oscuro).

## Atribución de dependencias

| Dependencia | Rol | Licencia |
|---|---|---|
| SAM 3 (Meta) | Segmentación base de robots y balón | Meta SAM License (**no MIT**) |
| ByteTrack / supervision | Seguimiento multi-objeto | MIT |
| OpenCV | IO de video y homografía | Apache 2.0 |
| NumPy · pandas · SciPy | Cómputo y procesamiento de datos | BSD |
| Matplotlib | Figuras (mapas de calor, Voronoi, shot map) | Matplotlib (estilo BSD) |

Los pesos de SAM 3 **no se incluyen** en el repo; se descargan según la licencia de Meta.
Texto completo de licencias de terceros en `docs/THIRD_PARTY_LICENSES.md`.

## Licencia

Código bajo licencia **MIT** (ver `LICENSE`). El uso de SAM 3 se rige por la licencia de
Meta.

## Equipo y créditos

**Ixtli-Cuayamani** — Categoría Profesional, INAOE:

| Integrante | Rol |
|---|---|
| Cristina Pérez Ramos | Integrante |
| Ángel Itzcoatl Huizar Bretado | Integrante |
| Miguel Galicia Cuamatzi | Integrante |

Asistencia con LLM (autorizada en la convocatoria): se usó Claude (Anthropic) para apoyo
en código, depuración y documentación. El diseño técnico, las decisiones y la validación
son responsabilidad y autoría del equipo.

## Calendario

- **2026-06-19**: deadline del entregable de GitHub (repositorio congelado).
- **2026-06-24 → 26**: Copa FutBotMX presencial (UPIITA-IPN, CDMX).
## Enlace a videos:
https://youtu.be/m0kYT9DCMss
Reel de Instagram : https://www.instagram.com/reel/DZxz9HHRq8IVkyeGNDb-WMts3YTCMizwBDe8vg0/?igsh=MTl2cHdseGFodTBkOA==
