# Ixtli-Cuayamani · Análisis de Fútbol Robótico con SAM 3

**Copa FutBotMX-Meta 2026 — Capítulo Visión por Computadora — Categoría Profesional**
**Equipo:** Cristina Pérez Ramos, Ángel Itzcoatl Huizar Bretado y Miguel Galicia Cuamatzi.

Sistema de visión para la segmentacion, rastreo y análisis de partidos de fútbol robótico a partir de video de un partido grabado con vista cenital. Integra **SAM 3** para segmentación de vocabulario abierto, **DINOv3** para identificación de equipos y **homografía** para proyectar los datos a coordenadas reales de la cancha. Como resultado, genera analíticas deportivas como: posesión por equipo, mapas de calor, control de espacio, tiros y goles.

🎥 **Video demo:** https://www.youtube.com/watch?v=m0kYT9DCMss

📱 **Reel:** https://www.instagram.com/reel/DZxz9HHRq8IVkyeGNDb-WMts3YTCMizwBDe8vg0/

📊 **Dashboard:** abrir `dashboard.html` en el navegador

## TL;DR

A partir de un recorte de 2 minutos de un partido, el sistema detecta y rastrea los robots y el balón, identifica el equipo de cada robot sin etiquetas, y produce métricas cuantitativas. 

- **Segmentación** de robots (`"small robot"`) y balón (`"mini orange ball"`) con **SAM 3**, separando robots en colisión y filtrando manos del árbitro o integrantes del equipo.
- **Seguimiento** con ByteTrack (IDs persistentes) e **identificación de equipos no supervisada** con embeddings **DINOv3** + K-Means (robusta a cambios de ID).
- **Homografía** de 7 puntos → coordenadas reales de cancha (cm), con error medio de 3.4 cm.
- **Métricas y eventos**: posesión (total y cara a cara), goles y disparos atribuidos por dirección de ataque, control de espacio (Voronoi).
- **Visualizaciones**: mapa de calor por equipo, gráfica de posesión, shot map, control de espacio y tarjeta-resumen del partido.

Resultado del partido analizado: 
**1 gol del Equipo A**, que dominó el territorio y la actividad ofensiva, mientras el **Equipo B** fue más parejo en la disputa directa del balón.
 
> *"El equipo B, aunque disputó el balón de forma pareja, el equipo A ganó territorio, tiro a gol y el gol."*.

## 1. El reto y el dataset
 
Ixtli Cuayamani participa en la **categoría profesional**.
 
**Cancha (Reglamento Copa FutBotMX 2026):** 243 cm de largo × 182 cm de ancho.
Portería **amarilla** en un extremo (y = 0) y **azul** en el otro (y = 243), de
60 cm de ancho. Áreas de penalti de 25 cm × 80 cm. El reglamento prohíbe los
colores naranja/amarillo/azul en los robots, por lo que **el balón naranja es el
único objeto naranja del campo**.
 
**Dataset:** video cenital de cámara fija, tomado del dataset del concurso. El análisis se hace sobre un **recorte
representativo de ~2 minutos (3607 frames a 30 fps)**, recorte del video IMG_9933.mov (videos/18abril/camara_superior - del dataset)

## 2. Arquitectura

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
## 3. Metodología por etapa

### 3.1 Segmentación con SAM 3
SAM 3 se usa por su ruta nativa en `transformers` (`Sam3Processor` / `Sam3Model`). Los robots se segmentan con el prompt de texto `"small robot"` (umbral 0.25), que superó al prompt `"robot"`. SAM 3 separa robots en colisión en lugar de fusionarlos. El balón se detecta con `"mini orange ball"` (umbral 0.35), que mejora la confianza de detección frente a `"orange ball"` (~0.4 → ~0.7 en frames con balón visible). Las manos de los participantes se filtran restando el prompt `"hand"` por solapamiento (IoU) y por la persistencia temporal del seguimiento.

### 3.2 Seguimiento (tracking)
Se asigna a cada robot un identificador persistente con **ByteTrack** (paquete `trackers`). La posición de cada robot es el centroide de su máscara, que en vista cenital, coincide con su punto de contacto siendo es estable ante rotaciones y colisiones. 

### 3.3 Homografía
La calibración se realiza con **7 puntos** (esquinas visibles + postes de portería, ya que la esquina superior-derecha no
aparece en cuadro) mediante `cv2.findHomography`, repartiendo el error de reproyección (promedio: 3.4 cm) sobre una cancha de 243 × 182 cm — menos de un tercio del tamaño de un robot.

### 3.4 Identificación de equipos (DINOv3) + mascara SAM3

Para asignar cada robot a su equipo, se recortan vistas del robot a lo largo del video, se extrae un embedding visual con **DINOv3** (`convnext-tiny`) de cada una, se promedian por `tracker_id` y se agrupan en dos conjuntos con **K-Means** (k = 2). Además, los equipos se distinguen por la presencia de su color dentro de la máscara que SAM 3 entrega para cada robot. 

### 3.5 Métricas y eventos

Sobre las trayectorias en coordenadas de cancha se calculan:

**Dirección de ataque.** Antes de atribuir eventos, determinamos qué portería ataca cada equipo midiendo, en los frames de posesión, hacia dónde tiende a moverse el balón. El equipo B empuja el balón hacia la portería amarilla (Δy medio −2.8 cm) y el equipo A hacia la azul (+2.7 cm).

**Posesión por equipo (dos lecturas).** En cada frame, el robot más cercano al balón (dentro de 30 cm) "posee"; se suma el tiempo por equipo. Reportamos dos cifras:

- *Posesión total*: sobre todos los frames con balón visible. Incluye los tramos en que a un equipo le retiraron robots, por lo que puede estar sesgada hacia quien jugó con superioridad numérica.
- *Posesión cara a cara*: solo sobre frames con ambos equipos presentes en cancha. Mide quién gana el balón en igualdad de condiciones.

**Control de espacio (Voronoi).** Se divide la cancha en celdas; cada celda pertenece al robot más cercano y suma para su equipo. El porcentaje de campo controlado por cada equipo mide el dominio territorial. Se calcula solo sobre frames con ambos equipos presentes.

**Llegadas al área y tiros.** Distinguimos dos conceptos: una *llegada al área* (el balón entra a los 25 cm frente a una portería) y un *tiro a gol* (el balón acelera por encima de un umbral dirigido a una portería, marcando el punto de origen). El tiro a gol es la base del *shot map*.

**Goles.** El balón cruza la línea de gol **dentro del ancho de portería** (61–121 cm en x), con consolidación temporal de detecciones cercanas. La atribución de equipo usa la dirección de ataque derivada de los datos. El filtro de ancho descarta correctamente los cruces de línea por los costados (balón fuera de banda).

**Mapas de calor por equipo.** Histograma 2D de posiciones, suavizado, normalizado por equipo (cada panel a su propio máximo, para comparar *patrones* de juego y no tiempo en cancha, dado que un equipo juega menos por las infracciones). Se excluyen los robots retirados/parados en la orilla mediante un umbral de radio de giro. También agregamos el mapa de calor por jugador. 

## 4. Innovación sobre SAM 3 (requisito Profesional § 3.7.3)

- **Integración con otros modelos:** SAM 3 (segmentación abierta) + clasificación de equipos no supervisada (DINOv3 embeddings visuales para clasificación + color-en-máscara con SAM + K-Means ) + homografía geométrica.
- **Balón con SAM 3:** la detección por concepto evita la confusión con piel/manos que sí afecta a métodos por color (HSV).
- **Identificación de equipos no supervisada:** por agrupamiento de embeddings — robusta a los cambios de ID del tracker.
- **Post-procesamiento geométrico:** control de espacio (pitch control) vía diagramas de Voronoi, y dirección de ataque **derivada de los datos** (movimiento del balón en posesión).
- **Validación cuantitativa y transparencia (§ 3.7.2):** error de homografía medido, verificación visual de equipos, consolidación y validación de eventos, y documentación explícita de los sesgos detectados (secciones 7 y 10).

## 5. Resultados (recorte analizado)
 
| Métrica | Resultado |
|---|---|
| Error de homografía | 3.4 cm promedio |
| Balón detectado | **41.9 %** (1513 / 3607 frames) |
| Posesión total (frames con balón, 454 fr) | **B 30 % — A 70 %** |
| Posesión cara a cara (ambos en cancha, 265 fr) | **B 51 % — A 49 %** |
| Control de espacio (Voronoi) | **B 38.7 % — A 61.3 %** |
| Tiros a gol | 1 (Equipo A) |
| Llegadas al área | 2 (Equipo A) |
| **Goles** | **1 — Equipo A (portería azul, frame 3180)** |

**Dos lecturas de la posesión, a propósito:** la *total* (70/30) se calcula sobre
todos los frames con balón; la *cara a cara* (51/49) solo sobre los frames con ambos
equipos en cancha. El A acumula más posesión total, pero en disputa directa el reparto es parejo.

## 6. Visualizaciones

Generadas con `generar_figuras.py` (en `outputs/`):
 
- `mapa_calor_equipos.png` — ocupación por equipo.
- `mapa_calor_jugadores.png` — ocupación por jugador (R1/R2).
- `posesion.png` — posesión total, cara a cara y por jugador.
- `voronoi_2v2.png` — control de espacio (por equipo y por jugador).
- `voronoi_acumulado.png` — dominancia de espacio sobre todo el partido.
- `shot_map.png` — tiros, llegadas y gol.
- `trayectorias.png` — recorrido de robots y balón.
- `tarjeta_resumen.png` — resumen del partido.

## 7. Reproducibilidad
### Requisitos
Python 3.11, GPU con CUDA. `torch ≥ 2.7`, `transformers`, `supervision`, `trackers`,
`scikit-learn`, `opencv-python`, `pandas`, `scipy`, `matplotlib`. Acceso a los
modelos gated `facebook/sam3` y `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`
en Hugging Face.
 
```bash
conda create -n futbotmx python=3.11 && conda activate futbotmx
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
 
### Orden de ejecución (desde la raíz)
```bash
python calibrar_homografia_multi.py   # -> data/homografia.npy
python run_tracking.py                # -> trayectorias_final.csv, balon_final.csv, video de mascaras
python limpiar_datos.py               # -> trayectorias_limpio.csv
python run_team_id_sd.py              # -> trayectorias_equipos.csv (identificacion de equipos)
python run_asignar_jugadores.py       # -> trayectorias_jugadores.csv (2 jugadores por equipo)
python generar_figuras.py             # -> figuras en outputs/
python generar_video_demo.py          # -> video demo (modo sam: mascaras; modo csv: rapido)
```

## Estructura del repositorio

```
.
├── src/                     segmentation, homography, tracking,
│                            events, cancha, equipos_robusto, asignar_jugadores...)
├── tests/                   variantes y diagnósticos
├── data/                    CSV de trayectorias, balon, homografia
├── dataset/        
├── outputs/                 visualizaciones y dashboard
├── docs/                    THIRD_PARTY_LICENSES.md y documentacion
├── calibrar_homografia_multi.py
├── run_tracking.py
├── limpiar_datos.py
├── run_team_id_sd.py        (identificacion de equipos por color-en-mascara)
├── run_asignar_jugadores.py
├── generar_figuras.py
├── generar_video_demo.py
├── requirements.txt
├── LICENSE                  (MIT)
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

Este proyecto se distribuye bajo licencia **MIT** (ver `LICENSE`). Las licencias de
los modelos y bibliotecas de terceros se detallan en
[`docs/THIRD_PARTY_LICENSES.md`](docs/THIRD_PARTY_LICENSES.md).

## Equipo y créditos

Equipo **Ixtli-Cuayamani** — Cristina, Ángel y Miguel.
Modelos: SAM 3 y DINOv3 (Meta AI). Dimensiones de cancha según el Reglamento oficial
de la Copa FutBotMX 2026.

Asistencia con LLM: se usó Claude (Anthropic) para apoyo en depuración y documentación, Gemini para generar la imagen del logo del equipo que se incluye en los videos. El diseño técnico, las decisiones y la validación
son responsabilidad y autoría del equipo.

## Calendario

- **2026-06-19**: deadline del entregable de GitHub (repositorio congelado).
- **2026-06-24 → 26**: Copa FutBotMX presencial (UPIITA-IPN, CDMX).
## Enlace a videos:
https://youtu.be/m0kYT9DCMss
Reel de Instagram : https://www.instagram.com/reel/DZxz9HHRq8IVkyeGNDb-WMts3YTCMizwBDe8vg0/?igsh=MTl2cHdseGFodTBkOA==
