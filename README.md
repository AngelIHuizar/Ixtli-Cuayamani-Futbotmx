# Ixtli-Cuayamani · Análisis de Fútbol Robótico con SAM 3

**Copa FutBotMX-Meta 2026 — Capítulo Visión por Computadora — Categoría Profesional**

Pipeline de visión por computadora que segmenta, rastrea y analiza partidos de fútbol robótico a partir de video cenital, usando **SAM 3** como motor de segmentación de vocabulario abierto, **DINOv3** para identificación de equipos y **homografía** para llevar todo a coordenadas reales de la cancha. A partir de ahí genera analítica deportiva: posesión por equipo, mapas de calor, control de espacio, tiros y goles — cada métrica acompañada de su nivel de confianza y sus limitaciones.

**Equipo:** Cristina, Ángel y Miguel.

## TL;DR

Pipeline end-to-end que toma un video cenital de un partido de fútbol robótico y produce:

- **Segmentación** de robots (`"small robot"`) y balón (`"mini orange ball"`) con **SAM 3**, separando robots en colisión y filtrando manos del árbitro.
- **Seguimiento** con ByteTrack (IDs persistentes) e **identificación de equipos no supervisada** con embeddings **DINOv3** + K-Means (sin etiquetas, robusta a cambios de ID).
- **Homografía** de 7 puntos → coordenadas reales de cancha (cm), con error medio de 3.4 cm.
- **Métricas y eventos**: posesión (total y cara a cara), goles y disparos atribuidos por dirección de ataque, control de espacio (Voronoi).
- **Visualizaciones**: mapa de calor por equipo, gráfica de posesión, shot map, control de espacio y tarjeta-resumen del partido.
- **Transparencia**: cada métrica reporta sus limitaciones y los sesgos detectados se documentan abiertamente (secciones 7 y 10).

> Cumple los entregables de la convocatoria: flujo de procesamiento (§ 3.5.1), visualización y narrativa de datos (§ 3.5.2) y documentación (§ 3.5.4). La innovación sobre SAM 3 corresponde a la **integración con otros modelos** (§ 3.7.3).
