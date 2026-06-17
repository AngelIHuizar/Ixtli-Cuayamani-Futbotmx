# Licencias de terceros

Este proyecto utiliza modelos y bibliotecas de terceros. A continuación se
atribuye cada uno y se indica su licencia, conforme al requisito § 3.6 de la
convocatoria. Los pesos de los modelos **no se incluyen** en este repositorio;
se descargan desde Hugging Face tras aceptar los términos correspondientes.

## Modelos

| Modelo | Autor | Uso en el proyecto | Licencia |
|---|---|---|---|
| **SAM 3** (`facebook/sam3`) | Meta AI | Segmentación de robots y balón (vocabulario abierto) | SAM License (ver repositorio oficial de SAM 3) |
| **DINOv3** (`facebook/dinov3-convnext-tiny-pretrain-lvd1689m`) | Meta AI | Embeddings visuales para identificación de equipos | Licencia DINOv3 de Meta (gated en Hugging Face) |

- SAM 3 — repositorio: https://github.com/facebookresearch/sam3 · modelo: https://huggingface.co/facebook/sam3
- DINOv3 — modelo: https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m

Ambos modelos requieren aceptar sus términos de uso en Hugging Face antes de
descargarse. Su uso se rige por las licencias propias de Meta.

## Bibliotecas de software

| Biblioteca | Uso | Licencia |
|---|---|---|
| transformers | carga e inferencia de SAM 3 y DINOv3 | Apache 2.0 |
| opencv-python | E/S de video, dibujo, espacio de color HSV | Apache 2.0 |
| supervision | anotadores y estructura de detecciones | MIT |
| trackers (Roboflow) | seguimiento ByteTrack | Apache 2.0 |
| scikit-learn | K-Means para agrupamiento de equipos | BSD-3-Clause |
| pandas | manejo de trayectorias en CSV | BSD-3-Clause |
| numpy | cómputo numérico | BSD-3-Clause |
| scipy | diagramas de Voronoi (control de espacio) | BSD-3-Clause |
| matplotlib | visualizaciones | Licencia estilo BSD (matplotlib) |
| pillow | conversión de imágenes para los modelos | HPND |

Cada dependencia se usa respetando su licencia. El equipo puede explicar el rol
de cada una dentro del pipeline (ver sección 4 del README).


