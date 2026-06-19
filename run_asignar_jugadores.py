# correr_asignar_jugadores.py
from src.asignar_jugadores import asignar_jugadores

df = asignar_jugadores(
    csv_equipos="data/trayectorias_equipos.csv",
    salida_csv="data/trayectorias_jugadores.csv",
)