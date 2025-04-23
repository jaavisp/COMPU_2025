# Importamos las librerías necesarias.
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

# Definimos la clase CuerpoCeleste, que modela un cuerpo celestial en el espacio.
@dataclass
class CuerpoCeleste:
    # Atributos de la clase:
    nombre: str         # Nombre del cuerpo celestial (ej. Sol, Tierra)
    masa: float         # Masa del cuerpo celestial
    posicion: np.ndarray  # Posición en el espacio en 2D [x, y]
    velocidad: np.ndarray # Velocidad en el espacio en 2D [vx, vy]
    trayectoria: List[np.ndarray] = None  # Lista para almacenar la trayectoria del cuerpo celestial

    def __post_init__(self):
        # Si no se define una trayectoria al crear el objeto, la inicializamos como una lista vacía.
        self.trayectoria = []

# Esta función calcula la fuerza gravitacional entre dos cuerpos.
@njit(fastmath=True, cache=True)
def calcular_fuerza(pos1: np.ndarray, pos2: np.ndarray, m1: float, m2: float) -> Tuple[float, float]:
    # Definimos la constante gravitacional.
    G = 6.67430e-11
    r = pos2 - pos1  # Vector de la distancia entre los dos cuerpos.
    norm_r = np.sqrt(np.sum(r * r))  # Calculamos la magnitud de esa distancia.
    
    if norm_r < 1e-10:  # Si la distancia es extremadamente pequeña, evitamos una división por cero.
        return 0.0, 0.0
    
    # Calculamos la magnitud de la fuerza gravitacional usando la ley de gravitación universal.
    fuerza_magnitud = G * m1 * m2 / (norm_r * norm_r)
    
    # Devolvemos la fuerza en la dirección x e y, normalizada por la distancia.
    return fuerza_magnitud * r[0] / norm_r, fuerza_magnitud * r[1] / norm_r

# Esta función actualiza las posiciones y velocidades de los cuerpos usando la ley de Newton.
@njit(fastmath=True, cache=True, parallel=True)
def actualizar_posiciones_numba(posiciones, velocidades, masas, dt):
    n = posiciones.shape[0]  # Número de cuerpos.
    aceleraciones = np.zeros((n, 2))  # Inicializamos un array para almacenar las aceleraciones.
    
    # Calculamos las aceleraciones debido a la fuerza gravitacional de los demás cuerpos.
    for i in prange(n):
        for j in range(n):
            if i != j:
                fx, fy = calcular_fuerza(posiciones[i], posiciones[j], masas[i], masas[j])
                aceleraciones[i, 0] += fx / masas[i]  # Aceleración en la dirección x.
                aceleraciones[i, 1] += fy / masas[i]  # Aceleración en la dirección y.

    # Actualizamos las posiciones y velocidades de los cuerpos usando el método de Euler.
    for i in prange(n):
        posiciones[i] += velocidades[i] * dt + 0.5 * aceleraciones[i] * dt * dt  # Posición.
        velocidades[i] += 0.5 * aceleraciones[i] * dt  # Velocidad.

# Clase principal que maneja la simulación de los cuerpos celestes.
class SimulacionNBody:
    def __init__(self, dt: float, tiempo_total: float):
        # Inicializamos la simulación con un intervalo de tiempo dt y un tiempo total tiempo_total.
        self.cuerpos: List[CuerpoCeleste] = []  # Lista de cuerpos celestes que participan en la simulación.
        self.dt = dt  # El paso de tiempo para la simulación.
        self.tiempo_total = tiempo_total  # Tiempo total que durará la simulación.

    def agregar_cuerpo(self, cuerpo: CuerpoCeleste):
        # Agregamos un cuerpo celeste a la simulación.
        self.cuerpos.append(cuerpo)

    def ejecutar_simulacion(self):
        # Calculamos el número de pasos de la simulación en función del tiempo total y el paso de tiempo.
        pasos = int(self.tiempo_total / self.dt)
        n = len(self.cuerpos)  # Número de cuerpos en la simulación.
        
        # Inicializamos arrays para las posiciones, velocidades y masas de los cuerpos.
        posiciones = np.array([c.posicion.copy() for c in self.cuerpos])
        velocidades = np.array([c.velocidad.copy() for c in self.cuerpos])
        masas = np.array([c.masa for c in self.cuerpos])
        
        # Ejecutamos la simulación en cada paso de tiempo.
        for paso in range(pasos):
            # Actualizamos las posiciones y velocidades usando la función optimizada.
            actualizar_posiciones_numba(posiciones, velocidades, masas, self.dt)
            
            # Cada ciertos pasos, guardamos la trayectoria de cada cuerpo.
            if paso % 10 == 0:
                for i in range(n):
                    self.cuerpos[i].trayectoria.append(posiciones[i].copy())
            
            # Actualizamos las posiciones y velocidades de los cuerpos.
            for i in range(n):
                self.cuerpos[i].posicion = posiciones[i]
                self.cuerpos[i].velocidad = velocidades[i]

    def visualizar_orbitas(self, filename: str = 'orbitas.png'):
        # Esta función se encarga de visualizar las órbitas de los cuerpos celestes y guardar la imagen.
        plt.figure(figsize=(12, 12))  # Creamos una figura de gran tamaño para la visualización.
        
        # Dibujamos la trayectoria de cada cuerpo celestial.
        for cuerpo in self.cuerpos:
            trayectoria = np.array(cuerpo.trayectoria)
            if len(trayectoria) > 0:  # Si hay trayectoria registrada, la graficamos.
                plt.plot(trayectoria[:, 0], trayectoria[:, 1], label=cuerpo.nombre, linewidth=1)
            
            # También dibujamos la posición actual de cada cuerpo con un punto.
            plt.plot(cuerpo.posicion[0], cuerpo.posicion[1], 'o')  # Posición actual del cuerpo.
        
        # Configuramos la visualización.
        plt.axis('equal')  # Aseguramos que las escalas de los ejes X e Y sean iguales.
        plt.grid(True)  # Activamos la cuadrícula para mejor referencia visual.
        plt.legend()  # Mostramos la leyenda con los nombres de los cuerpos.
        plt.title('Órbitas del Sistema Solar con 4 Cuerpos')  # Título de la gráfica.
        plt.xlabel('Posición X (m)')  # Etiqueta del eje X.
        plt.ylabel('Posición Y (m)')  # Etiqueta del eje Y.
        plt.savefig(filename)  # Guardamos la imagen como archivo en el path especificado.
        plt.close()  # Cerramos la figura para liberar recursos.

# Función para crear un sistema solar con 4 cuerpos celestes (Sol, Mercurio, Venus, Tierra, Marte).
def crear_sistema_4cuerpos(dt: float, tiempo_total: float) -> SimulacionNBody:
    sim = SimulacionNBody(dt, tiempo_total)  # Creamos la simulación con los parámetros dados.
    AU = 1.496e11  # Unidad astronómica (distancia media de la Tierra al Sol) en metros.
    
    # Definimos los cuerpos celestes que vamos a incluir en la simulación.
    cuerpos = [
        ("Sol", 1.989e30, np.array([0.0, 0.0]), np.array([0.0, 0.0])),  # El Sol en el centro.
        ("Mercurio", 3.301e23, np.array([0.387 * AU, 0.0]), np.array([0.0, 47.872e3])),  # Mercurio.
        ("Venus", 4.867e24, np.array([0.723 * AU, 0.0]), np.array([0.0, 35.024e3])),  # Venus.
        ("Tierra", 5.972e24, np.array([AU, 0.0]), np.array([0.0, 29.783e3])),  # Tierra.
        ("Marte", 6.417e23, np.array([1.524 * AU, 0.0]), np.array([0.0, 24.071e3]))  # Marte.
    ]
    
    # Agregamos cada cuerpo celeste a la simulación.
    for nombre, masa, posicion, velocidad in cuerpos:
        sim.agregar_cuerpo(CuerpoCeleste(nombre, masa, posicion, velocidad))
    
    # Retornamos la simulación con los cuerpos ya agregados.
    return sim

# Ejecutamos la simulación con un paso de tiempo de un día (24*3600 segundos) y una duración de 5 años.
sim = crear_sistema_4cuerpos(dt=24*3600, tiempo_total=5 * 365.25 * 24 * 3600)
sim.ejecutar_simulacion()

# Guardamos la visualización de las órbitas en el escritorio del usuario.
sim.visualizar_orbitas('C:/Users/alefu/Desktop/orbitas_del_sistema_solar.png') 