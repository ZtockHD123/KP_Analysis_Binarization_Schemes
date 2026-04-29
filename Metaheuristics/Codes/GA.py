import random
import numpy as np
from Util.util import selectionSort


###################################### GA Continuo ######################################

"""Cruzamiento SBX (variables continuas)"""
def sbx_crossover(p1, p2, eta_c=20, p_c=0.9, lb=0):
    # Obtenemos el número de variables (la longitud del vector de decisión).
    n = len(p1)
    # creamos una copia de los padres para no modificar los originales
    c1, c2 = np.copy(p1), np.copy(p2)
    # Se verifica si el cruce debe ocurrir. Un número aleatorio se compara con p_c.
    # Si es menor, el cruce se lleva a cabo. 
    # Con esto, existe la posibilidad de mantener los padres originales.
    if np.random.rand() < p_c:
        #  Iteramos por cada variable
        for i in range(n):
            #  Con una probabilidad del 50%, se decide si se aplica el cruce a la variable actual.
            if np.random.rand() <= 0.5:
                # Se verifica si los valores de los padres son significativamente diferentes
                # para evitar problemas de división por cero o errores de precisión.
                if abs(p1[i] - p2[i]) > 1e-14:
                    # Se ordenan los valores de los padres para el gen actual.
                    x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
                    # Generamos un n° aleatorio que tiene impacto en la forma de la distribución del hijo
                    rand = np.random.rand()
                    # Calcula el factor beta. El 0.0 representa el límite inferior del espacio de búsqueda.
                    beta = 1.0 + (2.0*(x1 - lb)/(x2 - x1))
                    # Calcula el factor alpha. Esta normaliza la distribución de probabilidad.
                    alpha = 2.0 - beta**-(eta_c+1)
                    # Si el número aleatorio está en el primer segmento de la distribución...
                    if rand <= 1.0/alpha:
                        beta_q = (rand*alpha)**(1.0/(eta_c+1))
                    # Si el número aleatorio está en el segundo segmento de la distribución...
                    else:
                        beta_q = (1.0/(2.0 - rand*alpha))**(1.0/(eta_c+1))
                    # Calculo de los nuevos genes para el hijo 1 y 2
                    c1[i] = 0.5*((x1+x2) - beta_q*(x2-x1))
                    c2[i] = 0.5*((x1+x2) + beta_q*(x2-x1))
    # Devuelve los dos hijos
    return c1, c2

"""Mutación polinomial (variables continuas)"""
def polynomial_mutation(x, eta_m=20, p_m=0.1):
    # Creamos una copia de la solución para no modificar la original
    y = np.copy(x)
    # Iteramos por cada gen (variable) del individo (solucion)
    for i in range(len(x)):
        # Comprueba si la variable actual debe ser mutada
        # Un número aleatorio se compara con la probabilidad de mutación (p_m)
        if np.random.rand() < p_m:
            # varaible delta que representa la cantidad del cambio
            delta = 0
            # numero aleatorio que determina la magnitud y dirección del cambio
            r = np.random.rand()
            if r < 0.5:
                # aplicamos alteración negativa
                delta = (2*r)**(1/(eta_m+1)) - 1
            else:
                # aplicamos alteración positiva
                delta = 1 - (2*(1-r))**(1/(eta_m+1))
            y[i] = y[i] + delta
    # Devuelve el individuo mutado
    return y

def seleccion_torneo(poblacion, fitness, k=5):
    """
    Selecciona un padre mediante el método de torneo.
    Se eligen 'k' individuos al azar y el de mejor aptitud (menor valor) gana.
    """
    # Obtener 'k' índices aleatorios de la población
    seleccion_ix = np.random.randint(len(poblacion))
    for ix in np.random.randint(0, len(poblacion), k-1):
        # Comprobar si el nuevo contendiente es mejor
        if fitness[ix] > fitness[seleccion_ix]:
            seleccion_ix = ix
    return poblacion[seleccion_ix]

def iterarGA_Continuo(population, fitness, cross, muta, lb):
    descendencia = []
    tam_poblacion = len(population)
    while len(descendencia) < tam_poblacion:
        # Selección de padres
        padre1 = seleccion_torneo(population, fitness)
        padre2 = seleccion_torneo(population, fitness)
        # Cruce
        hijo1, hijo2 = sbx_crossover(padre1, padre2, eta_c=20, p_c=cross, lb=lb)
        # Mutación
        hijo1 = polynomial_mutation(hijo1, eta_m=20, p_m=muta)
        hijo2 = polynomial_mutation(hijo2, eta_m=20, p_m=muta)
        descendencia.extend([hijo1, hijo2])
    
    # Reemplazar la población antigua por la nueva (descendencia)
    return np.array(descendencia)

###################################### GA Continuo ######################################



###################################### GA Binario ######################################

# selection of parent
def seleccion_ruleta(population, fitness, type_problem, k=3):
    """
    Selecciona un individuo de la población mediante el método de torneo,
    adaptándose a problemas de minimización o maximización.

    Args:
        poblacion (list or np.array): La población de individuos.
        aptitudes (list or np.array): Lista con los valores de aptitud de cada individuo.
        type_problem (str): Tipo de problema. Debe ser "MIN" o "MAX".
        k (int): El número de individuos que participan en el torneo.

    Returns:
        El individuo ganador del torneo.
    """
    # 1. Validar el tipo de problema
    if type_problem not in ["MIN", "MAX"]:
        raise ValueError("El parámetro 'type_problem' debe ser 'MIN' o 'MAX'")

    # 2. Seleccionar 'k' índices aleatorios de la población para el torneo
    num_individuos = len(population)
    indices_torneo = np.random.choice(np.arange(num_individuos), size=k, replace=False)
    
    # 3. Inicializar al ganador como el primer participante
    indice_ganador = indices_torneo[0]
    
    # 4. Realizar el torneo comparando según el tipo de problema
    for i in indices_torneo[1:]:
        # Si es un problema de MAXIMIZACIÓN, buscamos el valor más alto
        if type_problem == "MAX":
            if fitness[i] > fitness[indice_ganador]:
                indice_ganador = i
        # Si es un problema de MINIMIZACIÓN, buscamos el valor más bajo
        elif type_problem == "MIN":
            if fitness[i] < fitness[indice_ganador]:
                indice_ganador = i
                
    # 5. Retornar al individuo ganador
    return population[indice_ganador]

# crossover operator
def cruce_un_punto(p1, p2, r_cruce):
    """Realiza un cruce de un solo punto entre dos padres."""
    if np.random.rand() < r_cruce:
        return p1.copy(), p2.copy()
        
    # Elegir un punto de cruce aleatorio
    punto_cruce = np.random.randint(1, len(p1) - 1)
    # Crear hijos
    h1 = np.concatenate((p1[:punto_cruce], p2[punto_cruce:]))
    h2 = np.concatenate((p2[:punto_cruce], p1[punto_cruce:]))
    return h1, h2

# mutation operator
def mutacion_bit_flip(individuo, r_mutacion):
    """
    Aplica una mutación de 'bit flip'.
    Cada gen (bit) tiene una probabilidad 'r_mutacion' de cambiar su valor.
    """
    for i in range(len(individuo)):
        if np.random.rand() < r_mutacion:
            individuo[i] = 1 - individuo[i] # Cambia 0 a 1 y 1 a 0
    return individuo

def iterarGA(population, fitness, cross, muta, type_problem):
    tam_poblacion = len(population)
    # Crear la siguiente generación
    descendencia = []
    # Elitismo: conservar al mejor individuo de la generación actual
    if type_problem == 'MIN':
        elite_idx = np.argmin(fitness)
    else:
        elite_idx = np.argmax(fitness)
    descendencia.append(population[elite_idx].copy())
    
    while len(descendencia) < tam_poblacion:
        padre1 = seleccion_ruleta(population, fitness, type_problem)
        padre2 = seleccion_ruleta(population, fitness, type_problem)
        hijo1, hijo2 = cruce_un_punto(padre1, padre2, cross)
        hijo1 = mutacion_bit_flip(hijo1, muta)
        hijo2 = mutacion_bit_flip(hijo2, muta)
        descendencia.extend([hijo1, hijo2])
    
    return np.array(descendencia[:tam_poblacion])
        
###################################### GA Binario ######################################