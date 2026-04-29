import random
import numpy as np

def iterarPSO(maxIter, t, dim, positions, velocities, p_best_positions, g_best_position):
    '''
    maxIter: Máximo de iteraciones 
    t: iteración actual
    dim: Dimensión de las soluciones
    positions: population actual de soluciones
    velocities: velocidades de las particulas
    p_best_positions: Mejor posicion de la particula
    g_best_position: Mejor particula (best solution) del proceso de optimización
    '''
    wMax = 0.9
    wMin = 0.2
    c1 = 2.0
    c2 = 2.0

    # Update the W of PSO
    w = wMax - t * ((wMax - wMin) / maxIter)
    
    #For de población
    for i in range(len(positions)):
        r1 = np.random.rand(dim)
        r2 = np.random.rand(dim)
        
        cognitive_velocity = c1 * r1 * (p_best_positions[i] - positions[i])
        social_velocity = c2 * r2 * (g_best_position - positions[i])
        
        velocities[i] = w * velocities[i] + cognitive_velocity + social_velocity
        
        positions[i] = positions[i] + velocities[i]

    return positions, velocities


def iterarPSO_Chaotic_r1_r2(maxIter, t, dim, positions, velocities, p_best_positions, g_best_position, chaotic_map):
    '''
    maxIter: Máximo de iteraciones 
    t: iteración actual
    dim: Dimensión de las soluciones
    positions: population actual de soluciones
    velocities: velocidades de las particulas
    p_best_positions: Mejor posicion de la particula
    g_best_position: Mejor particula (best solution) del proceso de optimización
    '''
    
    wMax = 0.9
    wMin = 0.2
    c1 = 2.0
    c2 = 2.0

    # Update the W of PSO
    w = wMax - t * ((wMax - wMin) / maxIter)
    
    rng = np.random.default_rng()
    N = len(chaotic_map)
    
    #For de población
    for i in range(len(positions)):
        idx = rng.integers(0, N, size=dim)
        r1 = np.take(chaotic_map.ravel(), idx)
        idx = rng.integers(0, N, size=dim)
        r2 = np.take(chaotic_map.ravel(), idx)
        
        cognitive_velocity = c1 * r1 * (p_best_positions[i] - positions[i])
        social_velocity = c2 * r2 * (g_best_position - positions[i])
        
        velocities[i] = w * velocities[i] + cognitive_velocity + social_velocity
        
        positions[i] = positions[i] + velocities[i]

    return positions, velocities
    
def iterarPSO_Chaotic_r1(maxIter, t, dim, positions, velocities, p_best_positions, g_best_position, chaotic_map):
    '''
    maxIter: Máximo de iteraciones 
    t: iteración actual
    dim: Dimensión de las soluciones
    positions: population actual de soluciones
    velocities: velocidades de las particulas
    p_best_positions: Mejor posicion de la particula
    g_best_position: Mejor particula (best solution) del proceso de optimización
    '''
    
    wMax = 0.9
    wMin = 0.2
    c1 = 2.0
    c2 = 2.0

    # Update the W of PSO
    w = wMax - t * ((wMax - wMin) / maxIter)
    
    rng = np.random.default_rng()
    N = len(chaotic_map)
    
    #For de población
    for i in range(len(positions)):
        idx = rng.integers(0, N, size=dim)
        r1 = np.take(chaotic_map.ravel(), idx)
        r2 = np.random.rand(dim)
        
        cognitive_velocity = c1 * r1 * (p_best_positions[i] - positions[i])
        social_velocity = c2 * r2 * (g_best_position - positions[i])
        
        velocities[i] = w * velocities[i] + cognitive_velocity + social_velocity
        
        positions[i] = positions[i] + velocities[i]

    return positions, velocities
    
    
def iterarPSO_Chaotic_r2(maxIter, t, dim, positions, velocities, p_best_positions, g_best_position, chaotic_map):
    '''
    maxIter: Máximo de iteraciones 
    t: iteración actual
    dim: Dimensión de las soluciones
    positions: population actual de soluciones
    velocities: velocidades de las particulas
    p_best_positions: Mejor posicion de la particula
    g_best_position: Mejor particula (best solution) del proceso de optimización
    '''
    
    wMax = 0.9
    wMin = 0.2
    c1 = 2.0
    c2 = 2.0

    # Update the W of PSO
    w = wMax - t * ((wMax - wMin) / maxIter)
    
    rng = np.random.default_rng()
    N = len(chaotic_map)
    
    #For de población
    for i in range(len(positions)):
        r1 = np.random.rand(dim)
        idx = rng.integers(0, N, size=dim)
        r2 = np.take(chaotic_map.ravel(), idx)
        
        
        cognitive_velocity = c1 * r1 * (p_best_positions[i] - positions[i])
        social_velocity = c2 * r2 * (g_best_position - positions[i])
        
        velocities[i] = w * velocities[i] + cognitive_velocity + social_velocity
        
        positions[i] = positions[i] + velocities[i]

    return positions, velocities