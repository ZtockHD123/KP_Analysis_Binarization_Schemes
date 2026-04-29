#SCA
import random
import math
import numpy as np


def iterarSCA(maxIter, t, dimension, population, bestSolution):
    # a is a constant number, paper recommend use 2
    a = 2
    # aplicacion de la ecuacion 3.4
    r1 = a - (t * (a / maxIter))
    for i in range(population.__len__()):
        for j in range(dimension):
            rand1 = random.uniform(0.0, 1.0)
            r2 =  (2 * math.pi) * rand1
            rand2 = random.uniform(0.0, 1.0)
            r3 = 2 * rand2
            r4 = random.uniform(0.0, 1.0)
            if r4 < 0.5:
                population[i][j] = population[i][j] + ( ( ( r1 * math.sin(r2)) * abs( ( r3 * bestSolution[j] ) - population[i][j] ) ) )
            else:
                population[i][j] = population[i][j] + ( ( ( r1 * math.cos(r2)) * abs( ( r3 * bestSolution[j] ) - population[i][j] ) ) )
    return np.array(population)


def iterarSCA_Chaotic_r2_r3_r4(maxIter, t, dimension, population, bestSolution, chaotic_map):
    # a is a constant number, paper recommend use 2
    a = 2
    # aplicacion de la ecuacion 3.4
    r1 = a - (t * (a / maxIter))
    
    # mapas caoticos
    rng = np.random.default_rng()
    
    for i in range(population.__len__()):
        for j in range(dimension):
            rand1 = rng.choice(chaotic_map)
            
            r2 =  (2 * math.pi) * rand1
            rand2 = rng.choice(chaotic_map)
            r3 = 2 * rand2
            r4 = rng.choice(chaotic_map)
            if r4 < 0.5:
                population[i][j] = population[i][j] + ( ( ( r1 * math.sin(r2)) * abs( ( r3 * bestSolution[j] ) - population[i][j] ) ) )
            else:
                population[i][j] = population[i][j] + ( ( ( r1 * math.cos(r2)) * abs( ( r3 * bestSolution[j] ) - population[i][j] ) ) )
    return np.array(population)


def iterarSCA_Chaotic_r2(maxIter, t, dimension, population, bestSolution, chaotic_map):
    # a is a constant number, paper recommend use 2
    a = 2
    # aplicacion de la ecuacion 3.4
    r1 = a - (t * (a / maxIter))
    
    # mapas caoticos
    rng = np.random.default_rng()
    
    for i in range(population.__len__()):
        for j in range(dimension):
            rand1 = rng.choice(chaotic_map)
            
            r2 =  (2 * math.pi) * rand1
            rand2 = random.uniform(0.0, 1.0)
            r3 = 2 * rand2
            r4 = random.uniform(0.0, 1.0)
            if r4 < 0.5:
                population[i][j] = population[i][j] + ( ( ( r1 * math.sin(r2)) * abs( ( r3 * bestSolution[j] ) - population[i][j] ) ) )
            else:
                population[i][j] = population[i][j] + ( ( ( r1 * math.cos(r2)) * abs( ( r3 * bestSolution[j] ) - population[i][j] ) ) )
    return np.array(population)
    
    
def iterarSCA_Chaotic_r3(maxIter, t, dimension, population, bestSolution, chaotic_map):
    # a is a constant number, paper recommend use 2
    a = 2
    # aplicacion de la ecuacion 3.4
    r1 = a - (t * (a / maxIter))
    
    # mapas caoticos
    rng = np.random.default_rng()
    
    for i in range(population.__len__()):
        for j in range(dimension):
            rand1 = random.uniform(0.0, 1.0)
            
            r2 =  (2 * math.pi) * rand1
            rand2 = rng.choice(chaotic_map)
            r3 = 2 * rand2
            r4 = random.uniform(0.0, 1.0)
            if r4 < 0.5:
                population[i][j] = population[i][j] + ( ( ( r1 * math.sin(r2)) * abs( ( r3 * bestSolution[j] ) - population[i][j] ) ) )
            else:
                population[i][j] = population[i][j] + ( ( ( r1 * math.cos(r2)) * abs( ( r3 * bestSolution[j] ) - population[i][j] ) ) )
    return np.array(population)
    
def iterarSCA_Chaotic_r4(maxIter, t, dimension, population, bestSolution, chaotic_map):
    # a is a constant number, paper recommend use 2
    a = 2
    # aplicacion de la ecuacion 3.4
    r1 = a - (t * (a / maxIter))
    
    # mapas caoticos
    rng = np.random.default_rng()
    
    for i in range(population.__len__()):
        for j in range(dimension):
            rand1 = random.uniform(0.0, 1.0)
            
            r2 =  (2 * math.pi) * rand1
            rand2 = random.uniform(0.0, 1.0)
            r3 = 2 * rand2
            r4 = rng.choice(chaotic_map)
            if r4 < 0.5:
                population[i][j] = population[i][j] + ( ( ( r1 * math.sin(r2)) * abs( ( r3 * bestSolution[j] ) - population[i][j] ) ) )
            else:
                population[i][j] = population[i][j] + ( ( ( r1 * math.cos(r2)) * abs( ( r3 * bestSolution[j] ) - population[i][j] ) ) )
    return np.array(population)
    
def iterarSCA_Chaotic_r2_r3(maxIter, t, dimension, population, bestSolution, chaotic_map):
    # a is a constant number, paper recommend use 2
    a = 2
    # aplicacion de la ecuacion 3.4
    r1 = a - (t * (a / maxIter))
    
    # mapas caoticos
    rng = np.random.default_rng()
    
    for i in range(population.__len__()):
        for j in range(dimension):
            rand1 = rng.choice(chaotic_map)
            
            r2 =  (2 * math.pi) * rand1
            rand2 = rng.choice(chaotic_map)
            r3 = 2 * rand2
            r4 = random.uniform(0.0, 1.0)
            if r4 < 0.5:
                population[i][j] = population[i][j] + ( ( ( r1 * math.sin(r2)) * abs( ( r3 * bestSolution[j] ) - population[i][j] ) ) )
            else:
                population[i][j] = population[i][j] + ( ( ( r1 * math.cos(r2)) * abs( ( r3 * bestSolution[j] ) - population[i][j] ) ) )
    return np.array(population)
    
    
def iterarSCA_Chaotic_r2_r4(maxIter, t, dimension, population, bestSolution, chaotic_map):
    # a is a constant number, paper recommend use 2
    a = 2
    # aplicacion de la ecuacion 3.4
    r1 = a - (t * (a / maxIter))
    
    # mapas caoticos
    rng = np.random.default_rng()
    
    for i in range(population.__len__()):
        for j in range(dimension):
            rand1 = rng.choice(chaotic_map)
            
            r2 =  (2 * math.pi) * rand1
            rand2 = random.uniform(0.0, 1.0)
            r3 = 2 * rand2
            r4 = rng.choice(chaotic_map)
            if r4 < 0.5:
                population[i][j] = population[i][j] + ( ( ( r1 * math.sin(r2)) * abs( ( r3 * bestSolution[j] ) - population[i][j] ) ) )
            else:
                population[i][j] = population[i][j] + ( ( ( r1 * math.cos(r2)) * abs( ( r3 * bestSolution[j] ) - population[i][j] ) ) )
    return np.array(population)
    
def iterarSCA_Chaotic_r3_r4(maxIter, t, dimension, population, bestSolution, chaotic_map):
    # a is a constant number, paper recommend use 2
    a = 2
    # aplicacion de la ecuacion 3.4
    r1 = a - (t * (a / maxIter))
    
    # mapas caoticos
    rng = np.random.default_rng()
    
    for i in range(population.__len__()):
        for j in range(dimension):
            rand1 = random.uniform(0.0, 1.0)
            
            r2 =  (2 * math.pi) * rand1
            rand2 = rng.choice(chaotic_map)
            r3 = 2 * rand2
            r4 = rng.choice(chaotic_map)
            if r4 < 0.5:
                population[i][j] = population[i][j] + ( ( ( r1 * math.sin(r2)) * abs( ( r3 * bestSolution[j] ) - population[i][j] ) ) )
            else:
                population[i][j] = population[i][j] + ( ( ( r1 * math.cos(r2)) * abs( ( r3 * bestSolution[j] ) - population[i][j] ) ) )
    return np.array(population)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    