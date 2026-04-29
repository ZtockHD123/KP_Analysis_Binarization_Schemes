import numpy as np
import os
import time

from Problem.KP.problem import KP
from Metaheuristics.imports import iterarGWO,iterarSCA,iterarWOA,iterarPSA, iterarGA_Continuo
from Metaheuristics.imports import iterarFOX,iterarEOO,iterarRSA,iterarGOA,iterarHBA,iterarTDO,iterarSHO
from Metaheuristics.imports import iterarSCA_Chaotic_r2_r3_r4,iterarSCA_Chaotic_r2,iterarSCA_Chaotic_r3,iterarSCA_Chaotic_r4,iterarSCA_Chaotic_r2_r3,iterarSCA_Chaotic_r2_r4,iterarSCA_Chaotic_r3_r4
from Metaheuristics.imports import iterarPSO_Chaotic_r1_r2, iterarPSO_Chaotic_r1, iterarPSO_Chaotic_r2
from Diversity.imports import diversidadHussain,porcentajesXLPXPT
from Discretization import discretization as b
from Problem.Benchmark.Problem import fitness as f
from Util.util import convert_into_binary
from Util.log import initial_log, log_progress, final_log
from BD.sqlite import BD
from ChaoticMaps.chaoticMaps import logisticMap,piecewiseMap,sineMap,singerMap,sinusoidalMap,tentMap,circleMap

def solverBEN_ChaoticMaps(id, mh, maxIter, pop, function, lb, ub, dim, param, chaotic):
    dirResult = './Resultados/Transitorio/'
    
    os.makedirs(dirResult, exist_ok = True)
    
    # Inicializamos el archivo de resultados
    results = open(dirResult + f"{mh}_{function}_{id}.csv", "w")
    results.write('iter,fitness,time,XPL,XPT,DIV\n')
    
    bd = BD()
    
    chaotic_map = None
    
    if chaotic == 'LOG':
        quantityElements = maxIter * pop * dim
        chaotic_map = logisticMap(0.7,quantityElements)
    if chaotic == 'PIECE':
        quantityElements = maxIter * pop * dim
        chaotic_map = piecewiseMap(0.7,quantityElements)
    if chaotic == 'SINE':
        quantityElements = maxIter * pop * dim
        chaotic_map = sineMap(0.7,quantityElements)
    if chaotic == 'SINGER':
        quantityElements = maxIter * pop * dim
        chaotic_map = singerMap(0.7,quantityElements)
    if chaotic == 'SINU':
        quantityElements = maxIter * pop * dim
        chaotic_map = sinusoidalMap(0.7,quantityElements)
    if chaotic == 'TENT':
        quantityElements = maxIter * pop * dim
        chaotic_map = tentMap(0.6,quantityElements)
    if chaotic == 'CIRCLE':
        quantityElements = maxIter * pop * dim
        chaotic_map = circleMap(0.7,quantityElements)
    
    copy_lb = lb
    
    if not isinstance(lb, list):
        lb = [lb] * dim
        
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    initialTime = time.time()
    optimo = bd.obtenerOptimoInstancia(function)[0][0]
    
    if function == 'F8':
        optimo = optimo * dim
    
    initializationTime1 = time.time()
    # Generacion de poblacion aleatoria
    # Vectorización para generar población inicial
    lb = np.array(lb)
    ub = np.array(ub)
    population = np.random.uniform(0, 1, (pop, dim)) * (ub - lb) + lb
    
    
    maxDiversity = diversidadHussain(population)
    XPL , XPT, state = porcentajesXLPXPT(maxDiversity, maxDiversity)
    
    # Genero un vector donde almacenaré los fitness de cada individuo
    fitness = np.zeros(pop)

    # Genero un vetor dedonde tendré mis soluciones rankeadas
    solutionsRanking = np.zeros(pop)
    
    # calculo de factibilidad de cada individuo y calculo del fitness inicial
    for i in range(population.shape[0]):
        population[i] = np.clip(population[i], lb, ub)
        fitness[i] = f(function, population[i])
        
    # Inicialización PSO
    velocities = np.random.rand(pop, dim) * 0.1
    # Inicializar p_best y g_best
    p_best_positions = np.copy(population)
    p_best_scores = np.copy(fitness)    
    
        
        
    # esta funcion ordena de menor a mayor
    solutionsRanking = np.argsort(fitness) # rankings de los mejores fitnes
    # es de maximizacion
    bestRowAux = solutionsRanking[pop-1]
    
    # DETERMINO MI MEJOR SOLUCION Y LA GUARDO 
    best = population[bestRowAux].copy()
    bestFitness = fitness[bestRowAux]
    
    initializationTime2 = time.time()
    
    initial_log(function, dim, mh, bestFitness, optimo, 
                initializationTime1, initializationTime2, XPT,
                XPL, maxDiversity, results)
    
    # Función objetivo para GOA, HBA, TDO y SHO
    def fo(x):
        x = np.clip(x, lb, ub)
        return f(function,x)
    
    # Bucle de iteraciones
    for iter in range(1, maxIter + 1):
        # obtengo mi tiempo inicial
        timerStart = time.time()
        
        if mh == "SCA_r2_r3_r4":
            population = iterarSCA_Chaotic_r2_r3_r4(maxIter, iter, dim, population.tolist(), best.tolist(), chaotic_map)
        if mh == "SCA_r2":
            population = iterarSCA_Chaotic_r2(maxIter, iter, dim, population.tolist(), best.tolist(), chaotic_map)
        if mh == "SCA_r3":
            population = iterarSCA_Chaotic_r3(maxIter, iter, dim, population.tolist(), best.tolist(), chaotic_map)
        if mh == "SCA_r4":
            population = iterarSCA_Chaotic_r4(maxIter, iter, dim, population.tolist(), best.tolist(), chaotic_map)
        if mh == "SCA_r2_r3":
            population = iterarSCA_Chaotic_r2_r3(maxIter, iter, dim, population.tolist(), best.tolist(), chaotic_map)
        if mh == "SCA_r2_r4":
            population = iterarSCA_Chaotic_r2_r4(maxIter, iter, dim, population.tolist(), best.tolist(), chaotic_map)
        if mh == "SCA_r3_r4":
            population = iterarSCA_Chaotic_r3_r4(maxIter, iter, dim, population.tolist(), best.tolist(), chaotic_map)
        if mh == "GWO":
            population = iterarGWO(maxIter, iter, dim, population.tolist(), fitness.tolist(), 'MIN')
        if mh == 'WOA':
            population = iterarWOA(maxIter, iter, dim, population.tolist(), best.tolist())
        if mh == 'PSA':
            population = iterarPSA(maxIter, iter, dim, population.tolist(), best.tolist())
        if mh == 'FOX':
            population = iterarFOX(maxIter, iter, dim, population.tolist(), best.tolist())
        if mh == 'EOO':
            population = iterarEOO(maxIter, iter, population.tolist(), best.tolist())
        if mh == 'RSA':
            population = iterarRSA(maxIter, iter, dim, population.tolist(), best.tolist(),0,1)
        if mh == 'GOA':
            population = iterarGOA(maxIter, iter, dim, population, best.tolist(), fitness.tolist(),fo, 'MAX')
        if mh == 'HBA':
            population = iterarHBA(maxIter, iter, dim, population.tolist(), best.tolist(), fitness.tolist(),fo, 'MAX')
        if mh == 'TDO':
            population = iterarTDO(maxIter, iter, dim, population.tolist(), fitness.tolist(),fo, 'MIN')
        if mh == 'SHO':
            population = iterarSHO(maxIter, iter, dim, population.tolist(), best.tolist(),fo, 'MIN')
        if mh == "GA":
            cross = float(param.split(",")[0])
            muta = float(param.split(",")[1])
            population = iterarGA_Continuo(population, fitness, cross, muta, copy_lb)
        if mh == 'PSO_r1_r2':
            population, velocities = iterarPSO_Chaotic_r1_r2(maxIter, iter, dim, population, velocities, p_best_positions, best, chaotic_map)
        if mh == 'PSO_r1':
            population, velocities = iterarPSO_Chaotic_r1(maxIter, iter, dim, population, velocities, p_best_positions, best, chaotic_map)
        if mh == 'PSO_r2':
            population, velocities = iterarPSO_Chaotic_r2(maxIter, iter, dim, population, velocities, p_best_positions, best, chaotic_map)
            
        # calculo de factibilidad de cada individuo y calculo del fitness inicial
        for i in range(pop):
            population[i] = np.clip(population[i], lb, ub)
            if mh == 'PSO':
                current_score = f(function, population[i])
                if current_score < p_best_scores[i]:
                    p_best_scores[i] = current_score
                    p_best_positions[i] = population[i]
                    fitness[i] = current_score
                    
            else:
                fitness[i] = f(function, population[i])
            
        solutionsRanking = np.argsort(fitness) # rankings de los mejores fitness
        
            
        #Conservo el best
        if fitness[solutionsRanking[0]] < bestFitness:
            bestFitness = fitness[solutionsRanking[0]].copy()
            best = np.copy(population[solutionsRanking[0]])

        div_t = diversidadHussain(population)

        if maxDiversity < div_t:
            maxDiversity = div_t
            
        XPL , XPT, state = porcentajesXLPXPT(div_t, maxDiversity)

        timerFinal = time.time()
        # calculo mi tiempo para la iteracion t
        timeEjecuted = timerFinal - timerStart
        log_progress(iter, maxIter, bestFitness, optimo, timeEjecuted, XPT, XPL, div_t, results)
        
    finalTime = time.time()
    
    final_log(bestFitness, initialTime, finalTime)
    
    results.close()
    
    binary = convert_into_binary(dirResult + f"{mh}_{function}_{id}.csv")
    
    bd.insertarIteraciones(f"{mh}_{function}", binary, id)
    bd.insertarResultados(bestFitness, finalTime - initialTime, best, id)
    bd.actualizarExperimento(id, 'terminado')
    
    os.remove(dirResult + f"{mh}_{function}_{id}.csv")