import numpy as np
import os
import time

from Problem.USCP.problem import USCP
from Metaheuristics.imports import iterarGWO,iterarSCA,iterarWOA,iterarPSA,iterarMFO,iterarGA
from Metaheuristics.imports import iterarPSO,iterarFOX,iterarEOO,iterarRSA,iterarGOA,iterarHBA,iterarTDO,iterarSHO
from Diversity.imports import diversidadHussain,porcentajesXLPXPT
from Discretization import discretization as b
from Util.log import initial_log_scp_uscp, log_progress, final_log_scp
from Util.util import convert_into_binary
from BD.sqlite import BD

def solverUSCP(id, mh, maxIter, pop, instances, DS, repairType, param):
    
    bd = BD()
    dirResult = './Resultados/Transitorio/'
    instance = USCP(instances)
    
    chaotic_map = None
    
    # tomo el tiempo inicial de la ejecucion
    initialTime = time.time()
    initializationTime1 = time.time()
    
    results = open(dirResult + mh + "_" + instances.split(".")[0] + "_" + str(id) + ".csv", "w")
    results.write(f'iter,fitness,time,XPL,XPT,DIV\n')
    
    # Genero una población inicial binaria, esto ya que nuestro problema es binario
    population = np.random.randint(low=0, high=2, size = (pop, instance.getColumns()))

    maxDiversity = diversidadHussain(population)
    XPL , XPT, state = porcentajesXLPXPT(maxDiversity, maxDiversity)
    
    # Genero un vector donde almacenaré los fitness de cada individuo
    fitness = np.zeros(pop)

    # Genero un vetor dedonde tendré mis soluciones rankeadas
    solutionsRanking = np.zeros(pop)
    
    # calculo de factibilidad de cada individuo y calculo del fitness inicial
    for i in range(population.__len__()):
        flag, aux = instance.factibilityTest(population[i])
        if not flag: #solucion infactible
            population[i] = instance.repair(population[i], repairType)
            

        fitness[i] = instance.fitness(population[i])
    
    # Inicialización PSO
    velocities = np.random.rand(pop, instance.getColumns()) * 0.1
    # Inicializar p_best y g_best
    p_best_positions = np.copy(population)
    p_best_scores = np.copy(fitness)  
    
        
    solutionsRanking = np.argsort(fitness) # rankings de los mejores fitnes
    bestRowAux = solutionsRanking[0].copy()
    # DETERMINO MI MEJOR SOLUCION Y LA GUARDO 
    best = population[bestRowAux].copy()
    bestFitness = fitness[bestRowAux].copy()
    
    # PARA MFO
    bestFitnessArray = fitness[solutionsRanking].copy() 
    bestSolutions = population[solutionsRanking].copy()
    
    matrixBin = population.copy()
    
    initializationTime2 = time.time()
    
    # mostramos nuestro fitness iniciales
    initial_log_scp_uscp(instance, DS, bestFitness, instances, initializationTime1, initializationTime2, XPT, XPL, maxDiversity, results)

    bestPop = np.copy(population)

    # Función objetivo para GOA, HBA, TDO y SHO
    def fo(x):
        x = b.aplicarBinarizacion(x.tolist(), DS[0], DS[1], best, matrixBin[i].tolist(), iter, pop, maxIter, i, chaotic_map)
        x = instance.repair(x, repairType) # Reparación de soluciones
        return x,instance.fitness(x) # Return de la solución reparada y valor de función objetivo
    
    for iter in range(1, maxIter + 1):
        # obtengo mi tiempo inicial
        timerStart = time.time()
        
        if mh == "MFO":
            for i in range(bestSolutions.__len__()):
                bestFitnessArray[i] = instance.fitness(bestSolutions[i])
        
        # perturbo la poblacion con la metaheuristica, pueden usar SCA y GWO
        # en las funciones internas tenemos los otros dos for, for de individuos y for de dimensiones
        # print(population)
        if mh == "SCA":
            population = iterarSCA(maxIter, iter, instance.getColumns(), population.tolist(), best.tolist())
        if mh == "GWO":
            population = iterarGWO(maxIter, iter, instance.getColumns(), population.tolist(), fitness.tolist(), 'MIN')
        if mh == 'WOA':
            population = iterarWOA(maxIter, iter, instance.getColumns(), population.tolist(), best.tolist())
        if mh == 'PSA':
            population = iterarPSA(maxIter, iter, instance.getColumns(), population.tolist(), best.tolist())
        if mh == "MFO":
            population, bestSolutions = iterarMFO(maxIter, iter, instance.getColumns(), len(population), population, bestSolutions, fitness, bestFitnessArray )
        if mh == "GA":
            cross = float(param.split(",")[0])
            muta = float(param.split(",")[1])
            population = iterarGA(population, fitness, cross, muta, 'MIN')
        if mh == 'FOX':
            population = iterarFOX(maxIter, iter, instance.getColumns(), population.tolist(), best.tolist())
        if mh == 'EOO':
            population = iterarEOO(maxIter, iter, population.tolist(), best.tolist())
        if mh == 'RSA':
            population = iterarRSA(maxIter, iter, instance.getColumns(), population.tolist(), best.tolist(),0,1)
        if mh == 'GOA':
            population = iterarGOA(maxIter, iter, instance.getColumns(), population, best.tolist(), fitness.tolist(),fo, 'MIN')
        if mh == 'HBA':
            population = iterarHBA(maxIter, iter, instance.getColumns(), population.tolist(), best.tolist(), fitness.tolist(),fo, 'MIN')
        if mh == 'TDO':
            population = iterarTDO(maxIter, iter, instance.getColumns(), population.tolist(), fitness.tolist(),fo, 'MIN')
        if mh == 'SHO':
            population = iterarSHO(maxIter, iter, instance.getColumns(), population.tolist(), best.tolist(),fo, 'MIN')
        if mh == 'PSO':
            population, velocities = iterarPSO(maxIter, iter, instance.getColumns(), population, velocities, p_best_positions, best)
        
        # Binarizo, calculo de factibilidad de cada individuo y calculo del fitness
        for i in range(population.__len__()):

            if mh != "GA":
                population[i] = b.aplicarBinarizacion(population[i].tolist(), DS[0], DS[1], best, matrixBin[i].tolist(), iter, pop, maxIter, i, chaotic_map).copy()

            flag, aux = instance.factibilityTest(population[i])
            # print(aux)
            if not flag: #solucion infactible
                population[i] = instance.repair(population[i], repairType).copy()
                
            if mh == 'PSO':
                current_score = instance.fitness(population[i])
                if current_score < p_best_scores[i]:
                    p_best_scores[i] = current_score
                    p_best_positions[i] = population[i]
                    fitness[i] = current_score
            else:
                fitness[i] = instance.fitness(population[i])


        solutionsRanking = np.argsort(fitness) # rankings de los mejores fitness
        
        #Conservo el best
        if fitness[solutionsRanking[0]] < bestFitness:
            bestFitness = fitness[solutionsRanking[0]].copy()
            best = population[solutionsRanking[0]].copy()
        matrixBin = population.copy()

        div_t = diversidadHussain(population)

        if maxDiversity < div_t:
            maxDiversity = div_t
            
        XPL , XPT, state = porcentajesXLPXPT(div_t, maxDiversity)

        timerFinal = time.time()
        # calculo mi tiempo para la iteracion t
        timeEjecuted = timerFinal - timerStart
        
        log_progress(iter, maxIter, bestFitness, instance.getOptimum(), timeEjecuted, XPT, XPL, div_t, results)
    finalTime = time.time()
    
    numberOfSubsets = str(sum(best))
    
    final_log_scp(bestFitness, numberOfSubsets, initialTime, finalTime)
    
    results.close()
    
    binary = convert_into_binary(dirResult + mh + "_" + instances.split(".")[0] + "_" + str(id) + ".csv")
    
    fileName = mh + "_" + instances.split(".")[0]

    bd.insertarIteraciones(fileName, binary, id)
    bd.insertarResultados(bestFitness, finalTime - initialTime, best, id)
    bd.actualizarExperimento(id, 'terminado')
    
    os.remove(dirResult + mh + "_" + instances.split(".")[0] + "_" + str(id) + ".csv")
    
    
    