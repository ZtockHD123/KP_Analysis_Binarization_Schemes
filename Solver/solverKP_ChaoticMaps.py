import numpy as np
import os
import time

from Problem.KP.problem import KP
from Metaheuristics.imports import iterarGWO,iterarSCA,iterarWOA,iterarPSA,iterarMFO,iterarGA
from Metaheuristics.imports import iterarPSO,iterarFOX,iterarEOO,iterarRSA,iterarGOA,iterarHBA,iterarTDO,iterarSHO
from Diversity.imports import diversidadHussain,porcentajesXLPXPT
from Discretization import discretization as b
from Util.util import convert_into_binary
from Util.log import initial_log, log_progress, final_log_kp
from BD.sqlite import BD
from ChaoticMaps.chaoticMaps import logisticMap,piecewiseMap,sineMap,singerMap,sinusoidalMap,tentMap,circleMap

def solverKP_ChaoticMaps(id, mh, maxIter, pop, instancia, DS, param):
    
    dirResult = './Resultados/'
    instance = KP(instancia)
    optimo = instance.getOptimum()
    chaotic_map = None
    
    chaotic = DS[1].split("_")[1]
    
    if chaotic == 'LOG':
        elementQuantity = maxIter * pop * instance.getItems()
        chaotic_map = logisticMap(0.7,elementQuantity)
    if chaotic == 'PIECE':
        elementQuantity = maxIter * pop * instance.getItems()
        chaotic_map = piecewiseMap(0.7,elementQuantity)
    if chaotic == 'SINE':
        elementQuantity = maxIter * pop * instance.getItems()
        chaotic_map = sineMap(0.7,elementQuantity)
    if chaotic == 'SINGER':
        elementQuantity = maxIter * pop * instance.getItems()
        chaotic_map = singerMap(0.7,elementQuantity)
    if chaotic == 'SINU':
        elementQuantity = maxIter * pop * instance.getItems()
        chaotic_map = sinusoidalMap(0.7,elementQuantity)
    if chaotic == 'TENT':
        elementQuantity = maxIter * pop * instance.getItems()
        chaotic_map = tentMap(0.6,elementQuantity)
    if chaotic == 'CIRCLE':
        elementQuantity = maxIter * pop * instance.getItems()
        chaotic_map = circleMap(0.7,elementQuantity)
    
    # tomo el tiempo inicial de la ejecucion
    initialTime = time.time()
    
    initializationTime1 = time.time()
    
    results = open(dirResult+mh+"_"+instancia.split(".")[0]+"_"+str(id)+".csv", "w")
    results.write(
        f'iter,fitness,time,XPL,XPT,DIV\n'
    )
    
    # Genero una población inicial binaria, esto ya que nuestro problema es binario
    population = np.random.randint(low=0, high=2, size = (pop, instance.getItems()))

    maxDiversity = diversidadHussain(population)
    XPL , XPT, state = porcentajesXLPXPT(maxDiversity, maxDiversity)
    
    # Genero un vector donde almacenaré los fitness de cada individuo
    fitness = np.zeros(pop)

    # Genero un vetor dedonde tendré mis soluciones rankeadas
    solutionsRanking = np.zeros(pop)
    
    # calculo de factibilidad de cada individuo y calculo del fitness inicial
    for i in range(population.__len__()):
        flag = instance.factibilityTest(population[i])
        if not flag: #solucion infactible
            population[i] = instance.repair(population[i])
            

        fitness[i] = instance.fitness(population[i])
        
    # Inicialización PSO
    velocities = np.random.rand(pop, instance.getItems()) * 0.1
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
    
    # PARA MFO
    bestFitnessArray = fitness[solutionsRanking] 
    bestSolutions = population[solutionsRanking]
    
    matrixBin = population.copy()
    
    initializationTime2 = time.time()
    
    # mostramos nuestro fitness iniciales
    initial_log(instancia, instance.getItems(), mh, bestFitness, instance.getOptimum(), 
                initializationTime1, initializationTime2, XPT,
                XPL, maxDiversity, results)
    
    bestPop = np.copy(population)

    # Función objetivo para GOA, HBA, TDO y SHO
    def fo(x):
        x = b.aplicarBinarizacion(x.tolist(), DS[0], DS[1], best, matrixBin[i].tolist(), iter, pop, maxIter, i, chaotic_map)
        x = instance.repair(x) # Reparación de soluciones
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
            population = iterarSCA(maxIter, iter, instance.getItems(), population.tolist(), best.tolist())
        if mh == "GWO":
            population = iterarGWO(maxIter, iter, instance.getItems(), population.tolist(), fitness.tolist(), 'MIN')
        if mh == 'WOA':
            population = iterarWOA(maxIter, iter, instance.getItems(), population.tolist(), best.tolist())
        if mh == 'PSA':
            population = iterarPSA(maxIter, iter, instance.getItems(), population.tolist(), best.tolist())
        if mh == "MFO":
            population, bestSolutions = iterarMFO(maxIter, iter, instance.getItems(), len(population), population, bestSolutions, fitness, bestFitnessArray )
        if mh == "GA":
            cross = float(param.split(",")[0])
            muta = float(param.split(",")[1])
            population = iterarGA(population, fitness, cross, muta, 'MAX')
        if mh == 'PSO':
            population, velocities = iterarPSO(maxIter, iter, instance.getItems(), population, velocities, p_best_positions, best)
        if mh == 'FOX':
            population = iterarFOX(maxIter, iter, instance.getItems(), population.tolist(), best.tolist())
        if mh == 'EOO':
            population = iterarEOO(maxIter, iter, population.tolist(), best.tolist())
        if mh == 'RSA':
            population = iterarRSA(maxIter, iter, instance.getItems(), population.tolist(), best.tolist(),0,1)
        if mh == 'GOA':
            population = iterarGOA(maxIter, iter, instance.getItems(), population, best.tolist(), fitness.tolist(),fo, 'MAX')
        if mh == 'HBA':
            population = iterarHBA(maxIter, iter, instance.getItems(), population.tolist(), best.tolist(), fitness.tolist(),fo, 'MAX')
        if mh == 'TDO':
            population = iterarTDO(maxIter, iter, instance.getItems(), population.tolist(), fitness.tolist(),fo, 'MAX')
        if mh == 'SHO':
            population = iterarSHO(maxIter, iter, instance.getItems(), population.tolist(), best.tolist(),fo, 'MAX')
        
        # Binarizo, calculo de factibilidad de cada individuo y calculo del fitness
        for i in range(population.__len__()):

            if mh != "GA":
                population[i] = b.aplicarBinarizacion(population[i].tolist(), DS[0], DS[1], best, matrixBin[i].tolist(), iter-1, pop, maxIter, i, chaotic_map)

            flag = instance.factibilityTest(population[i])
            # print(aux)
            if not flag: #solucion infactible
                population[i] = instance.repair(population[i])
                

            if mh == 'PSO':
                current_score = instance.fitness(population[i])
                if current_score > p_best_scores[i]:
                    p_best_scores[i] = current_score
                    p_best_positions[i] = population[i]
                    fitness[i] = current_score
            else:
                fitness[i] = instance.fitness(population[i])


        solutionsRanking = np.argsort(fitness) # rankings de los mejores fitness
        
        #Conservo el best
        if fitness[solutionsRanking[pop-1]] > bestFitness:
            bestFitness = fitness[solutionsRanking[pop-1]].copy()
            best = population[solutionsRanking[pop-1]].copy()
        matrixBin = population.copy()

        div_t = diversidadHussain(population)

        if maxDiversity < div_t:
            maxDiversity = div_t
            
        XPL , XPT, state = porcentajesXLPXPT(div_t, maxDiversity)

        timerFinal = time.time()
        # calculo mi tiempo para la iteracion t
        timeEjecuted = timerFinal - timerStart
        
        log_progress(iter, maxIter, bestFitness, optimo, timerFinal - timerStart, XPT, XPL, div_t, results)
    finalTime = time.time()
    timeExecution = finalTime - initialTime
    #print("Tiempo de ejecucion (s): "+str(timeExecution))
    
    final_log_kp(bestFitness, str(sum(best)), initialTime, finalTime)
    
    results.close()
    
    binary = convert_into_binary(dirResult + mh+"_" + instancia.split(".")[0] + "_" + str(id) + ".csv")

    fileName = mh+"_"+instancia.split(".")[0]

    bd = BD()
    bd.insertarIteraciones(fileName, binary, id)
    bd.insertarResultados(bestFitness, timeExecution, best, id)
    bd.actualizarExperimento(id, 'terminado')
    
    os.remove(dirResult + mh + "_" + instancia.split(".")[0] + "_" + str(id) + ".csv")
    
    
    