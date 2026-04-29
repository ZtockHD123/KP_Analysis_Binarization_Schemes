from BD.sqlite import BD
import pandas as pd


def encontrar_outliers_iqr(df, columna_id, columna_resultado):
    """
    Encuentra outliers en un DataFrame usando el método IQR.
    
    Argumentos:
    df -- El DataFrame de pandas.
    columna_id -- El nombre de la columna con los IDs de experimento.
    columna_resultado -- El nombre de la columna con los resultados numéricos.
    
    Retorna:
    Un DataFrame de pandas que contiene solo las filas que son outliers.
    """
    
    # 1. Extraer la columna de resultados
    resultados = df[columna_resultado]
    
    # 2. Calcular Q1, Q3 e IQR
    Q1 = resultados.quantile(0.25)
    Q3 = resultados.quantile(0.75)
    IQR = Q3 - Q1
    
    # 3. Calcular los límites (rangos) para outliers
    limite_inferior = Q1 - (1.5 * IQR)
    limite_superior = Q3 + (1.5 * IQR)
    
    # 4. Imprimir la información de los rangos (como solicitaste)
    # print("--- Análisis de Outliers (Método IQR) ---")
    # print(f"Cuartil 1 (Q1): {Q1:.2f}")
    # print(f"Cuartil 3 (Q3): {Q3:.2f}")
    # print(f"Rango Intercuartílico (IQR): {IQR:.2f}")
    # print(f"-------------------------------------------")
    # print(f"❗️ Rango para datos NO outliers:")
    # print(f"   Límite Inferior: {limite_inferior:.2f}")
    # print(f"   Límite Superior: {limite_superior:.2f}")
    # print("-------------------------------------------")
    
    # 5. Identificar y filtrar los outliers
    # Un outlier es cualquier valor < limite_inferior O > limite_superior
    outliers = df[(resultados < limite_inferior) | (resultados > limite_superior)]
    
    return outliers


bd = BD()

instancias_ejecutadas = bd.obtenerInstanciasEjecutadas('BEN')
print(instancias_ejecutadas)

total_outliers = 0
total_sin_outliers = 0

for instancia in instancias_ejecutadas:
    instancia_id = instancia[0]
    nombre = instancia[1]
    experimentos = bd.obtenerExperimentosCorridas(nombre)
    mhs = bd.obtenerMHEjecutadas(nombre)
    
    for mh in mhs:
        for experimento in experimentos:            
            resultados = bd.obtenerResultadosByMH_Experimentos(nombre, mh[0],experimento[0])
            estructura = bd.obtenerEstructuraExperimento(nombre, mh[0],experimento[0])
            data_experimento = {
                'experimento': estructura[0][1],
                'MH': estructura[0][2],
                'binarizacion': estructura[0][3],
                'paramMH': estructura[0][4],
                'ML': '',
                'paramML': '',
                'ML_FS': '',
                'paramML_FS': '',
                'estado': 'pendiente'
            }
            
            
            
            datos = {'id_experimento':[], 'fitness':[]}
            for resultado in resultados:
                datos['id_experimento'].append(resultado[0])
                datos['fitness'].append(resultado[1])
            
            mi_df = pd.DataFrame(datos)
            
            # 2. Llama a la función para encontrar los outliers
            # Pasa tu DataFrame y los nombres EXACTOS de tus columnas
            outliers_encontrados = encontrar_outliers_iqr(mi_df, 
                                                        columna_id='id_experimento', 
                                                        columna_resultado='fitness')

            # 3. Muestra los resultados
            print("\n--- Experimentos Identificados como Outliers ---")
            if outliers_encontrados.empty:
                print("✅ ¡Buenas noticias! No se encontraron outliers en la muestra.")
            else:
                id_outliers = outliers_encontrados['id_experimento'].tolist()
                
                for id in id_outliers:
                    bd.actualizarExperimento(id, 'outlier')
                    bd.insertarExperimentos(data_experimento, 1, instancia_id)
                
                
            print(f"Total de experimentos con outliers para {nombre} {mh[0]} {experimento[0]}: {len(outliers_encontrados)}")
            print(f"Total de experimentos sin outliers para {nombre} {mh[0]} {experimento[0]}: {len(mi_df) - len(outliers_encontrados)}")
            total_outliers += len(outliers_encontrados)
            total_sin_outliers += (len(mi_df) - len(outliers_encontrados))
            
print("-" *100)
print(f"Total de experimentos outliers: {total_outliers}")
print(f"Total de experimentos que no son outliers: {total_sin_outliers}")
        