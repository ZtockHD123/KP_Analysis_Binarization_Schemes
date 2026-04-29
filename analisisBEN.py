import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import mannwhitneyu, wilcoxon

from Util.util import cargar_configuracion_exp, writeTofile
from BD.sqlite import BD
from Util.log import escribir_resumenes

# === Parámetros generales ===
GENERAR_DATA = False
GENERAR_GRAFICOS_DISTRIBUCION = False
GENERAR_MAE = True
GENERAR_ANALITICA_BEST_FAMILIA = False
GRAFICOS_BEST_FAMILIA = False
TEST_ESTADISTICO = False
COLORS = ['r', 'g']

# === Inicialización ===
bd = BD()

def generarMAE():
    # mapas_caoticos = ['LOG','PIECE','SINE','SINGER','SINU','TENT','CIRCLE']
    # instancias = bd.obtenerInstanciasEjecutadas("BEN")
    # mhs = bd.obtenerMHBD()    
    # # Nombres de las columnas que usarás al final
    # nombres_columnas = ['MH']
    # for instancia in instancias:
    #     titulo = instancia[1]
    #     nombres_columnas.append(titulo)
    # nombres_columnas.append('avg')
    # print(nombres_columnas)
    # print(mhs)
    # filas_acumuladas = []
    # for mh in mhs:
    #     for mapa in mapas_caoticos:
    #         algoritmo = f"{mh[0]}-{mapa}"
    #         fila_dinamica = [algoritmo]  # La primera columna es el nombre del MH
    #         promedio = []
    #         for instancia in instancias:
    #             print(f"[INFO] Procesando MH: {mh[0]} para el mapa caótico: {mapa} y la instancia {instancia[1]}...")
    #             resultados = bd.obtenerResultadosMAE(instancia[1], mh[0], mapa)
    #             optimo = bd.obtenerOptimoInstancia(instancia[1])[0][0]
    #             if instancia[1] == 'F8':
    #                 optimo = optimo * 30
    #             ae = [abs(resultado[0] - optimo) for resultado in resultados]
    #             mae = np.mean(ae)
    #             promedio.append(mae)
    #             fila_dinamica.append(np.round(mae, 2))
    #         fila_dinamica.append(np.round(np.mean(promedio), 2))
    #         filas_acumuladas.append(fila_dinamica)
    # # pd.options.display.float_format = '{:.2e}'.format
    # df_final = pd.DataFrame(filas_acumuladas, columns=nombres_columnas)
    # print(df_final)
    # # pd.reset_option('display.float_format')
        
        
    # mapas_caoticos = ['LOG','PIECE','SINE','SINGER','SINU','TENT','CIRCLE']
    # instancias = bd.obtenerInstanciasEjecutadas("BEN")
    # mhs = bd.obtenerMHBD()    
    # print(mhs)
    # # Nombres de las columnas que usarás al final
    # nombres_columnas = ['Inst']
    # for mh in mhs:
    #     for mapa in mapas_caoticos:
    #         algoritmo = f"{mh[0]}-{mapa}"
    #         nombres_columnas.append(algoritmo)
    # print(nombres_columnas)
    # print(mhs)
    # filas_acumuladas = []
    # for instancia in instancias:
    #     fila_dinamica = [instancia[1]]  # La primera columna es el nombre de la instancia
    #     for mh in mhs:
    #         for mapa in mapas_caoticos:
    #             # promedio = []
    #             print(f"[INFO] Procesando MH: {mh[0]} para el mapa caótico: {mapa} y la instancia {instancia[1]}...")
    #             resultados = bd.obtenerResultadosMAE(instancia[1], mh[0], mapa)
    #             optimo = bd.obtenerOptimoInstancia(instancia[1])[0][0]
    #             if instancia[1] == 'F8':
    #                 optimo = optimo * 30
    #             ae = [abs(resultado[0] - optimo) for resultado in resultados]
    #             mae = np.mean(ae)
    #             fila_dinamica.append(mae)
    #     filas_acumuladas.append(fila_dinamica)
    # pd.options.display.float_format = '{:.2e}'.format
    # df_final = pd.DataFrame(filas_acumuladas, columns=nombres_columnas)
    # df_final.loc[len(df_final)] = df_final.mean(axis=0)
    # print(df_final)
    # df_final.to_csv(f'./Resultados/resumen/BEN/MAE_BEN_chaotic.csv', index=False, float_format='%.2e')
    # pd.reset_option('display.float_format')
        
        
    instancias = bd.obtenerInstanciasEjecutadas("BEN")
    mhs = bd.obtenerMHBD()    
    print(mhs)
    # # Nombres de las columnas que usarás al final
    nombres_columnas = ['Inst']
    for mh in mhs:
        if mh[0] == 'SCA':
            algoritmo = f"{mh[0]}"
            nombres_columnas.append(algoritmo)
    print(nombres_columnas)
    print(mhs)
    filas_acumuladas = []
    for instancia in instancias:
        fila_dinamica = [instancia[1]]  # La primera columna es el nombre de la instancia
        for mh in mhs:
            if mh[0] == 'SCA':
                # promedio = []
                print(f"[INFO] Procesando MH: {mh[0]} y la instancia {instancia[1]}...")
                resultados = bd.obtenerResultadosMAENATIVO(instancia[1], mh[0])
                optimo = bd.obtenerOptimoInstancia(instancia[1])[0][0]
                if instancia[1] == 'F8':
                    optimo = optimo * 30
                ae = [abs(resultado[0] - optimo) for resultado in resultados]
                mae = np.mean(ae)
                fila_dinamica.append(mae)
        filas_acumuladas.append(fila_dinamica)
    pd.options.display.float_format = '{:.2e}'.format
    df_final = pd.DataFrame(filas_acumuladas, columns=nombres_columnas)
    df_final.loc[len(df_final)] = df_final.mean(axis=0)
    print(df_final)
    df_final.to_csv(f'./Resultados/resumen/BEN/MAE_BEN_NATIVO.csv', index=False, float_format='%.2e')
    pd.reset_option('display.float_format')

def generar_graficos_separados(ruta_csv, ruta_salida_box, nombre_instancia):
    # 1. Leer los datos
    print(f"[INFO] Leyendo datos de: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    
    # 2. Configurar el estilo visual
    sns.set_theme(style="whitegrid")
    
    # ---------------------------------------------------------
    # GRÁFICO 1: CAJA Y BIGOTES (BOXPLOT) INDEPENDIENTE
    # ---------------------------------------------------------
    # Creamos un lienzo nuevo solo para este gráfico
    plt.figure(figsize=(14, 8)) 
    
    sns.boxplot(x='MH', y='Data', data=df, palette='Set2')
    titulo = nombre_instancia
    plt.title(f'Boxplot - Instance {titulo}', fontsize=16, fontweight='bold')
    plt.xlabel('Experiment', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Guardamos y cerramos
    ruta_boxplot = os.path.join(ruta_salida_box, f'boxplot_{nombre_instancia}.png')
    plt.savefig(ruta_boxplot, dpi=300)
    plt.close() # MUY IMPORTANTE: Cierra el lienzo para no mezclar con el siguiente
    print(f"✅ Boxplot guardado en: {ruta_boxplot}")

    # # ---------------------------------------------------------
    # # GRÁFICO 2: VIOLÍN (VIOLINPLOT) INDEPENDIENTE
    # # ---------------------------------------------------------
    # # Creamos OTRO lienzo nuevo
    # plt.figure(figsize=(14, 8))
    
    # sns.violinplot(x='MH', y='Data', data=df, palette='Set3', inner='quartile')
    
    # plt.title(f'Violinplot - Instance {nombre_instancia}', fontsize=16, fontweight='bold')
    # plt.xlabel('MH', fontsize=12)
    # plt.ylabel('Fitness', fontsize=12)
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    
    # # Guardamos y cerramos
    # ruta_violin = os.path.join(ruta_salida_violin, f'violinplot_{nombre_instancia}.png')
    # plt.savefig(ruta_violin, dpi=300)
    # plt.close()
    # print(f"✅ Violinplot guardado en: {ruta_violin}")

def gestionar_archivo_csv(ruta_archivo):
    # Convertimos la ruta de texto a un objeto Path
    archivo = Path(ruta_archivo)
    
    # 1. Verificamos si existe
    if archivo.exists():
        print(f"✅ ESTADO: El archivo '{archivo.name}' YA EXISTE.")
        return {'mode': 'a', 'header': False}
        
    else:
    # 2 y 3. Si no existe, lo creamos e indicamos el modo
        print(f"⚠️ ESTADO: El archivo '{archivo.name}' NO EXISTE.")
        # .touch() crea un archivo en blanco en esa ruta
        df = pd.DataFrame(columns=['MH', 'Data'])
        df.to_csv(archivo, index=False)
        print("✨ ACCIÓN: Archivo creado exitosamente.")
        return {'mode': 'w', 'header': True}

def generar_data():
    instancias = bd.obtenerInstanciasEjecutadas("BEN")
    print(f"[INFO] Analizando {len(instancias)} instancias de BEN...")
    
    for instancia in instancias:
        print(f"\n [INFO] Procesando instancia: {instancia[1]}")
        mhs = bd.obtenerMHEjecutadas(instancia[1])
        print(f"[INFO] Analizando {len(mhs)} MHS para la instancia {instancia[1]}...")
        
        for mh in mhs:
            print(f"[INFO] Procesando MHS: {mh[0]} para la instancia {instancia[1]}...")
            
            experimentos = bd.obtenerExperimentosEjecutadosInstMH(instancia[1], mh[0])
            print(f"[INFO] Analizando {len(experimentos)} experimentos para la instancia {instancia[1]} y MHS {mh[0]}...")
            
            for experimento in experimentos:
                print(f"[INFO] Procesando experimento: {experimento[0]} para la instancia {instancia[1]} y MHS {mh[0]}...")
                resultados = bd.obtenerResultadosTiempos(instancia[1], experimento[0], mh[0])
                print(f"\n [INFO] Analizando {len(resultados)} resultados para la instancia {instancia[1]}, MHS {mh[0]} y experimento {experimento[0]}...")
                
                ruta_fitness = f'./Resultados/data/BEN/fitness/{instancia[1]}.csv'
                ruta_tiempos = f'./Resultados/data/BEN/tiempos/{instancia[1]}.csv'
                
                # verificacion archivo de fitness
                conf_fitness = gestionar_archivo_csv(ruta_fitness)
                # verificacion archivo de tiempos
                conf_tiempos = gestionar_archivo_csv(ruta_tiempos)
                
                datos_fitness = []
                datos_tiempos = []
                datos_mh = []
                nombre_mh = f'{mh[0]}-{experimento[0].split(" ")[2]}'
                
                for resultado in resultados:
                    datos_fitness.append(resultado[0])  # fitness
                    datos_tiempos.append(resultado[1])  # tiempo
                    datos_mh.append(nombre_mh)
                
                nuevos_datos_fitness = pd.DataFrame({
                    'MH': datos_mh,
                    'Data': datos_fitness
                })
                
                nuevos_datos_tiempos = pd.DataFrame({
                    'MH': datos_mh,
                    'Data': datos_tiempos
                })
            
                # Guardamos el DataFrame
                nuevos_datos_fitness.to_csv(
                    ruta_fitness, 
                    mode=conf_fitness['mode'], 
                    header=conf_fitness['header'], 
                    index=False  # ¡MUY IMPORTANTE! (Explicación abajo)
                )

                nuevos_datos_tiempos.to_csv(
                    ruta_tiempos, 
                    mode=conf_tiempos['mode'], 
                    header=conf_tiempos['header'], 
                    index=False  # ¡MUY IMPORTANTE! (Explicación abajo)
                )

def analizar_instancias():
    """
    Función principal para analizar instancias del KP.
    Procesa cada instancia, genera gráficos y resúmenes estadísticos.
    """
    if GENERAR_DATA:
        generar_data()
        
    if GENERAR_GRAFICOS_DISTRIBUCION:
        instancias = bd.obtenerInstanciasEjecutadas("BEN")
        for instancia in instancias:
            ruta_entrada = f'./Resultados/data/BEN/fitness/{instancia[1]}.csv'
            ruta_salida_box = f'./Resultados/boxplot/BEN/'
            
            generar_graficos_separados(ruta_entrada, ruta_salida_box, instancia[1])
    
    if GENERAR_MAE:
        generarMAE()
        
    # if GENERAR_ANALITICA_BEST_FAMILIA:
    #     generar_analitica_familiar()
        
    # if GRAFICOS_BEST_FAMILIA:
    #     graficos_best()
        
    # if TEST_ESTADISTICO:
    #     test_estadistico()

    # print(os.listdir('./Resultados/graficos/KP/'))  # Verifica que los archivos se hayan guardado correctamente

    print("[INFO] Análisis BEN completado con éxito.")
    print("-" * 50)