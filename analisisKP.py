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
GENERAR_RPD = False
GENERAR_ANALITICA_BEST_FAMILIA = False
GRAFICOS_BEST_FAMILIA = False
TEST_ESTADISTICO = True
COLORS = ['r', 'g']

# === Inicialización ===
bd = BD()

def obtenerExperimentosCSV():
    ruta_fitness = './Resultados/Test/KP/data/'
    df = pd.read_csv(os.path.join(ruta_fitness, 'knapPI_1_100_1000_1.csv'))
    return df['MH'].unique().tolist()

def test_estadistico():
    generar_datos = False
    test_wilconxon= True
    test_por_algoritmo = True
    tabla_doble_entrada = True
    binarizacion_en_analisis = ['STD', 'STD_CIRCLE']
    if generar_datos:
        ruta_base = './Resultados/data/KP/fitness/'

        instancias = bd.obtenerInstanciasEjecutadas("KP")
        
        for instancia in instancias:
            df_1 = pd.read_csv(f'{ruta_base}GWO/{instancia[1]}.csv')
            cambios = {
                f'{binarizacion_en_analisis[0]}': f'GWO_{binarizacion_en_analisis[0]}',
                f'{binarizacion_en_analisis[1]}': f'GWO_{binarizacion_en_analisis[1]}'
            }
            df_1['MH'] = df_1['MH'].replace(cambios)
            df_2 = pd.read_csv(f'{ruta_base}SCA/{instancia[1]}.csv')
            cambios = {
                f'{binarizacion_en_analisis[0]}': f'SCA_{binarizacion_en_analisis[0]}',
                f'{binarizacion_en_analisis[1]}': f'SCA_{binarizacion_en_analisis[1]}'
            }
            df_2['MH'] = df_2['MH'].replace(cambios)
            df_3 = pd.read_csv(f'{ruta_base}WOA/{instancia[1]}.csv')
            cambios = {
                f'{binarizacion_en_analisis[0]}': f'WOA_{binarizacion_en_analisis[0]}',
                f'{binarizacion_en_analisis[1]}': f'WOA_{binarizacion_en_analisis[1]}'
            }
            df_3['MH'] = df_3['MH'].replace(cambios)
            
            # print(df_1.head())
            # print(df_2.head())
            # print(df_3.head())
            df_unificado = pd.concat([
                df_1.query(f"MH == 'GWO_{binarizacion_en_analisis[0]}' or MH == 'GWO_{binarizacion_en_analisis[1]}'"),
                df_2.query(f"MH == 'SCA_{binarizacion_en_analisis[0]}' or MH == 'SCA_{binarizacion_en_analisis[1]}'"),
                df_3.query(f"MH == 'WOA_{binarizacion_en_analisis[0]}' or MH == 'WOA_{binarizacion_en_analisis[1]}'")
            ], ignore_index=True)
            
            df_unificado.to_csv(f'./Resultados/Test/KP/data/{instancia[1]}.csv', index=False)
            print(f"✅ Archivo unificado para la instancia {instancia[1]} generado exitosamente.")

    print(f"\n\n[INFO] Iniciando generación de test estadístico")
    
    algoritmos = obtenerExperimentosCSV() 
    if test_wilconxon:
        print(f"\n[INFO] Iniciando test Wilcoxon por instancia...")
        # 2. Preparamos las columnas dinámicas para el DataFrame de Wilcoxon
        # La primera columna es 'vs', seguida de todos los algoritmos
        columnas_wilcoxon = ['vs'] + algoritmos

        ruta_fitness = './Resultados/Test/KP/data/'
        ruta_salida_base = './Resultados/Test/KP/wilconxon/'

        # Asegurarnos de que el directorio de salida existe para evitar errores
        os.makedirs(ruta_salida_base, exist_ok=True)

        # 3. Iteración sobre los archivos (Instancias)
        for archivo in os.listdir(ruta_fitness):
            if not archivo.endswith('.csv'):
                continue
                
            # Inicializamos el dataframe dinámicamente para esta instancia
            resumen = pd.DataFrame(columns=columnas_wilcoxon)
            df = pd.read_csv(os.path.join(ruta_fitness, archivo))
            
            # MEJORA: Pre-extraer los valores a un diccionario. 
            # Esto evita hacer df.query() decenas de veces por archivo y lo hace instantáneo.
            fitness_dict = {}
            for algo, datos in df.groupby('MH'):
                valores = datos['Data'].dropna().values
                if len(valores) > 0:
                    fitness_dict[algo] = valores

            # 4. Comparación uno a uno (Matriz NxN)
            for algoritmo in algoritmos:
                resultados_fila = [algoritmo] # El primer valor de la fila va en la columna 'vs'
                
                for algoritmo_2 in algoritmos:
                    # Diagonal de la matriz: un algoritmo comparado consigo mismo
                    if algoritmo == algoritmo_2:
                        resultados_fila.append('X')
                    else:
                        # Validamos que AMBOS algoritmos tengan datos registrados en esta instancia
                        if algoritmo in fitness_dict and algoritmo_2 in fitness_dict:
                            x_vals = fitness_dict[algoritmo]
                            y_vals = fitness_dict[algoritmo_2]
                            
                            # Ejecutamos el test
                            stat, p_value = mannwhitneyu(x_vals, y_vals, alternative='greater')
                            resultados_fila.append(np.round(p_value, 3))
                        else:
                            # Si un algoritmo falló o no está en este CSV, ponemos 'N/A' (Not Available)
                            resultados_fila.append('N/A')
                            
                # Insertamos la fila dinámica
                resumen.loc[len(resumen)] = resultados_fila
                
            print(f"[INFO] ✅ Resumen Wilcoxon para {archivo} finalizado...")
            
            # 5. Guardar dinámicamente el archivo por instancia
            nombre_instancia = archivo.split(".")[0]
            ruta_salida = os.path.join(ruta_salida_base, f'test_wilcoxon_{nombre_instancia}.csv')
            resumen.to_csv(ruta_salida, index=False)

    if test_por_algoritmo:
        print(f"\n[INFO] Iniciando test por algoritmo (Resumen Wilcoxon)...")
        
        # Ajusta estas rutas a tu estructura actual
        ruta_wilcoxon = './Resultados/Test/KP/wilconxon/'
        ruta_salida_base = './Resultados/Test/KP/test_per_algoritmo/'
        
        # Nos aseguramos de que la carpeta exista
        os.makedirs(ruta_salida_base, exist_ok=True)
        
        for algoritmo in algoritmos:
            # 1. Definimos dinámicamente los algoritmos a comparar (excluyendo el actual)
            otros_algoritmos = [a for a in algoritmos if a != algoritmo]
            columnas = ['instance'] + otros_algoritmos + ['contador']
            
            # Buffer para guardar las filas. ¡Mucho más rápido y seguro que usar .loc!
            filas_resumen = []
            
            for archivo in os.listdir(ruta_wilcoxon):
                if not archivo.endswith('.csv'):
                    continue
                
                df = pd.read_csv(os.path.join(ruta_wilcoxon, archivo))
                
                # Extraemos el nombre de la instancia asumiendo el formato "test_wilcoxon_Instancia.csv"
                # Esto es más robusto que jugar con índices fijos de split()
                nombre_instancia = archivo.replace('test_wilcoxon_', '').replace('.csv', '')
                
                # Buscamos la fila correspondiente a nuestro algoritmo base
                fila = df.loc[df['vs'] == algoritmo]
                
                # Si el algoritmo no participó en esta instancia, saltamos
                if fila.empty:
                    continue
                    
                # Convertimos la fila a Serie para acceder a los valores más fácilmente
                fila = fila.iloc[0]
                
                # Preparamos los datos de la fila actual
                data_fila = [nombre_instancia]
                contador = 0
                
                for a in otros_algoritmos:
                    p_val = fila[a]
                    data_fila.append(p_val)
                    
                    # Validación segura: Intentamos convertir a float. 
                    # Si es 'N/A' o un texto, simplemente lo ignoramos y no suma al contador.
                    try:
                        if float(p_val) < 0.05:
                            contador += 1
                    except (ValueError, TypeError):
                        pass
                        
                data_fila.append(contador)
                
                # Agregamos la lista a nuestro buffer de filas
                filas_resumen.append(data_fila)
                
            # 2. Creamos el DataFrame de una sola vez
            resumen = pd.DataFrame(filas_resumen, columns=columnas)
            
            # 3. Guardar los resultados
            ruta_salida = os.path.join(ruta_salida_base, f'{algoritmo}.csv')
            resumen.to_csv(ruta_salida, index=False)
            print(f"[INFO] ✅ Resumen de victorias generado para {algoritmo}...")


    if tabla_doble_entrada:
        print(f"\n[INFO] Generando tablas de doble entrada (Resumen y Porcentajes)...")
        
        # Asegúrate de que las rutas coincidan con tu estructura
        ruta_per_algoritmo = './Resultados/Test/KP/test_per_algoritmo/'
        ruta_salida = './Resultados/Test/KP/'
        
        # 1. Columnas dinámicas usando tu lista 'algoritmos' ya existente
        columnas_matriz = ['vs'] + algoritmos
        
        # 2. Buffers para construir las tablas de un solo golpe al final
        filas_resumen = []
        filas_porcentaje = []
        
        # 3. Iteramos por cada algoritmo (Esto construirá las FILAS de la matriz)
        for algoritmo in algoritmos:
            ruta_csv = os.path.join(ruta_per_algoritmo, f"{algoritmo}.csv")
            
            # Validación de seguridad por si algún algoritmo no generó archivo
            if not os.path.exists(ruta_csv):
                print(f"⚠️ Advertencia: No se encontró {ruta_csv}. Saltando...")
                continue
                
            df = pd.read_csv(ruta_csv)
            
            # Iniciamos las listas de la fila actual
            fila_r = [algoritmo]
            fila_p = [algoritmo]
            
            # 4. Comparamos contra el resto (Esto construye las COLUMNAS)
            for algoritmo_2 in algoritmos:
                if algoritmo == algoritmo_2:
                    # Diagonal de la matriz
                    fila_r.append('X')
                    fila_p.append('X')
                else:
                    # Validar que la columna a comparar realmente exista en el CSV
                    if algoritmo_2 in df.columns:
                        # TRUCO DE SEGURIDAD: Convertir a numérico forzando errores a NaN
                        # Esto evita que el código explote si hay un 'N/A' atravesado en la columna
                        columna_numerica = pd.to_numeric(df[algoritmo_2], errors='coerce')
                        
                        total_filas = len(df) # El total de instancias
                        
                        if total_filas > 0:
                            # Contamos cuántos p-valores son menores o iguales a 0.05
                            casos_inferiores = (columna_numerica <= 0.05).sum()
                            porcentaje = (casos_inferiores / total_filas) * 100
                            
                            # Formateamos y agregamos
                            fila_r.append(f"{casos_inferiores}/{total_filas}")
                            fila_p.append(f"{porcentaje:.2f}%")
                        else:
                            fila_r.append("0/0")
                            fila_p.append("0.00%")
                    else:
                        # Si la columna no existe por alguna razón
                        fila_r.append('N/A')
                        fila_p.append('N/A')
            
            # Insertamos las filas terminadas en nuestros buffers
            filas_resumen.append(fila_r)
            filas_porcentaje.append(fila_p)
            
        # 5. Generación de los DataFrames
        resumen = pd.DataFrame(filas_resumen, columns=columnas_matriz)
        resumen_porcentaje = pd.DataFrame(filas_porcentaje, columns=columnas_matriz)
        
        print("\n--- Tabla Doble Entrada (Fracciones) ---")
        print(resumen)
        print("\n--- Tabla Doble Entrada (Porcentajes) ---")
        print(resumen_porcentaje)
        
        # 6. Guardado final
        resumen.to_csv(os.path.join(ruta_salida, 'tabla_doble_entrada_resumen.csv'), index=False)
        resumen_porcentaje.to_csv(os.path.join(ruta_salida, 'tabla_doble_entrada_porcentaje.csv'), index=False)
        
        print(f"[INFO] ✅ Tablas de doble entrada guardadas en: {ruta_salida}")
        print(f"[INFO] ✅ Generación FINALIZADA")

def graficar_exploracion_explotacion(iteraciones, xpl, xpt, titulo, mh, binarizacion):
    """
    Genera y guarda un gráfico de líneas de XPL y XPT sin título.
    Acepta DataFrames como entrada para las series.
    """
    # 1. Convertir DataFrames a arreglos 1D para evitar errores de Matplotlib y de cálculo de medias
    val_iter = iteraciones.values.flatten() if isinstance(iteraciones, pd.DataFrame) else iteraciones
    val_xpl = xpl.values.flatten() if isinstance(xpl, pd.DataFrame) else xpl
    val_xpt = xpt.values.flatten() if isinstance(xpt, pd.DataFrame) else xpt
    
    # 2. Construir la ruta de salida estricta
    output_dir = './Resultados/graficos/KP/'
    os.makedirs(output_dir, exist_ok=True) # Crea las carpetas si no existen
    
    nombre_archivo = f'percentage_{titulo}_{mh}_{binarizacion}.png'
    path_porcentaje = os.path.join(output_dir, nombre_archivo)
    
    # 3. Inicializar el lienzo
    fig, axPER = plt.subplots(figsize=(8, 5))
    
    # 4. Graficar las series (Usando los valores aplanados)
    axPER.plot(val_iter, val_xpl, color="r", label=rf"$\overline{{XPL}}$: {np.round(np.mean(val_xpl), 2)}%")
    axPER.plot(val_iter, val_xpt, color="b", label=rf"$\overline{{XPT}}$: {np.round(np.mean(val_xpt), 2)}%")
    
    # 5. Personalización (Se elimina el título según la instrucción)
    axPER.set_ylabel("Percentage")
    axPER.set_xlabel("Iteration")
    
    # 6. Configurar leyenda y márgenes
    axPER.legend(loc='center right')
    plt.tight_layout()
    
    # 7. Guardar y cerrar
    plt.savefig(path_porcentaje, dpi=300)
    plt.close()
    
    print(f"✅ Gráfico guardado exitosamente en: {path_porcentaje}")

def graficar_dos_series(eje_x, serie_1, serie_2, 
                        nombre_serie1="Algoritmo 1", 
                        nombre_serie2="Algoritmo 2", 
                        etiqueta_x="Iteraciones", 
                        etiqueta_y="Fitness",
                        ruta_guardado=None):
    """
    Genera un gráfico de líneas comparando dos series de datos.
    Los parámetros de entrada (eje_x, serie_1, serie_2) pueden ser columnas de un DataFrame (Series) o listas.
    """
    
    # 1. Configurar el tamaño del lienzo
    plt.figure(figsize=(8, 6))

    # 2. Graficar las series
    plt.plot(eje_x, serie_1, label=nombre_serie1, color='blue', linestyle='-')
    plt.plot(eje_x, serie_2, label=nombre_serie2, color='red', linestyle='--')

    # 3. Personalización de textos
    plt.xlabel(etiqueta_x, fontsize=12)
    plt.ylabel(etiqueta_y, fontsize=12)

    # 4. Activar leyenda y cuadrícula ('best' busca automáticamente el mejor rincón vacío)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)

    # 5. Ajustar márgenes
    plt.tight_layout()

    # 6. Guardar si se proporciona una ruta
    if ruta_guardado:
        plt.savefig(ruta_guardado, dpi=300)
        print(f"✅ Gráfico guardado exitosamente en: {ruta_guardado}")

    # 7. Mostrar el gráfico y limpiar la memoria
    plt.close()

def graficos_best():
    binarizacion_en_analisis = ['STD', 'STD_CIRCLE']
    instancias = bd.obtenerInstanciasEjecutadas("KP")
    mhs = bd.obtenerMHBD()    
    
    for instancia in instancias:
        titulo = instancia[1].split('_')[1] + "_" + instancia[1].split('_')[2]
        if '1000' in titulo or '2000' in titulo:
            for mh in mhs:
                experimento_base = f'{mh[0]} S2-{binarizacion_en_analisis[0]}'
                experimento_caotico = f'{mh[0]} S2-{binarizacion_en_analisis[1]}'
                best_caotico = bd.obtenerBestArchivoKP(instancia[1], mh[0], experimento_caotico)
                buffer_memoria_caotico = io.BytesIO(best_caotico[0][1])
                df_caotico = pd.read_csv(buffer_memoria_caotico, encoding='utf-8')
                
                
                best_base = bd.obtenerBestArchivoKP(instancia[1], mh[0], experimento_base)
                buffer_memoria_base = io.BytesIO(best_base[0][1])
                df_base = pd.read_csv(buffer_memoria_base, encoding='utf-8')
                print(f"\n[INFO] Generando gráfico de convergencia para la instancia {titulo} y MH {mh[0]}...")
                # Llamada avanzada (personalizando todo y guardando la imagen):
                graficar_dos_series(
                    eje_x=df_base['iter'], 
                    serie_1=df_base['fitness'], 
                    serie_2=df_caotico['fitness'],
                    nombre_serie1=binarizacion_en_analisis[0],
                    nombre_serie2=binarizacion_en_analisis[1],
                    etiqueta_x="Iterations",
                    etiqueta_y="Fitness",
                    ruta_guardado=f"./Resultados/best/KP/convergencia_{titulo}_{mh[0]}.png"
                )
                
                print(f"[INFO] Generando gráfico de exploración y explotación para la instancia {titulo}, MH {mh[0]}, binarización {binarizacion_en_analisis[0]}...")
                # Supongamos que df_datos es tu DataFrame con las iteraciones y porcentajes
                graficar_exploracion_explotacion(
                    iteraciones=df_base['iter'],   # Pasando un DataFrame
                    xpl=df_base['XPL'],                # Pasando un DataFrame
                    xpt=df_base['XPT'],                # Pasando un DataFrame
                    titulo=titulo,
                    mh=mh[0],  # mh es una lista/tupla, mh[0] será "SCA"
                    binarizacion=binarizacion_en_analisis[0]
                )
                
                print(f"[INFO] Generando gráfico de exploración y explotación para la instancia {titulo}, MH {mh[0]}, binarización {binarizacion_en_analisis[1]}...")
                # Supongamos que df_datos es tu DataFrame con las iteraciones y porcentajes
                graficar_exploracion_explotacion(
                    iteraciones=df_caotico['iter'],   # Pasando un DataFrame
                    xpl=df_caotico['XPL'],                # Pasando un DataFrame
                    xpt=df_caotico['XPT'],                # Pasando un DataFrame
                    titulo=titulo,
                    mh=mh[0],  # mh es una lista/tupla, mh[0] será "SCA"
                    binarizacion=binarizacion_en_analisis[1]
                )
            
def generar_analitica_familiar():
    configuraciones = ['STD', 'STD_LOG', 'STD_PIECE', 'STD_SINE', 'STD_SINGER', 'STD_SINU', 'STD_TENT', 'STD_CIRCLE']
    # columnas =  ['Instance','STD', '', '', 'STD_LOG', '', '', 'STD_PIECE', '', '', 'STD_SINE', '', '', 'STD_SINGER', '', '', 'STD_SINU', '', '', 'STD_TENT', '', '', 'STD_CIRCLE', '', '']
    # encabezado =  ['','best', 'avg', 'std', 'best', 'avg', 'std','best', 'avg', 'std','best', 'avg', 'std','best', 'avg', 'std','best', 'avg', 'std','best', 'avg', 'std','best', 'avg', 'std']
    columnas =  ['Instance','Opt', 'STD', '', 'STD_LOG', '', 'STD_PIECE', '', 'STD_SINE', '', 'STD_SINGER', '', 'STD_SINU', '', 'STD_TENT', '', 'STD_CIRCLE', '']
    encabezado =  ['', '','best', 'avg', 'best', 'avg', 'best', 'avg', 'best', 'avg', 'best', 'avg', 'best', 'avg', 'best', 'avg','best', 'avg']
    mhs = bd.obtenerMHBD()
    for mh in mhs:
        # 1. Creas tu buffer
        filas_acumuladas = [encabezado]
        archivos = os.listdir(f'./Resultados/data/KP/fitness/{mh[0]}/')
        for archivo in archivos:
            fila_dinamica = [archivo.split(".")[0].split('_')[1] + "_" + archivo.split(".")[0].split('_')[2]]
            df = pd.read_csv(f'./Resultados/data/KP/fitness/{mh[0]}/{archivo}')
            optimo = bd.obtenerOptimoInstancia(archivo.split(".")[0])
            fila_dinamica.append(optimo[0][0])
            for configuracion in configuraciones:
                df_filtrado = df[df['MH'] == configuracion]
                if not df_filtrado.empty:
                    best = df_filtrado['Data'].max()
                    promedio = df_filtrado['Data'].mean()
                    # std = df_filtrado['Data'].std()
                    fila_dinamica.append(best)
                    fila_dinamica.append(np.round(promedio, 2))
                    # fila_dinamica.append(np.round(std, 2))
            filas_acumuladas.append(fila_dinamica)
        df_final = pd.DataFrame(filas_acumuladas, columns=columnas)
        df_final.to_csv(f'./Resultados/resumen/KP/fitness/{mh[0]}.csv', index=False)
        print(f"✅ Archivo fitness_KP_{mh[0]}.csv generado exitosamente.")
            # buffer_columnas[nombre_algoritmo] = resultados
                    
def generarRPD():
    instancias = bd.obtenerInstanciasEjecutadas("KP")
    mhs = bd.obtenerMHBD()    
    # Nombres de las columnas que usarás al final
    nombres_columnas = ['BS']
    for instancia in instancias:
        titulo = instancia[1].split('_')[1] + "_" + instancia[1].split('_')[2]
        nombres_columnas.append(titulo)
    print(nombres_columnas)
    nombres_columnas.append('avg')
    for mh in mhs:
        experimentos = bd.obtenerExperimentosbyMH(mh[0])
        # 1. Creas tu buffer
        filas_acumuladas = []
        for experimento in experimentos:
            # 1. Creas tu buffer
            fila_dinamica = [experimento[0].split("-")[1]]  # La primera columna es el nombre del MH
            promedio = []
            print(f"[INFO] Procesando RPD experimento {experimento[0]}...") 
            for instancia in instancias:
                rpd = bd.obtenerRPD(instancia[1], mh[0], experimento[0])
                fila_dinamica.append(rpd[0][0])
                promedio.append(rpd[0][0])
            fila_dinamica.append(np.round(np.mean(promedio), 2))
            filas_acumuladas.append(fila_dinamica)
        # 2. Conviertes a DataFrame pasándole el nombre de las columnas
        df_final = pd.DataFrame(filas_acumuladas, columns=nombres_columnas)
        df_final.to_csv(f'./Resultados/resumen/KP/RPD/{mh[0]}.csv', index=False)
        print(f"✅ Archivo RPD_KP_{mh[0]}.csv generado exitosamente.")

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
    titulo = nombre_instancia.split('_')[1] + "_" + nombre_instancia.split('_')[2]
    plt.title(f'Boxplot - Instance {titulo}', fontsize=16, fontweight='bold')
    plt.xlabel('Binarization', fontsize=12)
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
    instancias = bd.obtenerInstanciasEjecutadas("KP")
    print(f"[INFO] Analizando {len(instancias)} instancias de KP...")
    
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
                
                ruta_fitness = f'./Resultados/data/KP/fitness/{mh[0]}/{instancia[1]}.csv'
                ruta_tiempos = f'./Resultados/data/KP/tiempos/{mh[0]}/{instancia[1]}.csv'
                
                # verificacion archivo de fitness
                conf_fitness = gestionar_archivo_csv(ruta_fitness)
                # verificacion archivo de tiempos
                conf_tiempos = gestionar_archivo_csv(ruta_tiempos)
                
                datos_fitness = []
                datos_tiempos = []
                datos_mh = []
                nombre_mh = experimento[0].split("-")[1]
                
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
        instancias = bd.obtenerInstanciasEjecutadas("KP")
        for instancia in instancias:
            mhs = bd.obtenerMHEjecutadas(instancia[1])
            for mh in mhs:
                ruta_entrada = f'./Resultados/data/KP/fitness/{mh[0]}/{instancia[1]}.csv'
                ruta_salida_box = f'./Resultados/boxplot/KP/{mh[0]}/'
                
                generar_graficos_separados(ruta_entrada, ruta_salida_box, instancia[1])
    if GENERAR_RPD:
        generarRPD()
        
    if GENERAR_ANALITICA_BEST_FAMILIA:
        generar_analitica_familiar()
        
    if GRAFICOS_BEST_FAMILIA:
        graficos_best()
        
    if TEST_ESTADISTICO:
        test_estadistico()

    # print(os.listdir('./Resultados/graficos/KP/'))  # Verifica que los archivos se hayan guardado correctamente

    print("[INFO] Análisis KP completado con éxito.")
    print("-" * 50)
