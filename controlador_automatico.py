import subprocess
import time
import sys
from Util.log import log_final, log_fecha_hora

from BD.sqlite import BD

def hay_experimentos_pendientes() -> bool:
    bd = BD()
    return bd.obtenerCantidadExperimentosPendientes()

def ejecutar_script(script_name: str) -> bool:
    """
    Ejecuta un script de Python y espera a que termine.
    Devuelve True si tuvo exito, False si fallo.
    """
    command = ["python3.11", script_name]
    print(f"\n--- Ejecutando: {' '.join(command)} ---")
    try:
        # Usamos subprocess.run() porque esta Si espera a que termine
        subprocess.run(command, check=True, text=True)
        print(f"--- Completado: {script_name} ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {script_name} fallo (codigo de salida {e.returncode}).", file=sys.stderr)
        print(f"Salida de error: {e.stderr}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"ERROR: No se encontro el script '{script_name}' o 'python3.11'.", file=sys.stderr)
        return False
        
def esperar_que_terminen_experimentos():
    """
    Bucle de sondeo (polling) que comprueba la DB hasta que
    no queden experimentos 'pendientes'.
    """
    print("\n--- Monitoreando la finalizacion de los experimentos (en tmux)... ---")
    PAUSA_ENTRE_CHECKS_SEGUNDOS = 15 # Comprobar cada 1 minuto

    while hay_experimentos_pendientes():
        print(f"INFO: El trabajo (main.py) sigue en ejecucion. Proximo chequeo en {PAUSA_ENTRE_CHECKS_SEGUNDOS}s...")
        time.sleep(PAUSA_ENTRE_CHECKS_SEGUNDOS)
    
    print("--- Trabajo completado, Todos los experimentos estan 'terminados'. ---")
    
    
    
    
    
def main_loop():
    """
    El bucle principal que ejecuta el flujo de trabajo.
    """
    ciclo_num = 1
    start_time_experimentacion = time.time()  # Registrar el tiempo inicial
    log_fecha_hora("Inicio de la experimentacion")
    while True:
        
        print(f"\n======================================")
        log_fecha_hora(f"INICIANDO CICLO DE EXPERIMENTACION #{ciclo_num}")
        print(f"======================================")

        # 0. Comprobar si hay trabajo que hacer
        if not hay_experimentos_pendientes():
            print("Exito total, No hay mas outliers ni experimentos pendientes.")
            print("Proceso finalizado.")
            break # Fin del bucle principal

        # 1. Levantar los workers (tmux)
        # Esto es rapido, solo lanza los comandos
        if not ejecutar_script("levantarCMD.py"):
            print("Error fatal: 'levantarCMD.py' fallo. Abortando.", file=sys.stderr)
            break

        # 2. ESPERAR a que los workers terminen
        # Esta es la parte clave. Sondeamos la DB.
        esperar_que_terminen_experimentos()
        
        # 3. Analizar resultados
        # Esto es sincrono, esperamos a que termine
        if not ejecutar_script("chequeoOutlier.py"):
            print("Error fatal: 'chequeoOutlier.py' fallo. Abortando.", file=sys.stderr)
            break

        print(f"--- Fin del ciclo #{ciclo_num}. Reiniciando para comprobar si hay nuevos outliers. ---")
        ciclo_num += 1
        time.sleep(5) # Una pequena pausa antes de reiniciar el bucle
        
    end_time_experimentacion = time.time()
    total_time_exeperimentacion = end_time_experimentacion - start_time_experimentacion
    log_final(total_time_exeperimentacion)

if __name__ == "__main__":
    main_loop()