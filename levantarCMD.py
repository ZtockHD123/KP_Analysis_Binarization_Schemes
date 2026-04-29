#import subprocess
#import os
#import sys
#import platform

#def abrir_cmds_ejecutar_main(num_cmds):
#    # Obtener la ruta actual del script
#    ruta_actual = os.path.dirname(os.path.abspath(__file__))
#    programa = 'main.py'
#    
#    # Verificar si main.py existe en la ruta actual
#    main_py_path = os.path.join(ruta_actual, programa)
#    if not os.path.isfile(main_py_path):
#        print(f"No se encontró {programa} en la ruta: {ruta_actual}")
#        return
#    
#    # Verificar si Python está disponible en el sistema
#    python_executable = sys.executable
#    if not os.path.isfile(python_executable):
#        print(f"No se encontró el ejecutable de Python en la ruta: {python_executable}")
#        return

#    # Determinar el sistema operativo
#    sistema = platform.system()
#    
#    if sistema == "Windows":
#        comando_base = f'start cmd /K "cd /d {ruta_actual} && {python_executable} {programa}"'
#    elif sistema == "Linux":
#        comando_base = f'gnome-terminal -- bash -c "cd {ruta_actual} && {python_executable} {programa}; exec bash"'
#        if subprocess.call("which gnome-terminal", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) != 0:
#            comando_base = f'xterm -hold -e "{python_executable} {programa}"'
#    else:
#        print(f"El sistema operativo {sistema} no es compatible con este script.")
#        return
#    
#    for _ in range(num_cmds):
#        try:
#            subprocess.Popen(comando_base, shell=True)
#        except Exception as e:
#            print(f"Error al intentar abrir la ventana de terminal: {e}")
#            return
#
#    print(f"Se han abierto {num_cmds} terminales ejecutando {programa} en la ruta: {ruta_actual}")
#
#if __name__ == "__main__":
#    # Definir la cantidad de terminales a levantar
#    num_cmds = 10
#    abrir_cmds_ejecutar_main(num_cmds)


import subprocess
import time

# --- Configuración ---
SESSION_NAME = "experimentos_main"
COMMAND_TO_RUN = "python3.11 main.py"  # El único comando a ejecutar
NUM_WINDOWS = 16                     # Cuántas copias quieres lanzar
# ---------------------

# --- NUEVA SECCIÓN DE LIMPIEZA ---
print(f"Intentando limpiar cualquier sesión antigua llamada '{SESSION_NAME}'...")
try:
    # Usamos check=True para que falle si el comando tmux no existe
    # Usamos capture_output=True y text=True para capturar los mensajes
    # (stderr=subprocess.DEVNULL oculta el error si la sesión no existe)
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        stderr=subprocess.DEVNULL,  
        check=True
    )
    print(f"Sesión antigua '{SESSION_NAME}' eliminada.")
except subprocess.CalledProcessError:
    # Esto es normal si la sesión no existía.
    print(f"No se encontró sesión antigua '{SESSION_NAME}', ¡perfecto!")
except FileNotFoundError:
    print("Error: No se encontró el comando 'tmux'. ¿Está instalado?")
    exit()
# ---------------------------------

print(f"Lanzando sesión de tmux: {SESSION_NAME}")

# Comando completo que se ejecutará en cada terminal
# Añadimos '; bash' para que la terminal no se cierre al terminar el script.
final_command = f"{COMMAND_TO_RUN} ; bash"

# 1. Crear la sesión detached y la primera ventana (índice 1)
try:
    window_name = "exec_1"
    # Creamos la sesión y ejecutamos el primer comando en ella
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION_NAME, 
         "-n", window_name, final_command],
        check=True
    )
    print(f"Sesión creada y lanzado script en ventana '{window_name}'.")

except subprocess.CalledProcessError as e:
    print(f"Error al crear sesión. ¿Quizás '{SESSION_NAME}' ya existe? Error: {e}")
    exit()

# 2. Crear el resto de las ventanas (de 2 a NUM_WINDOWS)
for i in range(2, NUM_WINDOWS + 1):
    window_name = f"exec_{i}"
    
    # Crear una nueva ventana y ejecutar el comando en ella
    subprocess.run([
        "tmux", "new-window", "-t", SESSION_NAME,
        "-n", window_name,
        final_command
    ])
    print(f"Lanzado script en nueva ventana '{window_name}'.")

print("\n¡Todo listo!")
print(f"Hay {NUM_WINDOWS} copias de '{COMMAND_TO_RUN}' corriendo.")
print(f"Puedes adjuntarte con: tmux attach -t {SESSION_NAME}")