import numpy as np
import matplotlib.pyplot as plt

def graficar_bifurcacion(mu_min=2.5, mu_max=4.0, resolucion_mu=10000):
    """
    Genera el diagrama de bifurcación para el Mapa Logístico.
    """
    print(f"[INFO] Calculando diagrama de bifurcación para mu entre {mu_min} y {mu_max}...")
    
    # 1. Configuración de parámetros
    # iteraciones_descarte: Se iteran varias veces para dejar que la ecuación se "estabilice" o caiga en su atractor
    iteraciones_descarte = 1000 
    # iteraciones_guardadas: Cuántos puntos vamos a dibujar por cada valor de mu
    iteraciones_guardadas = 5000

    # Crear el eje X (nuestros miles de valores para mu)
    mu = np.linspace(mu_min, mu_max, resolucion_mu)
    
    # Crear el estado inicial x_0 (por ejemplo, 0.5 para todos)
    x = np.ones(resolucion_mu) * 0.5

    # 2. Fase de Estabilización (Transitorios)
    # Calculamos la ecuación muchas veces sin guardar los resultados, 
    # solo para que los valores converjan a su ciclo natural.
    for i in range(iteraciones_descarte):
        x = mu * x * (1 - x)

    # 3. Preparar el lienzo
    plt.figure(figsize=(12, 8))

    # 4. Fase de Dibujo
    # Ahora sí, iteramos y dibujamos los puntos resultantes
    for i in range(iteraciones_guardadas):
        x = mu * x * (1 - x)
        
        # Dibujamos usando scatter. 
        # TRUCO VISUAL: s (tamaño) diminuto y alpha (transparencia) baja
        # Esto genera ese efecto de "tinta" donde el caos es denso y oscuro.
        plt.scatter(mu, x, color='blue', s=0.01, alpha=0.1, edgecolors='none')

    # 5. Personalización del gráfico
    plt.xlabel(r'Control parameter $c$', fontsize=14)
    plt.ylabel('Stabilized values of $r$', fontsize=14)
    
    plt.xlim(mu_min, mu_max)
    plt.ylim(0, 1)
    
    # Quitamos bordes arriba y a la derecha para un estilo más limpio
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # 6. Guardar y mostrar
    # plt.savefig('bifurcation_diagram.png', dpi=600) # dpi alto para gráficos fractales
    print("[INFO] ¡Cálculo terminado! Renderizando imagen...")
    plt.savefig('bifurcation_diagram.png', dpi=600) # Guardar con alta resolución

# Ejecutar la función
# graficar_bifurcacion(mu_min=2, mu_max=4.0)

from scipy.stats import mannwhitneyu

GWO_ELIT_41 = [433.0,433.0,437.0,433.0,433.0,433.0,433.0,433.0,433.0,437.0,433.0,437.0,433.0,433.0,433.0,433.0,433.0,433.0,433.0,433.0,437.0,433.0,433.0,433.0,437.0,433.0,433.0,433.0,433.0,433.0,433.0]
GWO_ELIT_SINU_41 = [436.0,435.0,436.0,436.0,436.0,435.0,436.0,435.0,436.0,435.0,437.0,435.0,436.0,436.0,437.0,436.0,435.0,435.0,435.0,435.0,435.0,436.0,436.0,436.0,436.0,436.0,435.0,436.0,435.0,436.0,436.0]

GWO_ELIT_51 = [267.0,268.0,267.0,267.0,267.0,268.0,267.0,268.0,267.0,267.0,268.0,268.0,268.0,267.0,267.0,267.0,267.0,267.0,269.0,267.0,267.0,267.0,267.0,267.0,267.0,267.0,269.0,268.0,268.0,268.0,267.0]
GWO_ELIT_SINU_51 = [269.0,269.0,269.0,269.0,269.0,269.0,269.0,269.0,269.0,268.0,270.0,269.0,269.0,270.0,269.0,269.0,269.0,269.0,269.0,269.0,269.0,268.0,269.0,269.0,268.0,269.0,269.0,270.0,269.0,268.0,269.0]

GWO_ELIT_61 = [145.0,145.0,145.0,145.0,145.0,144.0,141.0,144.0,141.0,141.0,144.0,141.0,141.0,141.0,145.0,145.0,145.0,141.0,141.0,141.0,144.0,145.0,141.0,145.0,145.0,141.0,141.0,141.0,141.0,141.0,141.0]
GWO_ELIT_SINU_61 = [142.0,142.0,142.0,142.0,142.0,141.0,142.0,142.0,141.0,141.0,142.0,142.0,141.0,142.0,142.0,142.0,142.0,141.0,142.0,141.0,142.0,141.0,141.0,142.0,141.0,142.0,142.0,141.0,142.0,141.0,141.0]

GWO_ELIT_a1 = [257.0,257.0,257.0,257.0,257.0,257.0,257.0,257.0,257.0,257.0,258.0,257.0,257.0,257.0,257.0,257.0,258.0,257.0,257.0,257.0,257.0,257.0,257.0,257.0,257.0,257.0,257.0,257.0,257.0,257.0,257.0]
GWO_ELIT_SINU_a1 = [260.0,260.0,263.0,263.0,261.0,261.0,261.0,261.0,261.0,263.0,259.0,261.0,261.0,261.0,261.0,261.0,259.0,261.0,262.0,260.0,262.0,262.0,261.0,261.0,260.0,263.0,263.0,261.0,262.0,260.0,261.0]

GWO_ELIT_b1 = [71.0,71.0,71.0,71.0,69.0,69.0,71.0,72.0,71.0,71.0,70.0,71.0,69.0,71.0,72.0,71.0,71.0,71.0,70.0,71.0,69.0,71.0,72.0,71.0,71.0,71.0,69.0,70.0,71.0,69.0,69.0]
GWO_ELIT_SINU_b1 = [69.0,70.0,71.0,71.0,70.0,70.0,70.0,70.0,70.0,70.0,72.0,70.0,70.0,70.0,70.0,70.0,70.0,71.0,70.0,70.0,71.0,70.0,70.0,69.0,71.0,70.0,69.0,70.0,71.0,70.0,69.0]

GWO_ELIT_c1 = [233.0,233.0,233.0,234.0,233.0,235.0,234.0,234.0,232.0,234.0,235.0,232.0,235.0,234.0,234.0,234.0,231.0,234.0,232.0,234.0,233.0,234.0,233.0,233.0,233.0,234.0,233.0,233.0,234.0,234.0,234.0]
GWO_ELIT_SINU_c1 = [238.0,239.0,235.0,237.0,238.0,237.0,236.0,237.0,237.0,238.0,239.0,237.0,237.0,238.0,236.0,238.0,237.0,236.0,237.0,236.0,238.0,238.0,238.0,237.0,238.0,238.0,238.0,237.0,235.0,238.0,236.0]

GWO_ELIT_d1 = [61.0,61.0,61.0,63.0,62.0,63.0,61.0,61.0,64.0,60.0,61.0,61.0,62.0,62.0,61.0,62.0,63.0,62.0,61.0,62.0,62.0,63.0,63.0,63.0,61.0,63.0,61.0,62.0,62.0,60.0,60.0]
GWO_ELIT_SINU_d1 = [63.0,63.0,63.0,63.0,63.0,62.0,62.0,62.0,63.0,63.0,62.0,62.0,62.0,62.0,62.0,63.0,63.0,63.0,62.0,63.0,64.0,62.0,63.0,62.0,63.0,63.0,63.0,63.0,62.0,62.0,63.0]

GWO_ELIT_nre1 = [29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0]
GWO_ELIT_SINU_nre1 = [29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0,29.0]

GWO_ELIT_nrf1 = [14.0,14.0,14.0,14.0,15.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,15.0,14.0,15.0,14.0,14.0,14.0,15.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0]
GWO_ELIT_SINU_nrf1 = [14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0,14.0]


stat, p_value = mannwhitneyu(GWO_ELIT_41, GWO_ELIT_SINU_41, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"41 ELIT vs SINU U statistic: {stat}, p-value: {p_value} - {validacion}")

stat, p_value = mannwhitneyu(GWO_ELIT_51, GWO_ELIT_SINU_51, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"51 ELIT vs SINU U statistic: {stat}, p-value: {p_value} - {validacion}")

stat, p_value = mannwhitneyu(GWO_ELIT_61, GWO_ELIT_SINU_61, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"61 ELIT vs SINU U statistic: {stat}, p-value: {p_value} - {validacion}")

stat, p_value = mannwhitneyu(GWO_ELIT_a1, GWO_ELIT_SINU_a1, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"a1 ELIT vs SINU U statistic: {stat}, p-value: {p_value} - {validacion}")

stat, p_value = mannwhitneyu(GWO_ELIT_b1, GWO_ELIT_SINU_b1, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"b1 ELIT vs SINU U statistic: {stat}, p-value: {p_value} - {validacion}")

stat, p_value = mannwhitneyu(GWO_ELIT_c1, GWO_ELIT_SINU_c1, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"c1 ELIT vs SINU U statistic: {stat}, p-value: {p_value} - {validacion}")

stat, p_value = mannwhitneyu(GWO_ELIT_d1, GWO_ELIT_SINU_d1, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"d1 ELIT vs SINU U statistic: {stat}, p-value: {p_value} - {validacion}")

stat, p_value = mannwhitneyu(GWO_ELIT_nre1, GWO_ELIT_SINU_nre1, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"nre1 ELIT vs SINU U statistic: {stat}, p-value: {p_value} - {validacion}")

stat, p_value = mannwhitneyu(GWO_ELIT_nrf1, GWO_ELIT_SINU_nrf1, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"nrf1 ELIT vs SINU U statistic: {stat}, p-value: {p_value} - {validacion}")


print("------------------------------------------------------------------------")

stat, p_value = mannwhitneyu(GWO_ELIT_SINU_41, GWO_ELIT_41, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"41 SINU vs ELIT U statistic: {stat}, p-value: {p_value} - {validacion}")

stat, p_value = mannwhitneyu(GWO_ELIT_SINU_51, GWO_ELIT_51, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"51 SINU vs ELIT U statistic: {stat}, p-value: {p_value} - {validacion}")

stat, p_value = mannwhitneyu(GWO_ELIT_SINU_61, GWO_ELIT_61, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"61 SINU vs ELIT U statistic: {stat}, p-value: {p_value} - {validacion}")

stat, p_value = mannwhitneyu(GWO_ELIT_SINU_a1, GWO_ELIT_a1, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"a1 SINU vs ELIT U statistic: {stat}, p-value: {p_value} - {validacion}")

stat, p_value = mannwhitneyu(GWO_ELIT_SINU_b1, GWO_ELIT_b1, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"b1 SINU vs ELIT U statistic: {stat}, p-value: {p_value} - {validacion}")

stat, p_value = mannwhitneyu(GWO_ELIT_SINU_c1, GWO_ELIT_c1, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"c1 SINU vs ELIT U statistic: {stat}, p-value: {p_value} - {validacion}")

stat, p_value = mannwhitneyu(GWO_ELIT_SINU_d1, GWO_ELIT_d1, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"d1 SINU vs ELIT U statistic: {stat}, p-value: {p_value} - {validacion}")

stat, p_value = mannwhitneyu(GWO_ELIT_SINU_nre1, GWO_ELIT_nre1, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"nre1 SINU vs ELIT U statistic: {stat}, p-value: {p_value} - {validacion}")

stat, p_value = mannwhitneyu(GWO_ELIT_SINU_nrf1, GWO_ELIT_nrf1, alternative='less')
validacion = "Significant difference" if p_value < 0.05 else "No significant difference"
print(f"nrf1 SINU vs ELIT U statistic: {stat}, p-value: {p_value} - {validacion}")