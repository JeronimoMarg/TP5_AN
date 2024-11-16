import numpy as np
import matplotlib.pyplot as plt

# Definimos las masas estándar para los tres cuerpos
m1, m2, m3 = 1.0, 1.0, 1.0
G = 1  # Constante gravitacional

# Función para calcular el centro de masa del sistema
def center_of_mass(r1, r2, r3):
    return (m1 * r1 + m2 * r2 + m3 * r3) / (m1 + m2 + m3)

# Función para calcular las fuerzas gravitacionales entre los cuerpos
def calc_forces(r1, r2, r3):
    def force(rA, rB, mA, mB):
        rAB = rB - rA
        dist = np.linalg.norm(rAB)
        return G * mA * mB * rAB / dist**3 if dist != 0 else np.zeros(2)
    
    F1 = force(r1, r2, m1, m2) + force(r1, r3, m1, m3)
    F2 = force(r2, r1, m2, m1) + force(r2, r3, m2, m3)
    F3 = force(r3, r1, m3, m1) + force(r3, r2, m3, m2)
    
    return F1, F2, F3

# Método de Euler
def euler_step(r1, r2, r3, v1, v2, v3, dt):
    F1, F2, F3 = calc_forces(r1, r2, r3)
    
    a1, a2, a3 = F1 / m1, F2 / m2, F3 / m3
    r1_new, r2_new, r3_new = r1 + v1 * dt, r2 + v2 * dt, r3 + v3 * dt
    v1_new, v2_new, v3_new = v1 + a1 * dt, v2 + a2 * dt, v3 + a3 * dt
    
    return r1_new, r2_new, r3_new, v1_new, v2_new, v3_new

# Método de Runge-Kutta de cuarto orden (RK4)
def rk4_step(r1, r2, r3, v1, v2, v3, dt):
    def get_accel(r1, r2, r3):
        F1, F2, F3 = calc_forces(r1, r2, r3)
        return F1 / m1, F2 / m2, F3 / m3
    
    # Primera evaluación (k1)
    a1, a2, a3 = get_accel(r1, r2, r3)
    k1_r1, k1_r2, k1_r3 = v1, v2, v3
    k1_v1, k1_v2, k1_v3 = a1, a2, a3

    # Segunda evaluación (k2)
    r1_2, r2_2, r3_2 = r1 + k1_r1 * dt / 2, r2 + k1_r2 * dt / 2, r3 + k1_r3 * dt / 2
    v1_2, v2_2, v3_2 = v1 + k1_v1 * dt / 2, v2 + k1_v2 * dt / 2, v3 + k1_v3 * dt / 2
    a1_2, a2_2, a3_2 = get_accel(r1_2, r2_2, r3_2)
    
    # Tercera evaluación (k3)
    r1_3, r2_3, r3_3 = r1 + v1_2 * dt / 2, r2 + v2_2 * dt / 2, r3 + v3_2 * dt / 2
    v1_3, v2_3, v3_3 = v1 + a1_2 * dt / 2, v2 + a2_2 * dt / 2, v3 + a3_2 * dt / 2
    a1_3, a2_3, a3_3 = get_accel(r1_3, r2_3, r3_3)
    
    # Cuarta evaluación (k4)
    r1_4, r2_4, r3_4 = r1 + v1_3 * dt, r2 + v2_3 * dt, r3 + v3_3 * dt
    v1_4, v2_4, v3_4 = v1 + a1_3 * dt, v2 + a2_3 * dt, v3 + a3_3 * dt
    a1_4, a2_4, a3_4 = get_accel(r1_4, r2_4, r3_4)

    r1_new = r1 + dt * (v1 + 2*v1_2 + 2*v1_3 + v1_4) / 6
    r2_new = r2 + dt * (v2 + 2*v2_2 + 2*v2_3 + v2_4) / 6
    r3_new = r3 + dt * (v3 + 2*v3_2 + 2*v3_3 + v3_4) / 6
    
    v1_new = v1 + dt * (a1 + 2*a1_2 + 2*a1_3 + a1_4) / 6
    v2_new = v2 + dt * (a2 + 2*a2_2 + 2*a2_3 + a2_4) / 6
    v3_new = v3 + dt * (a3 + 2*a3_2 + 2*a3_3 + a3_4) / 6
    
    return r1_new, r2_new, r3_new, v1_new, v2_new, v3_new

# Función para calcular la energía cinética y potencial del sistema
def calc_energies(r1, r2, r3, v1, v2, v3):
    E_kin = 0.5 * (m1 * np.dot(v1, v1) + m2 * np.dot(v2, v2) + m3 * np.dot(v3, v3))
    dist12, dist13, dist23 = np.linalg.norm(r2 - r1), np.linalg.norm(r3 - r1), np.linalg.norm(r3 - r2)
    E_pot = -G * (m1 * m2 / dist12 + m1 * m3 / dist13 + m2 * m3 / dist23)
    return E_kin + E_pot

# Función para calcular la energía acumulada usando Trapecio, Newton-Coates, y Cuadratura de Gauss
import numpy as np

def calc_accumulated_energy(energies, dt):
    energies = np.array(energies)
    
    # Método del Trapecio
    E_trapecio = np.trapezoid(energies, dx=dt)
    
    # Método de Newton-Coates (Simpson)
    if len(energies) % 2 == 0:
        energies = energies[:-1]  # Asegura que haya un número impar de puntos para Simpson
    E_newton_coates = np.sum((dt / 3) * (energies[0:-1:2] + 4 * energies[1::2] + energies[2::2]))
    
    # Cuadratura de Gauss en todo el intervalo con 4 puntos
    gauss_weights = np.array([0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451])
    gauss_points = np.array([-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116])  # Puntos de Gauss en [-1,1]

    # Transformación para todo el intervalo
    a, b = 0, len(energies) * dt  # Integrar sobre todo el intervalo de tiempo
    midpoint = (b + a) / 2
    half_range = (b - a) / 2

    # Convertimos los puntos de Gauss al rango total de integración [a, b]
    gauss_eval_points = midpoint + half_range * gauss_points

    # Convertimos los puntos de Gauss en índices para evaluar en 'energies'
    indices = (gauss_eval_points / dt).astype(int)
    indices = np.clip(indices, 0, len(energies) - 1)  # Asegura que los índices estén dentro del rango

    # Tomamos los valores de energía en estos índices
    gauss_eval_values = energies[indices]

    # Calculamos la integral con los pesos de Gauss-Legendre
    E_gauss = sum(w * e for w, e in zip(gauss_weights, gauss_eval_values)) * half_range

    return E_trapecio, E_newton_coates, E_gauss

    
# Función para simular, graficar y calcular la energía
def simulate_and_plot(method, title, steps=10000, dt=0.001):
    print(f"Simulación iniciada: {title}")
    
    r1, r2, r3 = r1_0, r2_0, r3_0
    v1, v2, v3 = v1_0, v2_0, v3_0

    r1_hist, r2_hist, r3_hist, cm_hist, energy_hist = [r1.copy()], [r2.copy()], [r3.copy()], [], []
    
    # Variables para la conservación de energía
    initial_energy = None
    max_energy_variation = 0

    for step in range(steps):
        r1, r2, r3, v1, v2, v3 = method(r1, r2, r3, v1, v2, v3, dt)
        cm = center_of_mass(r1, r2, r3)
        energy = calc_energies(r1, r2, r3, v1, v2, v3)
        energy_hist.append(energy)
        if(step % 1000 == 0):
            print("La energia actual calculada es de: ", energy)

        
        # Almacenar la energía inicial una vez
        if step == 0:
            initial_energy = energy
        
        # Calcular la variación respecto a la energía inicial
        delta_energy = abs(energy - initial_energy) / abs(initial_energy)
        if delta_energy > max_energy_variation:
            max_energy_variation = delta_energy

        # Guardado de posiciones para graficar
        cm_hist.append(cm.copy())
        r1_hist.append((r1 - cm).copy())
        r2_hist.append((r2 - cm).copy())
        r3_hist.append((r3 - cm).copy())
        
        '''
        # Mensaje de avance
        if step % (steps // 10) == 0:  # Imprimir cada 10% de avance
            print(f"Progreso de {title}: {100 * step // steps}% completado")
        '''

    # Comparación de energías calculadas
    final_energy = energy_hist[-1]
    print(f"{title} - Energía inicial: {initial_energy}")
    print(f"{title} - Energía final: {final_energy}")
    print(f"{title} - Variación máxima de energía durante la simulación: {max_energy_variation * 100:.2f}%")
    if max_energy_variation > 1e-3:
        print(f"{title} - Advertencia: La energía varió significativamente durante la simulación.")
    else:
        print(f"{title} - La energía se conserva bien a lo largo de la simulación.")

    # Calcular energías acumuladas
    E_trapecio, E_newton_coates, E_gauss = calc_accumulated_energy(energy_hist, dt)
    print(f"{title} - Energía acumulada (Trapecio): {E_trapecio}")
    print(f"{title} - Energía acumulada (Newton-Coates): {E_newton_coates}")
    print(f"{title} - Energía acumulada (Cuadratura de Gauss): {E_gauss}")

    # Graficar las trayectorias centradas en la masa
    plt.figure(figsize=(6, 6))
    plt.plot(np.array(r1_hist)[:, 0], np.array(r1_hist)[:, 1], label="Cuerpo 1")
    plt.plot(np.array(r2_hist)[:, 0], np.array(r2_hist)[:, 1], label="Cuerpo 2")
    plt.plot(np.array(r3_hist)[:, 0], np.array(r3_hist)[:, 1], label="Cuerpo 3")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

# Configuración para un sistema estable
r1_0_stable = np.array([0.97000436, -0.24308753])
r2_0_stable = np.array([-0.97000436, 0.24308753])
r3_0_stable = np.array([0.0, 0.0])

v1_0_stable = np.array([0.466203685, 0.43236573])
v2_0_stable = np.array([0.466203685, 0.43236573])
v3_0_stable = np.array([-0.93240737, -0.86473146])

# Configuración para un sistema caótico
r1_0_chaotic = np.array([1.0, 0.5])
r2_0_chaotic = np.array([-1.0, -0.5])
r3_0_chaotic = np.array([0.5, -1.0])

v1_0_chaotic = np.array([0.6, -0.4])
v2_0_chaotic = np.array([-0.6, 0.4])
v3_0_chaotic = np.array([0.0, 0.8])

# Ejecución para el sistema estable y caótico
print("\nSistema Estable con Método de Euler:")
r1_0, r2_0, r3_0 = r1_0_stable, r2_0_stable, r3_0_stable
v1_0, v2_0, v3_0 = v1_0_stable, v2_0_stable, v3_0_stable
simulate_and_plot(euler_step, "Sistema Estable - Euler")

print("\nSistema Caótico con Método de Euler:")
r1_0, r2_0, r3_0 = r1_0_chaotic, r2_0_chaotic, r3_0_chaotic
v1_0, v2_0, v3_0 = v1_0_chaotic, v2_0_chaotic, v3_0_chaotic
simulate_and_plot(euler_step, "Sistema Caótico - Euler")

print("\nSistema Estable con Método de Runge-Kutta 4:")
r1_0, r2_0, r3_0 = r1_0_stable, r2_0_stable, r3_0_stable
v1_0, v2_0, v3_0 = v1_0_stable, v2_0_stable, v3_0_stable
simulate_and_plot(rk4_step, "Sistema Estable - RK4")

print("\nSistema Caótico con Método de Runge-Kutta 4:")
r1_0, r2_0, r3_0 = r1_0_chaotic, r2_0_chaotic, r3_0_chaotic
v1_0, v2_0, v3_0 = v1_0_chaotic, v2_0_chaotic, v3_0_chaotic
simulate_and_plot(rk4_step, "Sistema Caótico - RK4")
