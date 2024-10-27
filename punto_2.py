import numpy as np
import matplotlib.pyplot as plt

# Definimos las masas estándar para los tres cuerpos
m1, m2, m3 = 1.0, 1.0, 1.0

# Constante gravitacional en unidades estándar
G = 1

# Función para calcular el centro de masa del sistema
def center_of_mass(r1, r2, r3):
    r_cm = (m1 * r1 + m2 * r2 + m3 * r3) / (m1 + m2 + m3)
    return r_cm

# Función que calcula las fuerzas gravitacionales entre los cuerpos
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
    
    a1 = F1 / m1
    a2 = F2 / m2
    a3 = F3 / m3
    
    r1_new = r1 + v1 * dt
    r2_new = r2 + v2 * dt
    r3_new = r3 + v3 * dt
    
    v1_new = v1 + a1 * dt
    v2_new = v2 + a2 * dt
    v3_new = v3 + a3 * dt
    
    return r1_new, r2_new, r3_new, v1_new, v2_new, v3_new

# Método de Runge-Kutta de cuarto orden (RK4)
def rk4_step(r1, r2, r3, v1, v2, v3, dt):
    def get_accel(r1, r2, r3):
        F1, F2, F3 = calc_forces(r1, r2, r3)
        a1 = F1 / m1
        a2 = F2 / m2
        a3 = F3 / m3
        return a1, a2, a3
    
    # Primera evaluación (k1)
    a1, a2, a3 = get_accel(r1, r2, r3)
    k1_r1, k1_r2, k1_r3 = v1, v2, v3
    k1_v1, k1_v2, k1_v3 = a1, a2, a3

    # Segunda evaluación (k2)
    r1_2 = r1 + k1_r1 * dt / 2
    r2_2 = r2 + k1_r2 * dt / 2
    r3_2 = r3 + k1_r3 * dt / 2
    v1_2 = v1 + k1_v1 * dt / 2
    v2_2 = v2 + k1_v2 * dt / 2
    v3_2 = v3 + k1_v3 * dt / 2
    a1_2, a2_2, a3_2 = get_accel(r1_2, r2_2, r3_2)
    
    # Tercera evaluación (k3)
    r1_3 = r1 + v1_2 * dt / 2
    r2_3 = r2 + v2_2 * dt / 2
    r3_3 = r3 + v3_2 * dt / 2
    v1_3 = v1 + a1_2 * dt / 2
    v2_3 = v2 + a2_2 * dt / 2
    v3_3 = v3 + a3_2 * dt / 2
    a1_3, a2_3, a3_3 = get_accel(r1_3, r2_3, r3_3)
    
    # Cuarta evaluación (k4)
    r1_4 = r1 + v1_3 * dt
    r2_4 = r2 + v2_3 * dt
    r3_4 = r3 + v3_3 * dt
    v1_4 = v1 + a1_3 * dt
    v2_4 = v2 + a2_3 * dt
    v3_4 = v3 + a3_3 * dt
    a1_4, a2_4, a3_4 = get_accel(r1_4, r2_4, r3_4)

    # Combinación final para obtener el siguiente paso
    r1_new = r1 + dt * (v1 + 2*v1_2 + 2*v1_3 + v1_4) / 6
    r2_new = r2 + dt * (v2 + 2*v2_2 + 2*v2_3 + v2_4) / 6
    r3_new = r3 + dt * (v3 + 2*v3_2 + 2*v3_3 + v3_4) / 6
    
    v1_new = v1 + dt * (a1 + 2*a1_2 + 2*a1_3 + a1_4) / 6
    v2_new = v2 + dt * (a2 + 2*a2_2 + 2*a2_3 + a2_4) / 6
    v3_new = v3 + dt * (a3 + 2*a3_2 + 2*a3_3 + a3_4) / 6
    
    return r1_new, r2_new, r3_new, v1_new, v2_new, v3_new

# Función para simular y graficar las trayectorias centradas en la masa
def simulate_and_plot(method, title, steps=30000, dt=0.001):
    r1, r2, r3 = r1_0, r2_0, r3_0
    v1, v2, v3 = v1_0, v2_0, v3_0

    r1_hist, r2_hist, r3_hist, cm_hist = [r1.copy()], [r2.copy()], [r3.copy()], []
    
    for _ in range(steps):
        r1, r2, r3, v1, v2, v3 = method(r1, r2, r3, v1, v2, v3, dt)
        cm = center_of_mass(r1, r2, r3)
        cm_hist.append(cm.copy())
        r1_hist.append((r1 - cm).copy())
        r2_hist.append((r2 - cm).copy())
        r3_hist.append((r3 - cm).copy())
    
    r1_hist = np.array(r1_hist)
    r2_hist = np.array(r2_hist)
    r3_hist = np.array(r3_hist)

    # Graficar las trayectorias centradas en la masa
    plt.figure(figsize=(6, 6))
    plt.plot(r1_hist[:, 0], r1_hist[:, 1], label="Cuerpo 1")
    plt.plot(r2_hist[:, 0], r2_hist[:, 1], label="Cuerpo 2")
    plt.plot(r3_hist[:, 0], r3_hist[:, 1], label="Cuerpo 3")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

# Configuración para un sistema estable: órbitas más coordinadas
r1_0_stable = np.array([0.97000436, -0.24308753])
r2_0_stable = np.array([-0.97000436, 0.24308753])
r3_0_stable = np.array([0.0, 0.0])

v1_0_stable = np.array([0.466203685, 0.43236573])
v2_0_stable = np.array([0.466203685, 0.43236573])
v3_0_stable = np.array([-0.93240737, -0.86473146])

# Configuración para un sistema caótico: posiciones y velocidades desbalanceadas
r1_0_chaotic = np.array([1.0, 0.5])
r2_0_chaotic = np.array([-1.0, -0.5])
r3_0_chaotic = np.array([0.5, -1.0])

v1_0_chaotic = np.array([0.6, -0.4])
v2_0_chaotic = np.array([-0.6, 0.4])
v3_0_chaotic = np.array([0.0, 0.8])

# Sistema estable - Método de Euler
r1_0, r2_0, r3_0 = r1_0_stable, r2_0_stable, r3_0_stable
v1_0, v2_0, v3_0 = v1_0_stable, v2_0_stable, v3_0_stable
simulate_and_plot(euler_step, "Sistema Estable - Euler")

# Sistema caótico - Método de Euler
r1_0, r2_0, r3_0 = r1_0_chaotic, r2_0_chaotic, r3_0_chaotic
v1_0, v2_0, v3_0 = v1_0_chaotic, v2_0_chaotic, v3_0_chaotic
simulate_and_plot(euler_step, "Sistema Caótico - Euler")

# Sistema estable - Método RK4
r1_0, r2_0, r3_0 = r1_0_stable, r2_0_stable, r3_0_stable
v1_0, v2_0, v3_0 = v1_0_stable, v2_0_stable, v3_0_stable
simulate_and_plot(rk4_step, "Sistema Estable - RK4")

# Sistema caótico - Método RK4
r1_0, r2_0, r3_0 = r1_0_chaotic, r2_0_chaotic, r3_0_chaotic
v1_0, v2_0, v3_0 = v1_0_chaotic, v2_0_chaotic, v3_0_chaotic
simulate_and_plot(rk4_step, "Sistema Caótico - RK4")
