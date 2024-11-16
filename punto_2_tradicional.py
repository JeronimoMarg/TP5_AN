import numpy as np
import matplotlib.pyplot as plt


# 6.67430e-11
G = 1
m1, m2, m3 = 1.0, 1.0, 1.0

def center_of_mass(r1, r2, r3):
    r_cm = (m1 * r1 + m2 * r2 + m3 * r3) / (m1 + m2 + m3)
    return r_cm

def calcular_fuerza(r1, r2, r3):
    def fuerza(rA, rB, mA, mB):
        rAB = rB - rA
        dist = np.linalg.norm(rAB)
        return G * mA * mB * rAB / dist**3 if dist != 0 else np.zeros(2)
    
    F1 = fuerza(r1, r2, m1, m2) + fuerza(r1, r3, m1, m3)
    F2 = fuerza(r2, r1, m2, m1) + fuerza(r2, r3, m2, m3)
    F3 = fuerza(r3, r1, m3, m1) + fuerza(r3, r2, m3, m2)
    
    return F1, F2, F3

# EULER
def paso_euler(r1, r2, r3, v1, v2, v3, dt):
    F1, F2, F3 = calcular_fuerza(r1, r2, r3)
    
    #print(F1, F2, F3)

    a1 = F1 / m1
    a2 = F2 / m2
    a3 = F3 / m3
    
    r1_nuevo = r1 + v1 * dt
    r2_nuevo = r2 + v2 * dt
    r3_nuevo = r3 + v3 * dt
    
    v1_nuevo = v1 + a1 * dt
    v2_nuevo = v2 + a2 * dt
    v3_nuevo = v3 + a3 * dt
    
    return r1_nuevo, r2_nuevo, r3_nuevo, v1_nuevo, v2_nuevo, v3_nuevo

# RK4
def paso_rk4(r1, r2, r3, v1, v2, v3, dt):

    def aceleracion(r1, r2, r3):
        F1, F2, F3 = calcular_fuerza(r1, r2, r3)
        a1 = F1 / m1
        a2 = F2 / m2
        a3 = F3 / m3
        return a1, a2, a3
    
    # (k1)
    a1, a2, a3 = aceleracion(r1, r2, r3)
    k1_r1, k1_r2, k1_r3 = v1, v2, v3
    k1_v1, k1_v2, k1_v3 = a1, a2, a3

    # (k2)
    r1_2 = r1 + k1_r1 * dt / 2
    r2_2 = r2 + k1_r2 * dt / 2
    r3_2 = r3 + k1_r3 * dt / 2
    v1_2 = v1 + k1_v1 * dt / 2
    v2_2 = v2 + k1_v2 * dt / 2
    v3_2 = v3 + k1_v3 * dt / 2
    a1_2, a2_2, a3_2 = aceleracion(r1_2, r2_2, r3_2)
    
    # (k3)
    r1_3 = r1 + v1_2 * dt / 2
    r2_3 = r2 + v2_2 * dt / 2
    r3_3 = r3 + v3_2 * dt / 2
    v1_3 = v1 + a1_2 * dt / 2
    v2_3 = v2 + a2_2 * dt / 2
    v3_3 = v3 + a3_2 * dt / 2
    a1_3, a2_3, a3_3 = aceleracion(r1_3, r2_3, r3_3)
    
    # (k4)
    r1_4 = r1 + v1_3 * dt
    r2_4 = r2 + v2_3 * dt
    r3_4 = r3 + v3_3 * dt
    v1_4 = v1 + a1_3 * dt
    v2_4 = v2 + a2_3 * dt
    v3_4 = v3 + a3_3 * dt
    a1_4, a2_4, a3_4 = aceleracion(r1_4, r2_4, r3_4)

    # determinacion del valor de la funcion en el siguiente paso
    r1_nuevo = r1 + dt * (v1 + 2*v1_2 + 2*v1_3 + v1_4) / 6
    r2_nuevo = r2 + dt * (v2 + 2*v2_2 + 2*v2_3 + v2_4) / 6
    r3_nuevo = r3 + dt * (v3 + 2*v3_2 + 2*v3_3 + v3_4) / 6
    
    v1_nuevo  = v1 + dt * (a1 + 2*a1_2 + 2*a1_3 + a1_4) / 6
    v2_nuevo  = v2 + dt * (a2 + 2*a2_2 + 2*a2_3 + a2_4) / 6
    v3_nuevo  = v3 + dt * (a3 + 2*a3_2 + 2*a3_3 + a3_4) / 6
    
    return r1_nuevo, r2_nuevo, r3_nuevo, v1_nuevo, v2_nuevo, v3_nuevo

# RK5
def paso_rk5(r1, r2, r3, v1, v2, v3, dt):

    def aceleracion(r1, r2, r3):
        F1, F2, F3 = calcular_fuerza(r1, r2, r3)
        a1 = F1 / m1
        a2 = F2 / m2
        a3 = F3 / m3
        return a1, a2, a3
    
    # k1
    a1, a2, a3 = aceleracion(r1, r2, r3)
    k1_r1, k1_r2, k1_r3 = v1, v2, v3
    k1_v1, k1_v2, k1_v3 = a1, a2, a3

    # k2
    r1_2 = r1 + k1_r1 * dt / 4
    r2_2 = r2 + k1_r2 * dt / 4
    r3_2 = r3 + k1_r3 * dt / 4
    v1_2 = v1 + k1_v1 * dt / 4
    v2_2 = v2 + k1_v2 * dt / 4
    v3_2 = v3 + k1_v3 * dt / 4
    a1_2, a2_2, a3_2 = aceleracion(r1_2, r2_2, r3_2)

    # k3
    r1_3 = r1 + (k1_r1 + v1_2) * dt / 8
    r2_3 = r2 + (k1_r2 + v2_2) * dt / 8
    r3_3 = r3 + (k1_r3 + v3_2) * dt / 8
    v1_3 = v1 + (k1_v1 + a1_2) * dt / 8
    v2_3 = v2 + (k1_v2 + a2_2) * dt / 8
    v3_3 = v3 + (k1_v3 + a3_2) * dt / 8
    a1_3, a2_3, a3_3 = aceleracion(r1_3, r2_3, r3_3)

    # k4
    r1_4 = r1 - v1_2 * dt / 2 + v1_3 * dt
    r2_4 = r2 - v2_2 * dt / 2 + v2_3 * dt
    r3_4 = r3 - v3_2 * dt / 2 + v3_3 * dt
    v1_4 = v1 - a1_2 * dt / 2 + a1_3 * dt
    v2_4 = v2 - a2_2 * dt / 2 + a2_3 * dt
    v3_4 = v3 - a3_2 * dt / 2 + a3_3 * dt
    a1_4, a2_4, a3_4 = aceleracion(r1_4, r2_4, r3_4)

    # k5
    r1_5 = r1 + (3/16) * k1_r1 * dt + (9/16) * v1_4 * dt
    r2_5 = r2 + (3/16) * k1_r2 * dt + (9/16) * v2_4 * dt
    r3_5 = r3 + (3/16) * k1_r3 * dt + (9/16) * v3_4 * dt
    v1_5 = v1 + (3/16) * k1_v1 * dt + (9/16) * a1_4 * dt
    v2_5 = v2 + (3/16) * k1_v2 * dt + (9/16) * a2_4 * dt
    v3_5 = v3 + (3/16) * k1_v3 * dt + (9/16) * a3_4 * dt
    a1_5, a2_5, a3_5 = aceleracion(r1_5, r2_5, r3_5)

    # k6
    r1_6 = r1 - (3/7) * k1_r1 * dt + (2/7) * v1_2 * dt + (12/7) * v1_3 * dt - (12/7) * v1_4 * dt + (8/7) * v1_5 * dt
    r2_6 = r2 - (3/7) * k1_r2 * dt + (2/7) * v2_2 * dt + (12/7) * v2_3 * dt - (12/7) * v2_4 * dt + (8/7) * v2_5 * dt
    r3_6 = r3 - (3/7) * k1_r3 * dt + (2/7) * v3_2 * dt + (12/7) * v3_3 * dt - (12/7) * v3_4 * dt + (8/7) * v3_5 * dt
    v1_6 = v1 - (3/7) * k1_v1 * dt + (2/7) * a1_2 * dt + (12/7) * a1_3 * dt - (12/7) * a1_4 * dt + (8/7) * a1_5 * dt
    v2_6 = v2 - (3/7) * k1_v2 * dt + (2/7) * a2_2 * dt + (12/7) * a2_3 * dt - (12/7) * a2_4 * dt + (8/7) * a2_5 * dt
    v3_6 = v3 - (3/7) * k1_v3 * dt + (2/7) * a3_2 * dt + (12/7) * a3_3 * dt - (12/7) * a3_4 * dt + (8/7) * a3_5 * dt
    a1_6, a2_6, a3_6 = aceleracion(r1_6, r2_6, r3_6)

    # Combinación final para obtener el siguiente valor de las posiciones y velocidades
    r1_siguiente = r1 + (1/90) * (7 * k1_r1 + 32 * v1_3 + 12 * v1_4 + 32 * v1_5 + 7 * v1_6) * dt
    r2_siguiente = r2 + (1/90) * (7 * k1_r2 + 32 * v2_3 + 12 * v2_4 + 32 * v2_5 + 7 * v2_6) * dt
    r3_siguiente = r3 + (1/90) * (7 * k1_r3 + 32 * v3_3 + 12 * v3_4 + 32 * v3_5 + 7 * v3_6) * dt

    v1_siguiente = v1 + (1/90) * (7 * k1_v1 + 32 * a1_3 + 12 * a1_4 + 32 * a1_5 + 7 * a1_6) * dt
    v2_siguiente = v2 + (1/90) * (7 * k1_v2 + 32 * a2_3 + 12 * a2_4 + 32 * a2_5 + 7 * a2_6) * dt
    v3_siguiente = v3 + (1/90) * (7 * k1_v3 + 32 * a3_3 + 12 * a3_4 + 32 * a3_5 + 7 * a3_6) * dt

    return r1_siguiente, r2_siguiente, r3_siguiente, v1_siguiente, v2_siguiente, v3_siguiente


# SIMULACION
def simulate_and_plot(method, title, steps=10000, dt=0.01):
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
r1_0_chaotic = np.array([-0.5, 1.0])
r2_0_chaotic = np.array([0.5, 0.0])
r3_0_chaotic = np.array([0.0, 0.001])

v1_0_chaotic = np.array([0.0, 0.347111])
v2_0_chaotic = np.array([0.0, -0.347111])
v3_0_chaotic = np.array([0.0, 0.0])

# Sistema estable - Método de Euler
r1_0, r2_0, r3_0 = r1_0_stable, r2_0_stable, r3_0_stable
v1_0, v2_0, v3_0 = v1_0_stable, v2_0_stable, v3_0_stable
simulate_and_plot(paso_euler, "Sistema Estable - Euler")

# Sistema caótico - Método de Euler
r1_0, r2_0, r3_0 = r1_0_chaotic, r2_0_chaotic, r3_0_chaotic
v1_0, v2_0, v3_0 = v1_0_chaotic, v2_0_chaotic, v3_0_chaotic
simulate_and_plot(paso_euler, "Sistema Caótico - Euler")

# Sistema estable - Método RK4
r1_0, r2_0, r3_0 = r1_0_stable, r2_0_stable, r3_0_stable
v1_0, v2_0, v3_0 = v1_0_stable, v2_0_stable, v3_0_stable
simulate_and_plot(paso_rk4, "Sistema Estable - RK4")

# Sistema caótico - Método RK4
r1_0, r2_0, r3_0 = r1_0_chaotic, r2_0_chaotic, r3_0_chaotic
v1_0, v2_0, v3_0 = v1_0_chaotic, v2_0_chaotic, v3_0_chaotic
simulate_and_plot(paso_rk4, "Sistema Caótico - RK4")

# Sistema estable - Método RK5
r1_0, r2_0, r3_0 = r1_0_stable, r2_0_stable, r3_0_stable
v1_0, v2_0, v3_0 = v1_0_stable, v2_0_stable, v3_0_stable
simulate_and_plot(paso_rk5, "Sistema Estable - RK5")

# Sistema caótico - Método RK5
r1_0, r2_0, r3_0 = r1_0_chaotic, r2_0_chaotic, r3_0_chaotic
v1_0, v2_0, v3_0 = v1_0_chaotic, v2_0_chaotic, v3_0_chaotic
simulate_and_plot(paso_rk5, "Sistema Caótico - RK5")
