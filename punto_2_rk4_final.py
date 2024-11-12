import numpy as np
import matplotlib.pyplot as plt
import time

# ------------------------------------------------------------------- #

m1 = 1.0
m2 = 1.0
m3 = 1.0

#SISTEMA CAOTICO
# Position
position_1 = np.array([-0.5,  1.0,  0.0])
position_2 = np.array([0.5,  0.0,  0.0])
position_3 = np.array([0.0,  0.001, 1.0])

# Velocity
velocity_1 = np.array([0.0, 0.347111, 0])
velocity_2 = np.array([0.0, -0.347111, 0.0])
velocity_3 = np.array([0.0, 0.0, -0.1])

#SISTEMA ESTABLE
# Position
position_1 = np.array([-1.0, 0.0, 0.0])
position_2 = np.array([1.0, 0.0, 0.0])
position_3 = np.array([0.0, 0.0, 0.0])

# Velocity
velocity_1 = np.array([0.0, 0.5, 0.0])
velocity_2 = np.array([0.0, -0.5, 0.0])
velocity_3 = np.array([0.0, 0.0, 0.0])

#SISTEMA ESTABLE (8)
# Position
position_1 = np.array([-0.97000436, 0.24308753, 0.0])
position_2 = np.array([0.97000436, -0.24308753, 0.0])
position_3 = np.array([0.0, 0.0, 0.0])

# Velocity
velocity_1 = np.array([0.466203685, 0.43236573, 0.0])
velocity_2 = np.array([0.466203685, 0.43236573, 0.0])
velocity_3 = np.array([-0.93240737, -0.86473146, 0.0])

#SISTEMA ESTABLE (EULER COLINEAL)
# Position
position_1 = np.array([-1.0, 0.0, 0.0])
position_2 = np.array([0.0, 0.0, 0.0])
position_3 = np.array([1.0, 0.0, 0.0])

# Velocity
velocity_1 = np.array([0.5, 0.0, 0.0])
velocity_2 = np.array([0.0, 0.0, 0.0])
velocity_3 = np.array([-0.5, 0.0, 0.0])

#SISTEMA ESTABLE (LAGRANGE)
# Position
position_1 = np.array([1.0, 0.0, 0.0])
position_2 = np.array([-0.5, np.sqrt(3)/2, 0.0])
position_3 = np.array([-0.5, -np.sqrt(3)/2, 0.0])

# Velocity
velocity_1 = np.array([0.0, 0.5, 0.0])
velocity_2 = np.array([-0.5 * np.sqrt(3)/2, -0.5, 0.0])
velocity_3 = np.array([0.5 * np.sqrt(3)/2, -0.5, 0.0])

#SISTEMA PETALOS
# Position
position_1 = np.array([1.0, 0.5, 0.0])
position_2 = np.array([-1.0, -0.5, 0.0])
position_3 = np.array([0.5, -1.0, 0.0])

# Velocity
velocity_1 = np.array([0.6, -0.4, 0.0])
velocity_2 = np.array([-0.6, 0.4, 0.0])
velocity_3 = np.array([0.0, 0.8, 0.0])


# Time parameters
time_s, time_e = 0, 7
dt = 0.001  # Small time step
t_points = np.arange(time_s, time_e, dt)

# ------------------------------------------------------------------- #

def calculate_acceleration(p1, p2, p3, m1, m2, m3):
    # Calculate gravitational acceleration based on Newton's law
    a1 = m3 * (p3 - p1) / np.linalg.norm(p3 - p1)**3 + m2 * (p2 - p1) / np.linalg.norm(p2 - p1)**3
    a2 = m3 * (p3 - p2) / np.linalg.norm(p3 - p2)**3 + m1 * (p1 - p2) / np.linalg.norm(p1 - p2)**3
    a3 = m1 * (p1 - p3) / np.linalg.norm(p1 - p3)**3 + m2 * (p2 - p3) / np.linalg.norm(p2 - p3)**3
    return a1, a2, a3

def rk4_step(position_1, position_2, position_3, velocity_1, velocity_2, velocity_3, dt, m1, m2, m3):
    # Helper function to perform one RK4 step

    # Calculate k1 (initial slope)
    a1_k1, a2_k1, a3_k1 = calculate_acceleration(position_1, position_2, position_3, m1, m2, m3)
    p1_k1, p2_k1, p3_k1 = velocity_1, velocity_2, velocity_3

    # Calculate k2 (at midpoint using k1 values)
    a1_k2, a2_k2, a3_k2 = calculate_acceleration(
        position_1 + p1_k1 * dt / 2, 
        position_2 + p2_k1 * dt / 2, 
        position_3 + p3_k1 * dt / 2, 
        m1, m2, m3)
    
    p1_k2, p2_k2, p3_k2 = velocity_1 + a1_k1 * dt / 2, velocity_2 + a2_k1 * dt / 2, velocity_3 + a3_k1 * dt / 2

    # Calculate k3 (at midpoint using k2 values)
    a1_k3, a2_k3, a3_k3 = calculate_acceleration(
        position_1 + p1_k2 * dt / 2, 
        position_2 + p2_k2 * dt / 2, 
        position_3 + p3_k2 * dt / 2, 
        m1, m2, m3)
    
    p1_k3, p2_k3, p3_k3 = velocity_1 + a1_k2 * dt / 2, velocity_2 + a2_k2 * dt / 2, velocity_3 + a3_k2 * dt / 2

    # Calculate k4 (at end using k3 values)
    a1_k4, a2_k4, a3_k4 = calculate_acceleration(
        position_1 + p1_k3 * dt, 
        position_2 + p2_k3 * dt, 
        position_3 + p3_k3 * dt, 
        m1, m2, m3)
    
    p1_k4, p2_k4, p3_k4 = velocity_1 + a1_k3 * dt, velocity_2 + a2_k3 * dt, velocity_3 + a3_k3 * dt

    # Update positions and velocities using RK4 formula
    position_1_next = position_1 + (p1_k1 + 2*p1_k2 + 2*p1_k3 + p1_k4) * dt / 6
    position_2_next = position_2 + (p2_k1 + 2*p2_k2 + 2*p2_k3 + p2_k4) * dt / 6
    position_3_next = position_3 + (p3_k1 + 2*p3_k2 + 2*p3_k3 + p3_k4) * dt / 6

    velocity_1_next = velocity_1 + (a1_k1 + 2*a1_k2 + 2*a1_k3 + a1_k4) * dt / 6
    velocity_2_next = velocity_2 + (a2_k1 + 2*a2_k2 + 2*a2_k3 + a2_k4) * dt / 6
    velocity_3_next = velocity_3 + (a3_k1 + 2*a3_k2 + 2*a3_k3 + a3_k4) * dt / 6

    return position_1_next, position_2_next, position_3_next, velocity_1_next, velocity_2_next, velocity_3_next

# ------------------------------------------------------------------- #

# Initialize arrays to store positions over time
positions_1 = [position_1]
positions_2 = [position_2]
positions_3 = [position_3]

t1 = time.time()

# RK4 loop
print ("Numero de iteraciones: " , len(t_points))
for t in t_points:
    position_1, position_2, position_3, velocity_1, velocity_2, velocity_3 = rk4_step(
        position_1, position_2, position_3, velocity_1, velocity_2, velocity_3, dt, m1, m2, m3)

    # Store updated positions
    positions_1.append(position_1.copy())
    positions_2.append(position_2.copy())
    positions_3.append(position_3.copy())

t2 = time.time()
print(f"Solved in: {t2-t1:.3f} [s]")

# Convert to arrays for easy plotting
positions_1 = np.array(positions_1)
positions_2 = np.array(positions_2)
positions_3 = np.array(positions_3)

# ------------------------------------------------------------------- #

# Plotting the orbits
fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

ax.plot(positions_1[:, 0], positions_1[:, 1], positions_1[:, 2], 'green', label='Planet 1', linewidth=1)
ax.plot(positions_2[:, 0], positions_2[:, 1], positions_2[:, 2], 'red', label='Planet 2', linewidth=1)
ax.plot(positions_3[:, 0], positions_3[:, 1], positions_3[:, 2], 'blue', label='Planet 3', linewidth=1)

ax.plot([positions_1[-1, 0]], [positions_1[-1, 1]], [positions_1[-1, 2]], 'o', color='green', markersize=6)
ax.plot([positions_2[-1, 0]], [positions_2[-1, 1]], [positions_2[-1, 2]], 'o', color='red', markersize=6)
ax.plot([positions_3[-1, 0]], [positions_3[-1, 1]], [positions_3[-1, 2]], 'o', color='blue', markersize=6)

ax.set_title("Problema de los tres cuerpos (RK4)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.grid()
plt.legend()
plt.show()

# ------------------------------------------------------------------- #
