import numpy as np
import matplotlib.pyplot as plt
import time

# ------------------------------------------------------------------- #

m1 = 1.0
m2 = 1.0
m3 = 0.000001

# Position
position_1 = np.array([-0.5,  1.0,  0.0])
position_2 = np.array([0.5,  0.0,  0.0])
position_3 = np.array([0.0,  0.001, 1.0])

# Velocity
velocity_1 = np.array([0.0, 0.347111, 0])
velocity_2 = np.array([0.0, -0.347111, 0.0])
velocity_3 = np.array([0.0, 0.0, -0.1])

# Time parameters
time_s, time_e = 0, 7
dt = 0.001  # Small time step
t_points = np.arange(time_s, time_e, dt)

# ------------------------------------------------------------------- #

def calculate_acceleration(p1, p2, p3):
    # Calculate gravitational acceleration based on Newton's law
    a1 = m3 * (p3 - p1) / np.linalg.norm(p3 - p1)**3 + m2 * (p2 - p1) / np.linalg.norm(p2 - p1)**3
    a2 = m3 * (p3 - p2) / np.linalg.norm(p3 - p2)**3 + m1 * (p1 - p2) / np.linalg.norm(p1 - p2)**3
    a3 = m1 * (p1 - p3) / np.linalg.norm(p1 - p3)**3 + m2 * (p2 - p3) / np.linalg.norm(p2 - p3)**3
    return a1, a2, a3

# ------------------------------------------------------------------- #

# Initialize arrays to store positions over time
positions_1 = [position_1]
positions_2 = [position_2]
positions_3 = [position_3]

t1 = time.time()

# Euler's method loop
for t in t_points:
    # Calculate accelerations
    a1, a2, a3 = calculate_acceleration(position_1, position_2, position_3)
    
    # Update velocities (v = v + a * dt)
    velocity_1 += a1 * dt
    velocity_2 += a2 * dt
    velocity_3 += a3 * dt
    
    # Update positions (x = x + v * dt)
    position_1 += velocity_1 * dt
    position_2 += velocity_2 * dt
    position_3 += velocity_3 * dt
    
    # Store updated positions
    positions_1.append(position_1.copy())
    positions_2.append(position_2.copy())
    positions_3.append(position_3.copy())

del positions_1[-1]
del positions_2[-1]
del positions_3[-1]

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

ax.set_title("The 3-Body Problem (RK4 Method)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.grid()
plt.legend()
plt.show()

# ------------------------------------------------------------------- #
