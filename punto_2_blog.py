#! python3
# A program that produces trajectories of three bodies
# according to Netwon's laws of gravitation

# import third-party libraries
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')

# masses of planets
m1 = 10
m2 = 20
m3 = 30
G = 1

# starting coordinates for planets
# p1_start = x_1, y_1, z_1
p1_start = np.array([-10, 10, -11])
v1_start = np.array([-3, 0, 0])

# p2_start = x_2, y_2, z_2
p2_start = np.array([0, 0, 0])
v2_start = np.array([0, 0, 0])

# p3_start = x_3, y_3, z_3
p3_start = np.array([10, 10, 12])
v3_start = np.array([3, 0, 0])

def accelerations(r1, r2, r3):
    def fuerza(rA, rB, mA, mB):
        rAB = rB - rA
        dist = np.linalg.norm(rAB)
        return G * mA * mB * rAB / dist**3 if dist != 0 else np.zeros(2)
    
    F1 = fuerza(r1, r2, m1, m2) + fuerza(r1, r3, m1, m3)
    F2 = fuerza(r2, r1, m2, m1) + fuerza(r2, r3, m2, m3)
    F3 = fuerza(r3, r1, m3, m1) + fuerza(r3, r2, m3, m2)
    
    return F1, F2, F3

# parameters
delta_t = 0.001
steps = 100000

# initialize trajectory array
p1 = np.array([[0.,0.,0.] for i in range(steps)])
v1 = np.array([[0.,0.,0.] for i in range(steps)])

p2 = np.array([[0.,0.,0.] for j in range(steps)])
v2 = np.array([[0.,0.,0.] for j in range(steps)])

p3 = np.array([[0.,0.,0.] for k in range(steps)])
v3 = np.array([[0.,0.,0.] for k in range(steps)])

# starting point and velocity
p1[0], p2[0], p3[0] = p1_start, p2_start, p3_start

v1[0], v2[0], v3[0] = v1_start, v2_start, v3_start

# evolution of the system
for i in range(steps-1):
	# calculate derivatives
	dv1, dv2, dv3 = accelerations(p1[i], p2[i], p3[i])
	dv1 = dv1 / m1
	dv2 = dv2 / m2
	dv3 = dv3 / m3

	v1[i + 1] = v1[i] + dv1 * delta_t
	v2[i + 1] = v2[i] + dv2 * delta_t
	v3[i + 1] = v3[i] + dv3 * delta_t

	p1[i + 1] = p1[i] + v1[i] * delta_t
	p2[i + 1] = p2[i] + v2[i] * delta_t
	p3[i + 1] = p3[i] + v3[i] * delta_t
	
fig = plt.figure(figsize=(8, 8))
# Cambia esto:
# ax = fig.gca(projection='3d')
# Por esto:
ax = fig.add_subplot(111, projection='3d')

plt.gca().patch.set_facecolor('black')

plt.plot([i[0] for i in p1], [j[1] for j in p1], [k[2] for k in p1], '^', color='red', lw=0.05, markersize=0.01, alpha=0.5)
plt.plot([i[0] for i in p2], [j[1] for j in p2], [k[2] for k in p2], '^', color='white', lw=0.05, markersize=0.01, alpha=0.5)
plt.plot([i[0] for i in p3], [j[1] for j in p3], [k[2] for k in p3], '^', color='blue', lw=0.05, markersize=0.01, alpha=0.5)

plt.axis('on')

# Optional: use if reference axes skeleton is desired
ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])

# Make panes have the same color as the background
ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0)), ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0)), ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

plt.show()
plt.close()