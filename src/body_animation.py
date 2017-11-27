import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.core.display import HTML
from scipy.integrate import ode


# Functions
def f(t, y, m1, m2, G):
    """Right hand side of Newtonian motion in inertial reference frame.
    t - current time, y - current state, m1, m2 - masses, G - gravitational constant"""
    x1 = y[0:3]    # position of m1
    x2 = y[3:6]    # position of m2
    v1 = y[6:9]    # velocity of m1
    v2 = y[9:12]   # velocity of m2
    F21 = G*m1*m2*(x2-x1)/(np.linalg.norm(x2-x1)**3) # Force of mass 2 on mass 1
    F12 = -F21
    return np.concatenate((v1, v2, F21/m1, F12/m2))
    
def animate(i):
    line1.set_data(x1[:i,0], x1[:i,1])
    line2.set_data(x2[:i,0], x2[:i,1])
    mark1.set_data(x1[i,0], x1[i,1])
    mark2.set_data(x2[i,0], x2[i,1])
    return (line1, line2, mark1, mark2)

O = ode(f).set_integrator('lsoda')


# Initial conditions & parameters
m1 = 0.5                             # the masses
m2 = 1
t0 = 0                               # initial time
x10 = np.array([1, -1, 0])           # position of m1 at t0
x20 = np.array([0, 0, 0])            # position of m2 at t0
v10 = np.array([0.3, 0.5, 0])        # velocity of m1 at t0
v20 = np.array([0.7, 0.8, 0])        # velocity of m2 at t0
G = 1 # gravitational constant, 1 in non-dimensional units


# Compute trajectories in the inertial reference frame
O.set_initial_value(np.concatenate((x10, x20, v10, v20)), t0).set_f_params(m1, m2, G)
t1 = 10
N = 1000
dt = (t1-t0)/N
orbit = np.empty((N+1,12))
orbit[0] = O.y
for i in range(1,N+1):
    orbit[i] = O.integrate(t0+i*dt)
    
    
# Visualize Trajectories in inertial reference frame

plt.close('all')       # close all figures (for repeated runs)

x1 = orbit[:,0:3]
x2 = orbit[:,3:6]
fig1 = plt.figure()
fig1.gca().set_aspect('equal')
line1, = fig1.gca().plot(x1[:,0], x1[:,1], lw=2, c='blue', alpha=0.7)
line2, = fig1.gca().plot(x2[:,0], x2[:,1], lw=2, c='orange', alpha=0.7)
mark1, = fig1.gca().plot(x1[-1,0], x1[-1,1], ls='', marker='o', markersize=m1*15, c='blue')
mark2, = fig1.gca().plot(x2[-1,0], x2[-1,1], ls='', marker='o', markersize=m2*15, c='orange')

fig1

anim = FuncAnimation(fig1, animate, frames=range(0,len(x1),2), interval=20)
HTML(anim.to_html5_video())
