import numpy as np
from matplotlib import pyplot
import math

###########################################
#               FUNCTIONS                 #
###########################################
#     MATH OPERATIONS    #
##########################
def coordinate_switch(u, y): # u = {ux, uy} ; y is a list of angle [°]
    u1 = u[0]
    u2 = u[1]
    for i in range(len(y)):
        y[i] = y[i] / 180 * np.pi
        v1 = u1 * np.cos(y[i]) - u2 * np.sin(y[i])
        v2 = u1 * np.sin(y[i]) + u2 * np.cos(y[i])
        u1 = v1
        u2 = v2
    return np.array([v1, v2])
    # returns the same vector in input with a different coordinate system.
    # The coordinate system in output is a rotation by y° of the one in input
    # One can rotate more than 1 time using a list of angles (but always along the same axis)

def module(u): # u is a vector
    S = 0
    for k in range(len(u)):
        S = S + u[k]**2
    return np.sqrt(S)
    # returns the module 2 of the vector in input

def vec_angle(u):   # u = {ux, uy}
    U = module(u)
    y = math.acos(u[0]/U)
    if u[1] < 0:
        y = -y
    return y / np.pi * 180
    # return the angle between the vector & the x-axis of the coordinate system which is used to express the vector

##########################
#    Atmospheric model   #
##########################
def lift(p, V): # p: density [kg/m3] ; V: velocity [m/s]
    return 1/2 * area * Cz * p * V**2
    # given the density and velocity, returns the lift force [N]

def drag(p, V): # p: density [kg/m3] ; V: velocity [m/s]
    return 1/2 * area * Cx * p * V**2
    # given the density and velocity, returns the drag force [N]

def gravity(h): # h: altitude [km]
    rad = R+h               # distance from Earth center [km]
    return g0 * (R / rad)**2
    # given the altitude, returns the gravity acceleration [m/s²]
    
def density(h): # h: altitude [km]
    h = h*1000                        # conversion from km to m
    return p0*np.exp(-1.378e-4 * h)

##########################
#    Re-entry flight     #
##########################
def reentry(h, V, y, T): # h : altitude [km] ; V : velocity vector in Rv
    global dT
    # Let us assume that we will use 3 coordinate systems :
    # 1/ Rv is relative to the spacecraft velocity so that : x-axis is given by V
    #                                                        y-axis is away from Earth
    # 2/ Rs is relative to the spacecraft local horizon :    x-axis is along the local horizon (clockwise motion)
    #                                                        y-axis is away from Earth
    # 3/ Re is relative to the Earth center :                z-axis is along the Earth rotational axis
    # The initial position of the spacecraft with this coordinate system is Rs : (0, h0)
    
    # Computation :
    p = density(h)                              # density of the air [kg/m3]
    g = gravity(h)                              # magnitude of the gravity [m/s²]
    D = np.array([-drag(p, module(V)), 0])      # drag vector (Rv)
    L = np.array([0, lift(p, module(V))])       # lift vector (Rv)
    W = np.array([0, - g * mass])               # Weight vector (Rs)
    V = coordinate_switch(V, [y])               # Velocity vector switch to Rs
    D = coordinate_switch(D, [y])               # Drag vector (Rs)
    L = coordinate_switch(L, [y])               # Lift vector (Rs)
    Acc = 1 / mass * (L + W + D)                # Dynamic equation : G = Sum(F) / mass (Rs)
    dV = Acc * dT                               # dV vector [m/s] (Rs)
    dh = V[1] * dT / 1000                       # vertical velocity magnitude [km/s]
    h = h + dh                                  # altitude [km]
    V = V + dV                                  # velocity vector [m/s](Rs)
    y = vec_angle(V)                            # flight path angle (velocity against local horizon for Rs)
    T = T + dT                                  # time
    V = coordinate_switch(V, [-y])              # velocity vector [m/s] (Rv)
    
    
    
    any_data = np.array([module(D), module(W), module(Acc), g])    
    # This one is truly special as Rey in SW8
    # It containts any useful data we want to get as an output
    
    return h, V, y, T, any_data
    # returns altitude, velocity vector, y angle (°) and time in Rs coordinate system


###########################################
#                CONSTANTS                #
###########################################
# General constants
Pi2 = 2 * np.pi

# Earth characteristics
R = 6371.008                # Earth radius [km]
g0 = 9.80665                # Earth gravity at surface [m/s²]
p0 = 1.225                  # Air density at sea level [kg/m3]

# Lander characteristics
radius = 1                  # cross section radius [m]
area = np.pi * radius**2    # cross section area [m²]
Cz = 0                      # lift coefficient []
Cx = 1.15                   # drag coefficient []
mass = 500                  # mass [kg]

# Flight characteristics
h0 = 100                    # initial height [km]
V0 = 12000                  # initial velocity [m/s]
y0 = -20                    # initial flight-path angle (°)
dT = 0.1                    # Step size (s)

###########################################
#             SIMULATION                  #
###########################################
# Initialization
h = h0
V = np.array([V0, 0])
y = y0
T = 0
i=1

while h > 0:
    h, V, y, T, data = reentry(h, V, y, T)
    i = i+1
    # This piece of code is supposed to return the length of the state-vector
    # Basically it computes all the simulation in order to know precisely the length
    # Hence a huge loss of resources (not that much as for time, program is pretty fast)

# State-vector initialization
h = np.zeros(i)
h[0] = h0                   # h is the altitude vector starting from h0 to ~ 0 km

V = np.zeros((i,2))
V[0] = np.array([V0, 0])    # V is the velocity vector {Vx, Vy} in Rv frame from Vini to Vfinal

y = np.zeros(i)
y[0] = y0                   # y is the flight-path angle vector from y0 to -90°

T = np.zeros(i)             # T is the time vector starting from 0 til the end of flight, evenly spaced by dT

Velocity = np.zeros(i)
Velocity[0] = V0            # velocity computes the module of the velocity vector at each step

Acceleration = np.zeros(i)  # acceleration is the acceleration [g] vector

Drag = np.zeros(i)

Weight = np.zeros(i)

Kinetic_energy = np.zeros(i)
Kinetic_energy[0] = 1 / 2 * mass * V0**2 # kinetic energy

k = 0
while h[k]>0:
    h[k+1], V[k+1], y[k+1], T[k+1], data = reentry(h[k], V[k], y[k], T[k])
        # Tip : "data" contains the following vector modules : [D, W, Acc, g]
    Velocity[k+1] = module(V[k+1])
    Acceleration[k+1] = data[2] / data[3] # Computes the value of acceleration in g-number
    Drag[k+1] = data[0]
    Weight[k+1] = data[1]
    Kinetic_energy[k+1] = 1 / 2 * mass * Velocity[k+1]**2
    k = k+1
    # This fills all the state vectors above.
    
    
###########################################
#                   PLOT                  #
###########################################

# Figure 1 : Plot velocity against time (normal & semi-log)
pyplot.figure(figsize=(12,4))
pyplot.subplot(1,2,1)
pyplot.plot(T, Velocity)
pyplot.xlabel('Time [s]')
pyplot.ylabel('Velocity [m/s]')
pyplot.title('Velocity against time (normal)')
pyplot.subplot(1,2,2)
pyplot.yscale('log')
pyplot.plot(T, Velocity)
pyplot.xlabel('Time [s]')
pyplot.ylabel('Velocity [m/s]')
pyplot.title('Velocity against time (semi-log)')
        
# Figure 2 : Plot velocity against altitude (normal & semi-log)
pyplot.figure(figsize=(12,4))    
pyplot.subplot(1,2,1)
pyplot.plot(h, Velocity)
pyplot.xlabel('Altitude [km]')
pyplot.ylabel('Velocity [m/s]')
pyplot.title('Velocity against altitude (normal)')
pyplot.subplot(1,2,2)
pyplot.yscale('log')
pyplot.plot(h, Velocity)
pyplot.xlabel('Altitude [km]')
pyplot.ylabel('Velocity [m/s]')
pyplot.title('Velocity against altitude (semi-log)')

# Figure 3 : Plot y against altitude (normal), against time (normal)
pyplot.figure(figsize=(12,4))
pyplot.subplot(1,2,1)
pyplot.plot(h, y)
pyplot.xlabel('Altitude [km]')
pyplot.ylabel('Flight path angle [°]')
pyplot.title('Flight path angle against altitude (normal)')
pyplot.subplot(1,2,2)
pyplot.plot(T, y)
pyplot.xlabel('Time [s]')
pyplot.ylabel('Flight path angle [°]')
pyplot.title('Flight path angle against time (normal)')

# Figure 4 : Plot altitude against time (normal)
pyplot.figure(figsize=(12,4))
pyplot.plot(T,h)
pyplot.xlabel('Time [s]')
pyplot.ylabel('Altitude [km]')
pyplot.title('Altitude against time (normal)')

# Figure 5 : Plot acceleration against altitude (normal), against time (normal)
pyplot.figure(figsize=(12,4))
pyplot.subplot(1,2,1)
pyplot.plot(h[1::], Acceleration[1::])  # Without the first point at t = 0
pyplot.xlabel('Altitude [km]')
pyplot.ylabel('Acceleration [g]')
pyplot.title('Acceleration against altitude (normal)')
pyplot.subplot(1,2,2)
pyplot.plot(T[1::], Acceleration[1::])  # Without the first point at t = 0
pyplot.xlabel('Time [s]')
pyplot.ylabel('Acceleration [g]')
pyplot.title('Acceleration against time (normal)')

# Figure 6 : Plot Drag force & Weight force against altitude (log)
pyplot.figure(figsize=(12,4))
pyplot.yscale('log')
pyplot.plot(h[1::], Drag[1::])          # Without the first point at t = 0
pyplot.plot(h[1::], Weight[1::])        # Without the first point at t = 0
pyplot.xlabel('Altitude [km]')
pyplot.ylabel('Force [N]')
pyplot.title('Magnitude of the forces exerced on the vehicle against altitude (log)')
pyplot.legend(('Drag force', 'Weight force'))

# Figure 7 : Plot Drag force & Weight force against time (log)
pyplot.figure(figsize=(12,4))
pyplot.yscale('log')
pyplot.plot(T[1::], Drag[1::])          # Without the first point at t = 0
pyplot.plot(T[1::], Weight[1::])        # Without the first point at t = 0
pyplot.xlabel('Time [s]')
pyplot.ylabel('Force [N]')
pyplot.title('Magnitude of the forces exerced on the vehicle against time (log)')
pyplot.legend(('Drag force', 'Weight force'))

# Figure 8 : Plot Kinetic energy against time, altitude
pyplot.figure(figsize=(12,4))
pyplot.subplot(1,2,1)
pyplot.plot(h, Kinetic_energy)
pyplot.xlabel('Altitude [km]')
pyplot.ylabel('Kinetic energy [J]')
pyplot.title('kinetic energy against altitude (normal)')
pyplot.subplot(1,2,2)
pyplot.plot(T, Kinetic_energy)
pyplot.xlabel('Time [s]')
pyplot.ylabel('Kinetic energy [J]')
pyplot.title('kinetic energy against time (normal)')
