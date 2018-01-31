# @author : Camille LEONI
"""
PRELIMINARY RE-ENTRY SIMULATION

This program aims to modelise the trajectory of an object within a planetary 
atmosphere. It can either perform a balistic trajectory, a lifting reentry, or
even compute orbital decay, even though the accuracy needed for such a problem
should be improved.

To solve this problem, the Fundamental Dynamics Principle is solved, given
the few forces that are considered to be applied on the point-mass vehicle.
Those forces are as follow :
    - Weight, towards the center of the planet (not its center of mass, thus 
      neglecting J2 approximation)
    - Drag, in the opposite direction of the velocity vector
    - Lift, normal to the Drag and away from the planet.

Boundary conditions are given by the user in the "INITIAL CONDITIONS" section.
Those are :
    - planetary constants (supposed to be constant) such as the radius, the 
      gravity on the ground.
    - atmosphere model given by a density function
    - the object parameters : surface area, Cx & Cz (supposed to be constant),
      mass...
    - the initial flight conditions : altitude, velocity, flight-path angle
    - the simulation settings : timestep size, maximum time for the simulation
      to run, Earth orientation for the plots

The simulation provides a set of different plots, as with the basic 
informations one's need about the re-entry (if the object modelised effectively 
hit the ground).
"""

import numpy as np
from matplotlib import pyplot
import math
# -----------------------------------------------------------------
#   FUNCTIONS :
#    - simulation
#    - density_USSA76
# -----------------------------------------------------------------

def simulation(h0, V0, y0, planet_data, vehicle_data):
    """ Simulate the effects of atmosphere on an object
    
    Parameters
    ----------
    h0 : scalar [km]
        initial altitude
    V0 : scalar [m/s]
        initial velocity (negative means the trajectory is counter-clockwise)
    y0 : scalar [°]
        initial flight-path angle (negative means the velocity vector is towards 
        the Earth)
    planet_data : list
        list that contains planetary settings: 
            - R : scalar [km]
                planetary radius
            - g0 : scalar [m/s²]
                gravity at surface
            - func_density : func
                function of the atmosphere's density as with the altitude
    vehicle_data : list
        list that contains vehicle settings :
            - surface_area : scalar [m²]
                cross section of the vehicle
            - mass_ini : scalar [kg]
                vehicle's mass before reentry
            - mass_end : scalar [kg]
                vehicle's mass after reentry
            - Cx : scalar
                drag coefficient
            - Cz : scalar
                lift coefficient
	
    Returns
    -------
    time : array [s]
        Time at each step
    altitude : array [km]
        Altitude of the object at each step
    velocity : array [m/s]
        Magnitude of the object's velocity vector at each step
    flight_path : array [°]
        Angle between the velocity vector and the local horizon at each step
    rotation : array [°]
        Position angle along with the fixed coordinate frame
    density : array [kg/m3]
        Atmosphere's density at each step
    Pdyn : array [kg/m/s²]
        Dynamic pressure at each step
    Drag : array [N]
        Magnitude of the drag exerced on the object at each step
    Lift: array [N]
        Magnitude of the lift exerced on the object at each step
    Weight : array [N]
        Magnitude of the weight exerced on the object at each step
    acceleration : array [m/s²]
        object's acceleration vector [1*2] at each step
    g_acc : array [g]
        g factor applied on the object at each step
    energy : array [J]
        Kinetic energy of the object at each step
    
    Other Parameters (see Notes)
    ----------------------------
    dT : scalar [s]
        time step size. This parameter may change as with the altitude and 
        velocity
    Tmax : scalar [s]
        maximum time for a simulation running
    O0 : scalar [°]
        initial position angle of the spacecraft.
	x_E, y_E : vectors
        fixed coordinate frame as with the planet center
    
    Notes
    -----
    Steps for solving the problem :
        Initialization of the parameters output
        Then, at each step :
            - instantaneous position & velocity of the current step are computed
                thanks to the previous steps with a first order approximation of
                the derivative - Newton's rule : Y'(i) = [Y(i) - Y(i-1)] / dT
            - altitude, time & angles are estimated thanks to the previous and
                current position & velocity.
            - with those angles, two coordinate frames are defined :
                > Object position frame :
                    y_S vector : local horizon vector, sense of the motion
                    x_S vector : forward the planet's center
                > Velocity position frame :
                    y_V vector : along the velocity vector
                    x_V vector : forward the planet
            - forces exerced on the object are defined :
                > Drag :    - D y_V
                > Lift :      L x_V
                > Weight: - m g x_S
            - First Law of Dynamics : mass * Acc = sum(Forces)
                Computation of the instantaneous acceleration which will help
                to compute the instantenous velocity of the next step
                
    The simulation is run until the object reaches the ground OR the time 
    reaches the maximum allowed Tmax.
    The initial position for the object is, in the (x_E, y_E) frame : 
        (R + altitude) * cos(O0) [x_E]
        (R + altitude) * sin(O0) [y_E]
    Built-in functions :
        - _steptime
        - _mass
        - dynamic_pressure
        - gravity
        - vector_angle
        - basis_change
        
    """
    # Simulation settings :
    dT = 0.1
    Tmax = 1e4                         # max. time for a simulation [s]
    O0 = 90                             # initial position of the spacecraft with Earth [°]
    x_E = np.array([1,0])               # X vector (fixed)
    y_E = np.array([0,1])               # Y vector (fixed)
    R = planet_data[0]                  # planet radius [km]
    g0 = planet_data[1]                 # planet gravity magnitude on the ground
    func_density = planet_data[2]       # density function for the planet's atmosphere
    surface_area = vehicle_data[0]      # surface area of the vehicle [m²]
    mass_ini = vehicle_data[1]              # mass of the vehicle [kg]
    mass_end = vehicle_data[2]
    Cx = vehicle_data[3]                # drag coefficient
    Cz = vehicle_data[4]                # lift coefficient
    
	# Functions    
    def _mass(altitude):
        if altitude > 100:
            return mass_ini
        else:
            return (mass_ini - mass_end) / 100 * altitude + mass_end
        
    def dynamic_pressure(density, velocity):
        """ Return the dynamic pressure for the spacecraft
        """
        return 1/2 * surface_area * density * velocity**2
    
    def gravity(altitude):
        """ Return the magnitude of the gravity force according to the 
            distance (altitude) from what is supposed to be the center of mass
            of the attractor body.
        """
        return g0 * (R / (R + altitude))**2
    
    def vector_angle(u_ini, u_end):
        """ Return the angle between two vectors, positive if counter
            clockwise.
        """
        acos_angle = np.dot(u_ini, u_end) / (np.linalg.norm(u_ini) * np.linalg.norm(u_end))
        if np.abs(acos_angle - 1) < 1e-15:
            acos_angle = 1
        sign = np.sign(np.cross(u_ini, u_end))
        return sign * math.acos(acos_angle) * 180 / np.pi
    
    def basis_change(u, angle):
        """ Return the value of a vector that has been rotated by an angle
        """
        angle = angle / 180 * np.pi
        return np.array([u[0] * np.cos(angle) - u[1] * np.sin(angle),
                         u[0] * np.sin(angle) + u[1] * np.cos(angle)])
    def func_flux(velocity, density, nradius):
        """ Return the value of the heat flow according to the 
            distance (altitude) from what is supposed to be the wall
            of the attractor body.
        """
        return 1.83e5*((velocity/1e4)**3.05)*(density/nradius)**0.5
    def func_twall(phi):
        epsilon= 0.8
        sigma= 5.67e-8
        return (phi/(epsilon*sigma))**0.25
        
        
	
    # Initialization :
    time = np.array([0])
    altitude = np.array([h0])
    mass = _mass(altitude[-1])
    rotation = np.array([O0])
    x_S = basis_change(x_E, rotation[-1] - 90)  #(in x_E, y_E basis)
    y_S = basis_change(y_E, rotation[-1] - 90)  #(in x_E, y_E basis)
    flight_path = np.array([y0])
    x_V = basis_change(x_S, flight_path[-1])    # in x_E, y_E basis
    y_V = basis_change(y_S, flight_path[-1])    # in x_E, y_E basis
    
    velocity = np.array([V0])
    density = np.array([func_density(altitude[-1])])
    Pdyn = np.array([dynamic_pressure(density[-1], velocity[-1])])
    Drag = np.array([Pdyn[-1] * Cx])
    Lift = np.array([Pdyn[-1] * Cz])
    Weight = np.array([gravity(altitude[-1])]) * mass
    acceleration = np.array([1 / mass * (- Drag[-1] * x_V + Lift[-1] * y_V - Weight[-1] * y_S)])        # xE, yE basis
    g_acc = np.array([np.linalg.norm(acceleration[-1]) / gravity(altitude[-1])])
    energy = np.array([ 1 / 2 * mass * np.linalg.norm(velocity[-1])**2])
    phi = np.array([func_flux(velocity[-1], density[-1],nradius)])
    twall = np.array([func_twall(phi[-1])])
    heat = np.array([0])
	
	# Simulation :
    while ((altitude[-1] > 0) and (time[-1] < Tmax)):
        vel_temp = velocity[-1] * x_V                   # V0
        acc_temp = acceleration[-1]                     # g0
        r1_temp = (R+altitude[-1]) * y_S                # r0
        r2_temp = r1_temp + dT * vel_temp / 1000        # r1
        vel_temp = vel_temp + dT * acc_temp             # V1
        time = np.append(time, time[-1] + dT)
        altitude = np.append(altitude, np.linalg.norm(r2_temp) - R)
        mass = _mass(altitude[-1])
        rotation = np.append(rotation, rotation[-1] + vector_angle(r1_temp, r2_temp))
        x_S = basis_change(x_E, rotation[-1] - 90)
        y_S = basis_change(y_E, rotation[-1] - 90)
        flight_path = np.append(flight_path, vector_angle(x_S, vel_temp))
        x_V = basis_change(x_S, flight_path[-1])
        y_V = basis_change(y_S, flight_path[-1])
        
        velocity = np.append(velocity, np.linalg.norm(vel_temp))
        density = np.append(density, func_density(altitude[-1]))
        Pdyn = np.append(Pdyn, dynamic_pressure(density[-1], velocity[-1]))
        Drag = np.append(Drag, Pdyn[-1] * Cx)
        Lift = np.append(Lift, Pdyn[-1] * Cz)
        Weight = np.append(Weight, gravity(altitude[-1]) * mass)
        acceleration = np.append(acceleration, np.array([1 / mass * (- Drag[-1] * x_V + Lift[-1] * y_V - Weight[-1] * y_S)]), axis=0)
        g_acc = np.append(g_acc, np.linalg.norm(acceleration[-1]) / gravity(altitude[-1]))
        energy = np.append(energy, 1 / 2 * mass * np.linalg.norm(velocity[-1])**2)
        phi = np.append(phi,func_flux(velocity[-1], density[-1],nradius))
        twall = np.append(twall,func_twall(phi[-1]))
        heat = np.append(heat, np.trapz(phi,time))
        
        
    if time[-1] < Tmax:
        altitude[-1] = 0
    
    # Printing
    print('')
    print('Time of flight : ' + repr(time[-1]) + 's')
    print('Terminal velocity : ' + repr(np.linalg.norm(velocity[-1])) + ' m/s')
    print('Maximum deceleration : ' + repr(max(g_acc)) + ' g')
    return time, altitude, velocity, flight_path, rotation, density, Pdyn, Drag, Lift, Weight, acceleration, g_acc, energy, phi, twall, heat


def density_USSA76(altitude):
    """Compute the density of the atmosphere for altitudes from sea level to 1,000 km high
        using exponential interpolation.
        Reference : U.S. Standard Atmosphere, 1976
        [https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770009539.pdf]
    """
    # Set of altitudes [km] :
    h = np.array([0, 25, 30, 40, 50, 60, 70, 
                  80, 90, 100, 110, 120, 130, 140, 
                  150, 180, 200, 250, 300, 350, 400, 
                  450, 500, 600, 700, 800, 900, 1000])
    
    # Corresponding USSA76 density [kg/m3] :
    r = np.array([1.225,     4.008e-2,  1.841e-2,  3.996e-3,  1.027e-3,  3.097e-4,  8.283e-5,
                  1.846e-5,  3.416e-6,  5.606e-7,  9.708e-8,  2.222e-8,  8.152e-9,  3.831e-9,
                  2.076e-9,  5.194e-10, 2.541e-10, 6.073e-11, 1.916e-11, 7.014e-12, 2.803e-12,
                  1.184e-12, 5.215e-13, 1.137e-13, 3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15])
	# Scale heights [km] for the exponential approximation
    alpha = np.array([ 7.310,  6.427,  6.546,   7.360,   8.342,   7.583,  6.661,
                      5.927,  5.533,  5.703,   6.782,   9.973,  13.243, 16.322,
                      21.652, 27.974, 34.934,  43.342,  49.755,  54.513, 58.019,
                      60.980, 65.654, 76.377, 100.587, 147.203, 208.020])
	# Handle altitudes outside of the range :
    if (altitude > 1000):
        altitude = 1000
    elif (altitude < 0) :
        altitude = 0
	# Exponential interpolation :
    for i in range (len(h)-1):
        if ((altitude >= h[i]) and (altitude < h[i+1])) :
            return r[i] * np.exp(-(altitude - h[i]) / alpha[i])
        if altitude == 1000:
            return r[-2] * np.exp(-(altitude - h[-2]) / alpha[-1])	

###########################################
#                CONSTANTS                #
###########################################
""" Parameters used for the simulation
"""
# Earth characteristics
R = 6371.008                # Earth radius [km]
g0 = 9.80665                # Earth gravity at surface [m/s²]
p0 = 1.225

# Lander characteristics
radius = 0.4                  # cross section radius [m]
area = np.pi * radius**2    # cross section area [m²]
Cz = 0                      # lift coefficient []
Cx = 1.15                   # drag coefficient []
mass_ini = 150
mass_end = 150                  # mass [kg]
nradius = 60e-2             # nose radius [m]
cflow = 1.705e-7            # C constant for heatflow

# Flight characteristics
h0 = 500                    # initial height [km]
V0 = 12500                  # initial velocity [m/s]
y0 = -19                   # initial flight-path angle (°)


###########################################
#             SIMULATION                  #
###########################################
"""Simulation
"""
T, h, V, y, o, p, P, D, L, W, G, g, K, phi ,twall, heat = simulation(h0, V0, y0, [R, g0, density_USSA76], [area, mass_ini, mass_end, Cx, Cz])

# Return Values
beta = area * Cx / mass_ini
print('Coefficient Balistique  : '+ str(beta)+'m²/kg')
print('Maximum heat flow : ' + str(np.amax(phi)) +'kW/m²')
print('Total heat value : ' + str(np.amax(heat)) + 'kJ/m²')
###########################################
#                   PLOT                  #
###########################################
""" Several plots showing each output of the simulation : velocity, altitude,
position, g factor, kinetic energy, drag & lift & weight, dynamic pressure,
density and the heat flow.
"""



# Figure 1 : Plot velocity against time (normal & semi-log)
pyplot.figure(figsize=(12,4))
pyplot.subplot(1,2,1)
pyplot.plot(T, V)
pyplot.xlabel('Time [s]')
pyplot.ylabel('Velocity [m/s]')
pyplot.title('Velocity against time (normal)')
pyplot.subplot(1,2,2)
pyplot.yscale('log')
pyplot.plot(T, V)
pyplot.xlabel('Time [s]')
pyplot.ylabel('Velocity [m/s]')
pyplot.title('Velocity against time (semi-log)')
        
# Figure 2 : Plot velocity against altitude (normal & semi-log)
pyplot.figure(figsize=(12,4))    
pyplot.subplot(1,2,1)
pyplot.plot(h, V)
pyplot.xlabel('Altitude [km]')
pyplot.ylabel('Velocity [m/s]')
pyplot.title('Velocity against altitude (normal)')
pyplot.subplot(1,2,2)
pyplot.yscale('log')
pyplot.plot(h, V)
pyplot.xlabel('Altitude [km]')
pyplot.ylabel('Velocity [m/s]')
pyplot.title('Velocity against altitude (semi-log)')
#
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
pyplot.plot(h, g)  # Without the first point at t = 0
pyplot.xlabel('Altitude [km]')
pyplot.ylabel('Acceleration [g]')
pyplot.title('Acceleration against altitude (normal)')
pyplot.subplot(1,2,2)
pyplot.plot(T, g)  # Without the first point at t = 0
pyplot.xlabel('Time [s]')
pyplot.ylabel('Acceleration [g]')
pyplot.title('Acceleration against time (normal)')

# Figure 6 : Plot Drag force & Weight force against altitude (log), against time (log)
pyplot.figure(figsize=(12,4))
pyplot.subplot(1,2,1)
pyplot.yscale('log')
pyplot.plot(h, p)          
pyplot.xlabel('Altitude [km]')
pyplot.ylabel('Density [kg/m3]')
pyplot.title('Density (altitude) (log)')
pyplot.subplot(1,2,2)
pyplot.yscale('log')
pyplot.plot(T, p)
pyplot.xlabel('Time [s]')
pyplot.ylabel('Density [kg/m3]')
pyplot.title('Density (time) (log)')

pyplot.figure(figsize=(12,4))
pyplot.subplot(1,2,1)
pyplot.yscale('log')
pyplot.plot(h, P)          
pyplot.xlabel('Altitude [km]')
pyplot.ylabel('Dynamic pressure [N]')
pyplot.title('Dynamic pressure (altitude) (log)')
pyplot.subplot(1,2,2)
pyplot.yscale('log')
pyplot.plot(T, P)
pyplot.xlabel('Time [s]')
pyplot.ylabel('Dynamic pressure [N]')
pyplot.title('Dynamic pressure (time) (log)')

pyplot.figure(figsize=(12,4))
pyplot.subplot(1,2,1)
pyplot.yscale('log')
pyplot.plot(h, D)          # Without the first point at t = 0
pyplot.plot(h, W)        # Without the first point at t = 0
pyplot.xlabel('Altitude [km]')
pyplot.ylabel('Force [N]')
pyplot.title('Magnitude of the forces (altitude) (log)')
pyplot.legend(('Drag force', 'Weight force'))
pyplot.subplot(1,2,2)
pyplot.yscale('log')
pyplot.plot(T, D)
pyplot.plot(T, W)
pyplot.xlabel('Time [s]')
pyplot.ylabel('Force [N]')
pyplot.title('Magnitude of the forces (time) (log)')
pyplot.legend(('Drag force', 'Weight force'))

# Figure 8 : Plot kinetic energy against altitude (normal), against time (normal)
pyplot.figure(figsize=(12,4))
pyplot.subplot(1,2,1)
pyplot.plot(h, K)
pyplot.xlabel('Altitude [km]')
pyplot.ylabel('Kinetic energy [J]')
pyplot.title('Kinetic energy against altitude (normal)')
pyplot.subplot(1,2,2)
pyplot.plot(T, K)
pyplot.xlabel('Time [s]')
pyplot.ylabel('Kinetic energy [J]')
pyplot.title('Kinetic energy against time (normal)')
#

# Figure 9 : Heat flow & total heat
pyplot.figure(figsize=(12,4))
pyplot.subplot(1,2,1)
pyplot.plot(phi, h, label="Heat flow[kW/m²]")
pyplot.plot(heat, h, label="Heat [kJ/m²]")     
pyplot.xlabel('Altitude [km]')
pyplot.title('Total Heat & heat flow (altitude)')
pyplot.subplot(1,2,2)
pyplot.plot(phi, T, label="Heat flow[kW/m²]")
pyplot.plot(heat, T, label="Heat [kJ/m²]")  
pyplot.xlabel('Time [s]')
pyplot.ylabel('Heat [J]')
pyplot.title('Total Heat & heat flow (time)')

# Figure 10 : Wall temperature against altitude (normal)
pyplot.figure(figsize=(12,12))
pyplot.plot(twall, h)
pyplot.ylabel('Altitude [km]')
pyplot.xlabel('K')
pyplot.title('Wall temperature against altitude (normal)')
#
# Figure 11 : Plot trajectory
R_x = (R + h) * np.cos(o/180*np.pi)         # x component of the spacecraft trajectory
R_y = (R + h) * np.sin(o/180*np.pi)         # y component of the spacecraft trajectory
Re_x = R * np.cos(o/180*np.pi)              # x component of the Earth ground
Re_y = R * np.sin(o/180*np.pi)              # y component of the Earth ground
pyplot.figure(figsize=(12,12))
pyplot.plot(R_x, R_y, 'b')
pyplot.plot(Re_x, Re_y, 'k')
pyplot.xlabel('x vector [km]')
pyplot.ylabel('y vector [km]')
pyplot.title('Trajectory of the spacecraft')
pyplot.legend(('Spacecraft', 'Ground'))
pyplot.axis('equal')

