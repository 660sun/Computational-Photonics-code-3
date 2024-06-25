'''Test script for Homework 3, Computational Photonics, SS 2024:  FDTD 1D convergence dx.
'''


import numpy as np
import time
from function_headers_fdtd import fdtd_1d, Fdtd1DAnimation
from matplotlib import pyplot as plt

# dark bluered colormap, registers automatically with matplotlib on import
import bluered_dark

plt.rcParams.update({
        'figure.figsize': (12/2.54, 9/2.54),
        'figure.subplot.bottom': 0.15,
        'figure.subplot.left': 0.165,
        'figure.subplot.right': 0.90,
        'figure.subplot.top': 0.9,
        'axes.grid': False,
        'image.cmap': 'bluered_dark',
})

plt.close('all')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# constants
c = 2.99792458e8 # speed of light [m/s]
mu0 = 4*np.pi*1e-7 # vacuum permeability [Vs/(Am)]
eps0 = 1/(mu0*c**2) # vacuum permittivity [As/(Vm)]
Z0 = np.sqrt(mu0/eps0) # vacuum impedance [Ohm]

# geometry parameters
x_span = 18e-6 # width of computatinal domain [m]
n1 = 1 # refractive index in front of interface
n2 = 2 # refractive index behind interface
x_interface = x_span/4 #postion of dielectric interface

# simulation parameters
dx = np.linspace(5e-9, 25e-9, 5) # grid spacing [m]
time_span = 60e-15 # duration of simulation [s]

 # number of grid points

# source parameters
source_frequency = 500e12 # [Hz]
source_position = 0 # [m]
source_pulse_length = 1e-15 # [s]

# %% create permittivity distribution and run simulation %%%%%%%%%%%%%%%%%%%%%%

oper_time = np.zeros(len(dx))
error = np.zeros(len(dx))
rel_error = np.zeros(len(dx))

# please add your code here
for i, dxi in enumerate(dx):
    Nx = int(round(x_span/dxi)) + 1
    start = time.time()
    x = np.linspace(-x_span/2, x_span/2, Nx)
    eps_rel = np.ones(Nx)
    for j in range(Nx):
        if x[j] > x_interface:
            eps_rel[j] = n2**2
        else:
            eps_rel[j] = n1**2
    Ez, Hy, x, t = fdtd_1d(eps_rel, dxi, time_span, source_frequency, source_position, source_pulse_length)
    end = time.time()
    oper_time[i] = end-start
    error[i] = np.sum(Ez[-1, :]**2)
    print(f'Elapsed time for Nx = {Nx}: {end-start:.2f} s')

for i in range(len(dx)):
    rel_error[i] = (error[i] - error[0])/error[0]

plt.figure()
plt.plot(dx, oper_time, 'o-')
plt.xlabel('Number of grid points')
plt.ylabel('Operation time [s]')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(dx, rel_error, 'o-')
plt.xlabel('Number of grid points')
plt.ylabel('Relative error')
plt.grid(True)
plt.show()