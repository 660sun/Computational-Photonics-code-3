'''Test script for Homework 3, Computational Photonics, SS 2024:  FDTD 1D convergence dt.
'''


import numpy as np
import time
from function_headers_fdtd import fdtd_1d_t, Fdtd1DAnimation
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
dx = 15e-9 # grid spacing [m]
time_span = 60e-15 # duration of simulation [s]

Nx = int(round(x_span/dx)) + 1 # number of grid points

# source parameters
source_frequency = 500e12 # [Hz]
source_position = 0 # [m]
source_pulse_length = 1e-15 # [s]

# %% create permittivity distribution and run simulation %%%%%%%%%%%%%%%%%%%%%%

# please add your code here
x = np.linspace(-x_span/2, x_span/2, Nx)
eps_rel = np.ones(Nx)
for i in range(Nx):
    if x[i] > x_interface:
        eps_rel[i] = n2**2
    else:
        eps_rel[i] = n1**2

# time step
conv_test_fac =np.linspace(0.5, 6.5, 13)
dt = dx / (conv_test_fac * c)
# error evaluation _energy
error = np.zeros(len(dt))
# operation time
operation_time = np.zeros(len(dt))

for i, dti in enumerate(dt):
    Nt = int(round(time_span / dti)) + 1
    t = np.linspace(0, time_span, Nt)
    start = time.time()
    Ez, Hy, x = fdtd_1d_t(eps_rel, dx, dti, Nt, source_frequency, source_position, source_pulse_length)
    end = time.time()
    operation_time[i] = end - start
    error[i] = np.sum(Ez[-1, :]**2 + Hy[-1, :]**2)
    print(f"dt = {dti:.2e}, runtime = {end-start:.2f} s")

# rel_error = abs(error / error[-1] - 1)

plt.figure()
plt.plot(dt, operation_time, 'o-')
plt.xlabel('time step dt [s]')
plt.ylabel('operation time [s]')
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
plt.show()

plt.figure()
plt.plot(dt, error, 'o-')
plt.xlabel('time step dt [s]')
plt.ylabel('relative error')
# plt.xscale('log')
# plt.yscale('log')
plt.grid()
plt.show()

