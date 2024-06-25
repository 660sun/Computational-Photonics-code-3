'''Test script for Homework 3, Computational Photonics, SS 2024:  FDTD 1D convergence dt.
'''


import numpy as np
import time
from function_headers_fdtd import fdtd_1d_t
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

# Reference time step
dt_ref = dx / (20 * c)  # Fine time step for reference solution
Nt_ref = int(round(time_span / dt_ref)) + 1

# Run reference simulation
Ez_ref, Hy_ref, x = fdtd_1d_t(eps_rel, dx, dt_ref, Nt_ref, source_frequency, source_position, source_pulse_length)

# Define time steps for convergence test
conv_test_fac = np.linspace(2, 12, 11)
dt_values = dx / (conv_test_fac * c)

# Error evaluation and operation time
errors = np.zeros(len(dt_values))
operation_time = np.zeros(len(dt_values))

for i, dti in enumerate(dt_values):
    start = time.time()
    Nt = int(round(time_span / dti)) + 1
    Ez, Hy, x = fdtd_1d_t(eps_rel, dx, dti, Nt, source_frequency, source_position, source_pulse_length)
    end = time.time()
    operation_time[i] = end - start
    
    # Interpolate Ez to match the reference grid
    Ez_interp = np.interp(x, x, Ez[-1])
    Ez_ref_interp = np.interp(x, x, Ez_ref[-1])
    
    # Compute L2 norm of the error
    errors[i] = np.linalg.norm(Ez_interp - Ez_ref_interp) / np.linalg.norm(Ez_ref_interp)
    print(f"dt = {dti:.2e}, runtime = {end-start:.2f} s, L2 error = {errors[i]:.2e}")

# Plot operation time vs. time step
plt.figure()
plt.plot(dt_values, operation_time, 'o-')
plt.xlabel('Time step dt [s]')
plt.ylabel('Operation time [s]')
plt.grid()
plt.title('Operation Time vs. Time Step')
plt.show()

# Plot L2 error norm vs. time step
plt.figure()
plt.plot(dt_values, errors, 'o-')
plt.xlabel('Time step dt [s]')
plt.ylabel('L2 Norm of Error')
plt.yscale('log')  # Use logarithmic scale for better visualization
plt.grid()
plt.title('L2 Norm of Error vs. Time Step')
plt.show()
