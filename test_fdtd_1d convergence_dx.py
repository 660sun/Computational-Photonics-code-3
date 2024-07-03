'''Test script for Homework 3, Computational Photonics, SS 2024:  FDTD 1D method.
'''

# %% 

import time
import numpy as np
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

for i, xi in enumerate(x):
    if abs(xi - source_position) < dx/2:
        source_position = i

# calculate reference field with initial value for dx
Ez_0, Hy_0, x_0, t_0 = fdtd_1d(eps_rel, dx, time_span, source_frequency, source_position, source_pulse_length)


# %% convergence for dx

# dx = np.logspace(-8, -7, num=10)
dx = np.linspace(10*1e-9,20*1e-9,10)
# print(dx)

errors = np.zeros(len(dx))
operation_time = np.zeros(len(dx))
 
for i, dxi in enumerate(dx):
    start = time.time()
    Nx = int(round(x_span/dxi)) + 1
    x = np.linspace(-x_span/2, x_span/2, Nx)
    # permettivity with new step size
    eps_rel = np.ones(Nx)
    for j in range(Nx):
        if x[j] > x_interface:
            eps_rel[j] = n2**2
        else:
            eps_rel[j] = n1**2

    for l, xi in enumerate(x):
        if abs(xi - source_position) < dxi/2:
            source_position = l

    Ez, Hy, x, t= fdtd_1d(eps_rel, dxi, time_span, source_frequency, source_position, source_pulse_length)
    end = time.time()
    operation_time[i] = end - start
    
   
    # Interpolate Ez to match the reference grid
    Ez_interp = np.interp(x_0, x, Ez[-1])
    Ez_ref_interp = Ez_0 #reference grid do not need to be interpolated since it's fixed
    
    # Compute L2 norm of the error
    errors[i] = np.linalg.norm(Ez_interp - Ez_ref_interp) / np.linalg.norm(Ez_ref_interp)
    print(f"dt = {dxi:.2e}, runtime = {end-start:.2f} s, L2 error = {errors[i]:.2e}")

# %%
# Plot operation time vs. time step
plt.figure()
plt.plot(dx*1e9, operation_time, 'o-')
plt.xlabel('spatial step $dx$ [nm]')
plt.ylabel('Operation time [s]')
# plt.xscale('log')  # Use logarithmic scale for better visualization
# plt.yscale('log')
plt.grid()
plt.title('Operation Time vs. Spatial Step')
plt.savefig('comp_time_dx.png', bbox_inches="tight")
plt.show()

# Plot L2 error norm vs. time step
plt.figure()
plt.plot(dx*1e9, errors, 'o-')
plt.xlabel('spatial step $dx$ [nm]')
plt.ylabel('L2 Norm of Error')
# plt.yscale('log')
# plt.xscale()
plt.grid()
plt.title('L2 Norm of Error vs. Spatial Step')
plt.savefig('error_dx.png', bbox_inches="tight")
plt.show()


# %%
