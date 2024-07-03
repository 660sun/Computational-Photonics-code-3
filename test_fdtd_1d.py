'''Test script for Homework 3, Computational Photonics, SS 2024:  FDTD 1D method.
'''


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

Ez, Hy, x, t = fdtd_1d(eps_rel, dx, time_span, source_frequency, source_position, source_pulse_length)

# %% make video %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fps = 25
step = t[-1]/fps/30
ani = Fdtd1DAnimation(x, t, Ez, Hy, x_interface=x_interface, step=step, fps=fps)
ani.save('fdtd_1d_animation.mp4', writer='ffmpeg', dpi=300)
plt.show()

# %% create representative figures of the results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# please add your code here

# figures of magnetic field
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
indices = [50, 550, 800, 1300]

# Plot each index in a subplot
for ax, idx in zip(axs.ravel(), indices):
    ax.plot(x * 1e6, Z0*Hy[idx] * 1e6, color = 'orange')
    ax.vlines(x_interface * 1e6, ymin=-5, ymax=5, ls='--', color='k')
    ax.set_xlim(x[[0,-1]]*1e6)
    ax.set_xlabel('$x$ [µm]')
    ax.set_ylabel('$\\Re\\{H_y\\}$ [µV/m]')
    ax.set_title(f'Magnetic field at $ct = {c*t[idx]*1e6:.2f}$µm, i={idx}')
plt.tight_layout()
plt.show()

# figures of electric field
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
indices = [50, 550, 800, 1300]

# Plot each index in a subplot
for ax, idx in zip(axs.ravel(), indices):
    ax.plot(x * 1e6, Ez[idx] * 1e6)
    ax.vlines(x_interface * 1e6, ymin=-5, ymax=5, ls='--', color='k')
    ax.set_xlim(x[[0,-1]]*1e6)
    ax.set_xlabel('$x$ [µm]')
    ax.set_ylabel('$\\Re\\{E_z\\}$ [µV/m]')
    ax.set_title(f'Electric field at $ct = {c*t[idx]*1e6:.2f}$µm, i={idx}')
plt.tight_layout()
plt.show()

