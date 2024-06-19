'''Test script for Homework 3, Computational Photonics, SS 2024:  FDTD 3D method.
'''


import numpy as np
from function_headers_fdtd import fdtd_3d, Fdtd3DAnimation
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

# simulation parameters
Nx = 199 # number of grid points in x-direction
Ny = 201 # number of grid points in y-direction
Nz = 5   # number of grid points in z-direction
dr = 30e-9 # grid spacing in [m]
time_span = 10e-15 # duration of simulation [s]

# x coordinates
x = np.arange(-int(np.ceil((Nx-1)/2)), int(np.floor((Nx-1)/2)) + 1)*dr
# y coordinates
y = np.arange(-int(np.ceil((Ny-1)/2)), int(np.floor((Ny-1)/2)) + 1)*dr

# source parameters
freq = 500e12 # pulse [Hz]
tau = 1e-15 # pulse width [s]
source_width = 2*dr # width of Gaussian current dist. [grid points]

# grid midpoints
midx = int(np.ceil((Nx-1)/2))
midy = int(np.ceil((Ny-1)/2))
midz = int(np.ceil((Nz-1)/2))


# %% create relative permittivity distribution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# eps_rel = ...

eps_rel = np.ones((Nx, Ny, Nz)) 

# %% current distributions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# jx = jy = np.zeros(...) 
# jz : Gaussion distribution in the xy-plane with a width of 2 grid points, 
# constant along z

# jx = ...
# jy = ...
# jz = ...

jx = np.zeros((Nx, Ny, Nz))
jy = np.zeros((Nx, Ny, Nz))
jz = np.zeros((Nx, Ny, Nz))
for i, xi in enumerate(x):
    for j, yj in enumerate(y):
        jz[i, j, :] = np.exp(-((xi)**2 + (yj)**2)/(source_width**2))

# plt.figure()
# plt.imshow(jz[:, :, midz].T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]])
# plt.colorbar(label='Current density $J_z$')
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.title('Current density distribution in the xy-plane')
# plt.show()

# output parameters
z_ind = midz # z-index of field output
output_step = 4 # time steps between field output

field_component = 'hx'
hx, ez, t = fdtd_3d(eps_rel, dr, time_span, freq, tau, jx, jy, jz, field_component, z_ind, output_step)

#%% run simulations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# please add your code here
# field_component = str(input('Enter the field component you want to plot (hx, ez): '))
# if field_component == 'hx':
#     hx, t = fdtd_3d(eps_rel, dr, time_span, freq, tau, jx, jy, jz, field_component, z_ind, output_step)
#     F1 = hx*Z0*1e6
#     titlestr = 'x-Component of Magnetic Field'
#     cb_label = '$\\Re\\{Z_0H_x\\}$ [µV/m]'
#     rel_color_range = 1/3
#     fps = 10

#     ani = Fdtd3DAnimation(x, y, t, F1, titlestr, cb_label, rel_color_range, fps)
#     ani.save('fdtd_3d_hx_animation.mp4', writer='ffmpeg', dpi=300)
#     plt.show()
# elif field_component == 'ez':
#     ez, t = fdtd_3d(eps_rel, dr, time_span, freq, tau, jx, jy, jz, field_component, z_ind, output_step)
#     F2 = ez*1e6
#     titlestr = 'z-Component of Electric Field'
#     cb_label = '$\\Re\\{E_z\\}$ [µV/m]'
#     rel_color_range = 1/3
#     fps = 10

#     ani = Fdtd3DAnimation(x, y, t, F2, titlestr, cb_label, rel_color_range, fps)
#     ani.save('fdtd_3d_ez_animation.mp4', writer='ffmpeg', dpi=300)
#     plt.show()
# else:
#     print('Invalid field component. Please enter either hx or ez.')


#%% movie of Hx %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F1 = hx*Z0*1e6
titlestr = 'x-Component of Magnetic Field'
cb_label = '$\\Re\\{Z_0H_x\\}$ [µV/m]'
rel_color_range = 1/3
fps = 10

ani = Fdtd3DAnimation(x, y, t, F1, titlestr, cb_label, rel_color_range, fps)
ani.save('fdtd_3d_hx_animation.mp4', writer='ffmpeg', dpi=300)
plt.show()

#%% movie of Ez %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
F2 = ez*1e6
titlestr = 'z-Component of Electric Field'
cb_label = '$\\Re\\{E_z\\}$ [µV/m]'
rel_color_range = 1/3
fps = 10

ani = Fdtd3DAnimation(x, y, t, F2, titlestr, cb_label, rel_color_range, fps)
ani.save('fdtd_3d_ez_animation.mp4', writer='ffmpeg', dpi=300)
plt.show()

# %% create representative figures of the results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# please add your code here