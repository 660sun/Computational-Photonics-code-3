'''Test script for Homework 3, Computational Photonics, SS 2024:  FDTD 3D method.
'''

# %% 
import time
import numpy as np
from function_headers_fdtd import fdtd_3d, Fdtd3DAnimation
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

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

# output parameters
z_ind = midz # z-index of field output
output_step = 4 # time steps between field output

field_component = 'hx'
hx_0, ez_0, t_0 = fdtd_3d(eps_rel, dr, time_span, freq, tau, jx, jy, jz, field_component, z_ind, output_step)

# %% convergence test for dt

# dt = np.linspace()


# %%
errors = np.zeros(len(dr))
operation_time = np.zeros(len(dr))
print(f'ez_0:{ez_0.shape}')
for l, dri in enumerate(dr):
    start = time.time()
    # insert time

    hx, ez, t= fdtd_3d(eps_rel, dri, time_span, freq, tau, jx, jy, jz, field_component, z_ind, output_step)
    end = time.time()
    operation_time[l] = end - start
    
    print(f'ez: {ez.shape}')
    interp_func_time = RegularGridInterpolator((t,), ez, bounds_error=False, fill_value=0)
    
    # Interpolate ez to match the reference grid
    t_mesh, x_mesh, y_mesh = np.meshgrid(t_0, x, y, indexing='ij')
    points = np.array([t_mesh.ravel(), x_mesh.ravel(), y_mesh.ravel()]).T
    Ez_interp = interp_func_time(t_0).reshape(len(t_0), Nx, Ny)

    # Interpolate Ez to match the reference grid
    # Ez_interp = np.interp(t_0, t, ez[-1])
    Ez_ref_interp = ez_0 #np.interp(x, x_0, Ez_0[-1]) #reference grid do not need to be interpolated since it's fixed
    
    # Compute L2 norm of the error
    errors[l] = np.linalg.norm(Ez_interp - Ez_ref_interp) / np.linalg.norm(Ez_ref_interp)
    print(f"dt = {dri:.2e}, runtime = {end-start:.2f} s, L2 error = {errors[l]:.2e}")

# %%
# Plot operation time vs. time step
plt.figure()
plt.plot(dt, operation_time, 'o-')
plt.xlabel('time step $dt$ [s]')
plt.ylabel('Operation time [s]')
# plt.xscale('log')  # Use logarithmic scale for better visualization
# plt.yscale('log')
plt.grid()
plt.title('Operation Time vs. Spatial Step')
plt.savefig('3d_comp_time_dx.png', bbox_inches="tight")
plt.show()

# Plot L2 error norm vs. time step
plt.figure()
plt.plot(dt, errors, 'o-')
plt.xlabel('spatial step $dt$ [s]')
plt.ylabel('L2 Norm of Error')
# plt.yscale('log')
# plt.xscale()
plt.grid()
plt.title('L2 Norm of Error vs. Spatial Step')
plt.savefig('3d_error_dx.png', bbox_inches="tight")
plt.show()
# %%
