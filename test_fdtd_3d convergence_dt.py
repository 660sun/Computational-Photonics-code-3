import time
import numpy as np
from function_headers_fdtd import fdtd_3d
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# Constants
c = 2.99792458e8
mu0 = 4 * np.pi * 1e-7
eps0 = 1 / (mu0 * c**2)
Z0 = np.sqrt(mu0 / eps0)

# Simulation parameters
initial_dt = 0.5e-12
freq = 500e12
tau = 1e-15
source_width = 2 * initial_dt

# Specify the grid size to test
Nx, Ny, Nz = (99, 101, 3)

dt_factors = np.linspace(0.1, 1.0, 10)
dt_values = dt_factors * initial_dt

errors_grid = np.zeros(len(dt_values))
operation_times = np.zeros(len(dt_values))

# Additional parameters for fdtd_3d
field_component = 'ez'
z_ind = 0
output_step = 10

dr = 30e-9

eps_rel = np.ones((Nx, Ny, Nz))

# Grid midpoints
midx = Nx // 2
midy = Ny // 2
midz = Nz // 2

# Current distributions
jx, jy = np.zeros((Nx, Ny, Nz)), np.zeros((Nx, Ny, Nz))
jz = np.exp(-((np.arange(Nx).reshape(-1, 1, 1) - midx) ** 2 +
              (np.arange(Ny).reshape(1, -1, 1) - midy) ** 2) / source_width ** 2)

for j, dt in enumerate(dt_values):
    try:
        start = time.time()
        hx, ez, t = fdtd_3d(eps_rel, dr, dt, freq, tau, jx, jy, jz, field_component, z_ind, output_step)
        end = time.time()
        operation_times[j] = end - start

        # Interpolate Ez to compare with reference
        interp_func = RegularGridInterpolator((t,), ez[:, midx, midy], bounds_error=False, fill_value=0)
        Ez_interp = interp_func(t)

        # Compute L2 norm of the error
        ez_0 = ez[:, midx, midy]
        errors_grid[j] = np.linalg.norm(Ez_interp - ez_0) / np.linalg.norm(ez_0)

        print(f"Grid size ({Nx}, {Ny}, {Nz}), dt = {dt:.2e}, runtime = {operation_times[j]:.2f} s, L2 error = {errors_grid[j]:.2e}")

    except ValueError as e:
        print(f"Simulation failed for grid size ({Nx}, {Ny}, {Nz}), dt = {dt:.2e} with error: {e}")
        errors_grid[j] = np.nan
        operation_times[j] = np.nan
    except IndexError as e:
        print(f"IndexError: {e}")
        print(f"Grid size ({Nx}, {Ny}, {Nz}), dt = {dt:.2e}, midx: {midx}, midy: {midy}, midz: {midz}")

# Plot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(dt_values, errors_grid, 'o-', label=f'Grid size ({Nx}, {Ny}, {Nz})')
plt.xlabel('Time Step (s)')
plt.ylabel('L2 Norm of Error')
plt.title(f'L2 Norm of Error vs. Time Step for Grid Size ({Nx}, {Ny}, {Nz})')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(dt_values, operation_times, 'o-', label=f'Grid size ({Nx}, {Ny}, {Nz})')
plt.xlabel('Time Step (s)')
plt.ylabel('Runtime (s)')
plt.title(f'Runtime vs. Time Step for Grid Size ({Nx}, {Ny}, {Nz})')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()