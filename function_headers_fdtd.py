'''Homework 3, Computational Photonics, SS 2020:  FDTD method.
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import time


def fdtd_1d(eps_rel, dx, time_span, source_frequency, source_position, source_pulse_length):
    '''Computes the temporal evolution of a pulsed excitation using the
    1D FDTD method. The temporal center of the pulse is placed at a
    simulation time of 3*source_pulse_length. The origin x=0 is in the
    center of the computational domain. All quantities have to be
    specified in SI units.

    Arguments
    ---------
        eps_rel : 1d-array
            Rel. permittivity distribution within the computational domain.
        dx : float
            Spacing of the simulation grid (please ensure dx <= lambda/20).
        time_span : float
            Time span of simulation.
        source_frequency : float
            Frequency of current source.
        source_position : float
            Spatial position of current source.
        source_pulse_length :
            Temporal width of Gaussian envelope of the source.

    Returns
    -------
        Ez : 2d-array
            Z-component of E(x,t) (each row corresponds to one time step)
        Hy : 2d-array
            Y-component of H(x,t) (each row corresponds to one time step)
        x  : 1d-array
            Spatial coordinates of the field output
        t  : 1d-array
            Time of the field output
    '''
    
    # basic parameters
    c = 2.99792458e8 # speed of light [m/s]
    mu0 = 4*np.pi*1e-7 # vacuum permeability [Vs/(Am)]
    eps0 = 1/(mu0*c**2) # vacuum permittivity [As/(Vm)]
    Z0 = np.sqrt(mu0/eps0) # vacuum impedance [Ohm]

    # time step
    dt = dx / (2 * c)
    Nt = int(round(time_span / dt)) + 1
    t = np.linspace(0, time_span, Nt)

    # construction of matrices
    Ez = np.zeros((Nt, len(eps_rel)))
    Hy = np.zeros((Nt, len(eps_rel)))
    jz = np.zeros((Nt, len(eps_rel) - 1))

    # spatial coordinates of the fields
    x = np.linspace(-(len(eps_rel) - 1)/2*dx, (len(eps_rel) - 1)/2*dx, len(eps_rel))

    # source matrix
    for n in range(Nt):
        jz[n, source_position] = np.exp(-(((n + 0.5)*dt - 3*source_pulse_length)/source_pulse_length)**2) * np.cos(2*np.pi*source_frequency * (n + 0.5)*dt)
    
    # main loop
    for n in range(1, Nt):
        for i in range(1, len(eps_rel) - 1):
            Ez[n, i] = Ez[n-1, i] + (Hy[n-1, i] - Hy[n-1, i-1]) * 1 / ( eps0 * eps_rel[i] ) * dt / dx - jz[n-1, i] * dt / ( eps0 * eps_rel[i] )
        for i in range(len(eps_rel) - 1):
            Hy[n, i] = Hy[n-1, i] + (Ez[n, i+1] - Ez[n, i]) * 1 / mu0 * dt / dx

    # postprocessing - interpolation of output
    for n in range(1, len(Ez)):
        Hy[n, 0] = 0.5 * (Hy[n, 0] + Hy[n-1, 0])
        Hy[n, -1] = 0.5 * (Hy[n, -2] + Hy[n-1, -2])
        for i in range(1, len(eps_rel)-1):
            Hy[n, i] = 0.25 * (Hy[n, i] + Hy[n, i-1] + Hy[n-1, i] + Hy[n-1, i-1])

    return Ez, Hy, x, t

def fdtd_1d_t(eps_rel, dx, dt, Nt, source_frequency, source_position, source_pulse_length):

    # basic parameters
    c = 2.99792458e8 # speed of light [m/s]
    mu0 = 4*np.pi*1e-7 # vacuum permeability [Vs/(Am)]
    eps0 = 1/(mu0*c**2) # vacuum permittivity [As/(Vm)]

    # construction of matrices
    Ez = np.zeros((Nt, len(eps_rel)))
    Hy = np.zeros((Nt, len(eps_rel)))
    jz = np.zeros((Nt, len(eps_rel) - 1))

    # spatial coordinates of the fields
    x = np.linspace(-(len(eps_rel) - 1)/2*dx, (len(eps_rel) - 1)/2*dx, len(eps_rel))

    # source matrix
    for n in range(Nt):
        jz[n, source_position] = np.exp(-(((n + 0.5)*dt - 3*source_pulse_length)/source_pulse_length)**2) * np.cos(2*np.pi*source_frequency * (n + 0.5)*dt)

    # main loop
    for n in range(1, Nt):
        for i in range(1, len(eps_rel) - 1):
            Ez[n, i] = Ez[n-1, i] + (Hy[n-1, i] - Hy[n-1, i-1]) * 1 / ( eps0 * eps_rel[i] ) * dt / dx - jz[n-1, i] * dt / ( eps0 * eps_rel[i] )
        for i in range(len(eps_rel) - 1):
            Hy[n, i] = Hy[n-1, i] + (Ez[n, i+1] - Ez[n, i]) * 1 / mu0 * dt / dx

    # postprocessing - interpolation of output
    for n in range(1, len(Ez)):
        Hy[n, 0] = 0.5 * (Hy[n, 0] + Hy[n-1, 0])
        Hy[n, -1] = 0.5 * (Hy[n, -2] + Hy[n-1, -2])
        for i in range(1, len(eps_rel)-1):
            Hy[n, i] = 0.25 * (Hy[n, i] + Hy[n, i-1] + Hy[n-1, i] + Hy[n-1, i-1])

    return Ez, Hy, x


def fdtd_3d(eps_rel, dr, time_span, freq, tau, jx, jy, jz,
            field_component, z_ind, output_step):
    '''Computes the temporal evolution of a pulsed spatially extended current source using the 3D FDTD method. Returns z-slices of the selected field at the given z-position every output_step time steps. The pulse is centered at a simulation time of 3*tau. All quantities have to be specified in SI units.

    Arguments
    ---------
        eps_rel: 3d-array
            Rel. permittivity distribution within the computational domain.
        dr: float
            Grid spacing (please ensure dr<=lambda/20).
        time_span: float
            Time span of simulation.
        freq: float
            Center frequency of the current source.
        tau: float
            Temporal width of Gaussian envelope of the source.
        jx, jy, jz: 3d-array
            Spatial density profile of the current source.
        field_component : str
            Field component which is stored (one of 'ex','ey','ez','hx','hy','hz').
        z_index: int
            Z-position of the field output.
        output_step: int
            Number of time steps between field outputs.

    Returns
    -------
        F: 3d-array
            Z-slices of the selected field component at the z-position specified by z_ind stored every output_step         time steps (time varies along the first axis).
        t: 1d-array
            Time of the field output.
    '''
    
    # basic parameters
    c = 2.99792458e8 # speed of light [m/s]
    mu0 = 4*np.pi*1e-7 # vacuum permeability [Vs/(Am)]
    eps0 = 1/(mu0*c**2) # vacuum permittivity [As/(Vm)]
    Z0 = np.sqrt(mu0/eps0) # vacuum impedance [Ohm]

    # time step
    dt = dr / (2 * c)
    Nt = int(round(time_span / dt)) + 1
    t = np.linspace(0, time_span, Nt)
       
    # construction of matrices
    ex = np.zeros((eps_rel.shape[0], eps_rel.shape[1], eps_rel.shape[2]))
    ey = np.zeros((eps_rel.shape[0], eps_rel.shape[1], eps_rel.shape[2]))
    ez = np.zeros((eps_rel.shape[0], eps_rel.shape[1], eps_rel.shape[2]))
    hx = np.zeros((eps_rel.shape[0], eps_rel.shape[1], eps_rel.shape[2]))
    hy = np.zeros((eps_rel.shape[0], eps_rel.shape[1], eps_rel.shape[2]))
    hz = np.zeros((eps_rel.shape[0], eps_rel.shape[1], eps_rel.shape[2]))

    F1 = []
    F2 = []
    Ex = []
    Ey = []
    Ez = []
    Hx = []
    Hy = []
    Hz = []

    # Main loop
    for n in range(0, Nt):
        # Add perfect electric conductor boundary conditions
        ex[:, 0, :] = 0
        ex[:, -1, :] = 0
        ex[:, :, 0] = 0
        ex[:, :, -1] = 0
        ey[0, :, :] = 0
        ey[-1, :, :] = 0
        ey[:, :, 0] = 0
        ey[:, :, -1] = 0
        ez[0, :, :] = 0
        ez[-1, :, :] = 0
        ez[:, 0, :] = 0
        ez[:, -1, :] = 0
        hx[0, :, :] = 0
        hx[-1, :, :] = 0
        hy[:, 0, :] = 0
        hy[:, -1, :] = 0
        hz[:, :, 0] = 0
        hz[:, :, -1] = 0

        
        # Update electric fields
        ex = ex + dt / (eps0 * eps_rel) * ((hz - np.roll(hz, 1, axis=1)) - (hy - np.roll(hy, 1, axis=2))) / dr - jx * np.cos(2 * np.pi * freq * (n + 0.5) * dt) * np.exp(-(((n + 0.5) * dt - 3 * tau) / tau) ** 2) * dt / (eps0  * eps_rel)

        ey = ey + dt / (eps0  * eps_rel) * ((hx - np.roll(hx, 1, axis=2)) - (hz - np.roll(hz, 1, axis=0))) / dr - jy * np.cos(2 * np.pi * freq * (n + 0.5) * dt) * np.exp(-(((n + 0.5) * dt - 3 * tau) / tau) ** 2) * dt / (eps0  * eps_rel)

        ez = ez + dt / (eps0) * ((hy - np.roll(hy, 1, axis=0)) - (hx - np.roll(hx, 1, axis=1))) / dr - jz * np.cos(2 * np.pi * freq * (n + 0.5) * dt) * np.exp(-(((n + 0.5) * dt - 3 * tau) / tau) ** 2) * dt / (eps0)

        # Update magnetic fields
        hx = hx - dt / mu0 * ((ey - np.roll(ey, -1, axis=2)) - (ez - np.roll(ez, -1, axis=1))) / dr

        hy = hy - dt / mu0 * ((ez - np.roll(ez, -1, axis=0)) - (ex - np.roll(ex, -1, axis=2))) / dr

        hz = hz - dt / mu0 * ((ex - np.roll(ex, -1, axis=1)) - (ey - np.roll(ey, -1, axis=0))) / dr

        # Save the field components for a specific z-plane index `z_ind`
        # F1.append(hx[:, :, z_ind])
        # F2.append(ez[:, :, z_ind])

        # Save the field components at a specific time
        Ex.append(ex)
        Ey.append(ey)
        Ez.append(ez)
        Hx.append(hx)
        Hy.append(hy)
        Hz.append(hz)

    
    # F1 = np.array(F1)
    # F2 = np.array(F2)
    Ex = np.array(Ex)
    Ey = np.array(Ey)
    Ez = np.array(Ez)
    Hx = np.array(Hx)
    Hy = np.array(Hy)
    Hz = np.array(Hz)

    # Postprocessing - interpolation of output
    Ex = 0.5 * (Ex + np.roll(Ex, 1, axis=1))

    Ey = 0.5 * (Ey + np.roll(Ey, 1, axis=2))
    
    Ez = 0.5 * (Ez + np.roll(Ez, 1, axis=3))

    Hx = 0.125 * (Hx + np.roll(Hx, 1, axis=2) + Hx + np.roll(Hx, 1, axis=3) + np.roll(np.roll(Hx, 1, axis=2), 1, axis=3) + np.roll((Hx + np.roll(Hx, 1, axis=2) + Hx + np.roll(Hx, 1, axis=3) + np.roll(np.roll(Hx, 1, axis=2), 1, axis=3)), 1, axis=0))
        
    Hy = 0.125 * (Hy + np.roll(Hy, 1, axis=1) + Hy + np.roll(Hy, 1, axis=3) + np.roll(np.roll(Hy, 1, axis=1), 1, axis=3) + np.roll((Hy + np.roll(Hy, 1, axis=1) + Hy + np.roll(Hy, 1, axis=3) + np.roll(np.roll(Hy, 1, axis=1), 1, axis=3)), 1, axis=0))
        
    Hz = 0.125 * (Hz + np.roll(Hz, 1, axis=1) + Hz + np.roll(Hz, 1, axis=2) + np.roll(np.roll(Hz, 1, axis=1), 1, axis=2) + np.roll((Hz + np.roll(Hz, 1, axis=1) + Hz + np.roll(Hz, 1, axis=2) + np.roll(np.roll(Hz, 1, axis=1), 1, axis=2)), 1, axis=0))

    F1 = np.zeros((len(t), eps_rel.shape[0], eps_rel.shape[1]))
    F2 = np.zeros((len(t), eps_rel.shape[0], eps_rel.shape[1]))
    if field_component == 'hx' or 'ez':
            
            for n in range(0, len(t)):
                F1[n, :, :] = Hx[n, :, :, z_ind]
                F2[n, :, :] = Ez[n, :, :, z_ind]

            F1 = F1[::output_step, :, :]
            F2 = F2[::output_step, :, :]

    t = t[::output_step]

    return F1, F2, t


class Fdtd1DAnimation(animation.TimedAnimation):
    '''Animation of the 1D FDTD fields.

    Based on https://matplotlib.org/examples/animation/subplots.html

    Arguments
    ---------
    x : 1d-array
        Spatial coordinates
    t : 1d-array
        Time
    x_interface : float
        Position of the interface (default: None)
    step : float
        Time step between frames (default: 2e-15/25)
    fps : int
        Frames per second (default: 25)
    Ez: 2d-array
        Ez field to animate (each row corresponds to one time step)
    Hy: 2d-array
        Hy field to animate (each row corresponds to one time step)
    '''

    def __init__(self, x, t, Ez, Hy, x_interface=None, step=2e-15/25, fps=25):
        # constants
        c = 2.99792458e8 # speed of light [m/s]
        mu0 = 4*np.pi*1e-7 # vacuum permeability [Vs/(Am)]
        eps0 = 1/(mu0*c**2) # vacuum permittivity [As/(Vm)]
        Z0 = np.sqrt(mu0/eps0) # vacuum impedance [Ohm]
        self.Ez = Ez
        self.Z0Hy = Z0*Hy
        self.x = x
        self.ct = c*t

        # index step between consecutive frames
        self.frame_step = int(round(step/(t[1] - t[0])))

        # set up initial plot
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        vmax = max(np.max(np.abs(Ez)),np.max(np.abs(Hy))*Z0)*1e6
        fig, ax = plt.subplots(2,1, sharex=True, gridspec_kw={'hspace': 0.4})
        self.line_E, = ax[0].plot(x*1e6, self.E_at_step(0),
                         color=colors[0], label='$\\Re\\{E_z\\}$')
        self.line_H, = ax[1].plot(x*1e6, self.H_at_step(0),
                         color=colors[1], label='$Z_0\\Re\\{H_y\\}$')
        if x_interface is not None:
            for a in ax:
                a.axvline(x_interface*1e6, ls='--', color='k')
        for a in ax:
            a.set_xlim(x[[0,-1]]*1e6)
            a.set_ylim(np.array([-1.1, 1.1])*vmax)
        ax[0].set_ylabel('$\\Re\\{E_z\\}$ [µV/m]')
        ax[1].set_ylabel('$Z_0\\Re\\{H_y\\}$ [µV/m]')
        self.text_E = ax[0].set_title('')
        self.text_H = ax[1].set_title('')
        ax[1].set_xlabel('$x$ [µm]')
        super().__init__(fig, interval=1000/fps, blit=False)

    def E_at_step(self, n):
        return self.Ez[n,:].real*1e6

    def H_at_step(self, n):
        return self.Z0Hy[n,:].real*1e6

    def new_frame_seq(self):
        return iter(range(0, self.ct.size, self.frame_step))

    def _init_draw(self):
        self.line_E.set_ydata(self.x*np.nan)
        self.line_H.set_ydata(self.x*np.nan)
        self.text_E.set_text('')
        self.text_E.set_text('')

    def _draw_frame(self, framedata):
        i = framedata
        self.line_E.set_ydata(self.E_at_step(i))
        self.line_H.set_ydata(self.H_at_step(i))
        self.text_E.set_text(
                'Electric field, $ct = {0:1.2f}$µm'.format(self.ct[i]*1e6))
        self.text_H.set_text(
                'Magnetic field, $ct = {0:1.2f}$µm'.format(self.ct[i]*1e6))
        self._drawn_artists = [self.line_E, self.line_H,
                               self.text_E, self.text_H]


class Fdtd3DAnimation(animation.TimedAnimation):
    '''Animation of a 3D FDTD field.

    Based on https://matplotlib.org/examples/animation/subplots.html

    Arguments
    ---------
    x, y : 1d-array
        Coordinate axes.
    t : 1d-array
        Time
    field: 3d-array
        Slices of the field to animate (the time axis is assumed to be the first axis of the array)
    titlestr : str
        Plot title.
    cb_label : str
        Colrbar label.
    rel_color_range: float
        Range of the colormap relative to the full scale of the field magnitude.
    fps : int
        Frames per second (default: 25)
    '''

    def __init__(self, x, y, t, field, titlestr, cb_label, rel_color_range, fps=25):
        # constants
        c = 2.99792458e8 # speed of light [m/s]
        self.ct = c*t

        self.fig = plt.figure()
        self.F = field
        color_range = rel_color_range*np.max(np.abs(field))
        phw = 0.5*(x[1] - x[0]) # pixel half-width
        extent = ((x[0] - phw)*1e6, (x[-1] + phw)*1e6,
                  (y[-1] + phw)*1e6, (y[0] - phw)*1e6)
        self.mapable = plt.imshow(self.F[0,:,:].real.T,
                                  vmin=-color_range, vmax=color_range,
                                  extent=extent)
        cb = plt.colorbar(self.mapable)
        plt.gca().invert_yaxis()
        self.titlestr = titlestr
        self.text = plt.title('')
        plt.xlabel('x position [µm]')
        plt.ylabel('y position [µm]')
        cb.set_label(cb_label)
        super().__init__(self.fig, interval=1000/fps, blit=False)

    def new_frame_seq(self):
        return iter(range(self.ct.size))

    def _init_draw(self):
        self.mapable.set_array(np.nan*self.F[0, :, :].real.T)
        self.text.set_text('')

    def _draw_frame(self, framedata):
        i = framedata
        self.mapable.set_array(self.F[i, :, :].real.T)
        self.text.set_text(self.titlestr
                           + ', $ct$ = {0:1.2f}µm'.format(self.ct[i]*1e6))
        self._drawn_artists = [self.mapable, self.text]


class Timer(object):
    '''Tic-toc timer.
    '''
    def __init__(self):
        '''Initializer.
        Stores the current time.
        '''
        self._tic = time.time()

    def tic(self):
        '''Stores the current time.
        '''
        self._tic = time.time()

    def toc(self):
        '''Returns the time in seconds that has elapsed since the last call to tic().
        '''
        return time.time() - self._tic


