import numpy as np
from polaron_functions import kcos_func
from scipy.integrate import odeint
from PolaronHamiltonian import amplitude_update, phase_update


class GaussianState:
    # """ This is a class that stores information about the Gaussian state """

    def __init__(self, grid_space):

        size = grid_space.size()
        # This is a part of the coherent state
        self.amplitude = np.zeros(2 * size, dtype=float)
        self.gamma = np.zeros(2 * size, 2 * size, dtype=float)
        self.phase = 0
        self.grid = grid_space

        self.dV = np.append(grid_space.dV(), grid_space.dV())

        # Construct the matrices of observables in the momentum basis:
        # self.kcos - momentum operator in 2D Spherical coordinates
        self.kcos = np.append(kcos_func(self.grid), kcos_func(self.grid))
        self.sigma = np.bmat([[np.zeros(size, size), np.identity(size)],
                              [np.zeros(size, size), np.identity(size)]])

    # EVOLUTION

    def evolve(self, dt, hamiltonian):

        # ODE solver parameters: absolute and relevant error
        abserr = 1.0e-8
        relerr = 1.0e-6

        # Create the time samples for the output of the ODE solver.
        t = [0, dt]

        # Call the ODE solver.
        amplitude_sol = odeint(amplitude_update, self.amplitude, t, args=(self, hamiltonian),
                               atol=abserr, rtol=relerr)
        gamma_sol = odeint(gamma_update, self.gamma, t, args=(self, hamiltonian),
                           atol=abserr, rtol=relerr)
        # phase_sol = odeint(phase_update, self.phase, t, args=(self, hamiltonian),
        #                   atol=abserr, rtol=relerr)

        # Overrite the solution to its container
        self.amplitude = amplitude_sol[-1]
        self.gamma = gamma_sol[-1]
        #self.phase = phase_sol[-1]

    # OBSERVABLES

    def get_PhononNumber(self):

        coherent_amplitude = self.amplitude
        return 0.5 * np.dot(coherent_amplitude * coherent_amplitude, self.dV) +\
            0.5 * np.dot(np.diagonal(self.Gamma), self.dV)

    def get_PhononMomentum(self):

        coherent_amplitude = self.amplitude
        return 0.5 * np.dot(self.kcos, coherent_amplitude * coherent_amplitude * self.dV) +\
            0.5 * np.dot(self.kcos * np.diagonal(self.Gamma), self.dV)

    # def get_DynOverlap(self):
    #     # dynamical overlap/Ramsey interferometry signal
    #     # this parn needs a substantial modfication
    #     NB_vec = self.get_PhononNumber()
    #     exparg = -1j * self.phase - (1 / 2) * NB_vec
    #     return np.exp(exparg)
