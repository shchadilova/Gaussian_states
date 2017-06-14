import numpy as np
from polaron_functions import kcos_func
from scipy.integrate import odeint


def amplitude_update(variables_t, t, Gaussian_state, Hamiltonian):
    # This is a generic equation of motion for the Gaussian state
    # All the details of Hamiltonian are in the get_h_amplitude
    return Gaussian_state.sigma @ Hamiltonian.get_h_amplitude(Gaussian_state)


def gamma_update(variables_t, t, Gaussian_state, Hamiltonian):
    # This is a generic equation of motion for the Gaussian state
    # All the details of Hamiltonian are in the get_h_amplitude
    right = (Gaussian_state.sigma @ Hamiltonian.get_h_gamma(Gaussian_state)) @ Gaussian_state.gamma
    left = Gaussian_state.gamma @ (Hamiltonian.get_h_gamma(Gaussian_state) @ Gaussian_state.sigma)
    return np.reshape(right - left, right.size)


class GaussianState:
    # """ This is a class that stores information about the Gaussian state """

    def __init__(self, grid_space):

        # Description of the parameters of the grid
        self.size = grid_space.size()
        self.grid = grid_space
        self.dV = np.append(grid_space.dV(), grid_space.dV())

        # Variational parameters:
        # self.amplitude decribes the coherent part of the wavefunction
        # self.gamma decribes the coherent part of the wavefunction
        # self.phase describes the global phase of the wavefunction
        self.amplitude = np.zeros(2 * self.size, dtype=float)
        self.gamma = np.asarray(np.bmat([[np.identity(self.size), np.zeros((self.size, self.size))],
                                         [np.zeros((self.size, self.size)), np.identity(self.size)]]))
        self.phase = 0

        # Construct the matrices of observables in the momentum basis:
        # self.kcos - momentum operator in 2D Spherical coordinates
        self.kcos = np.append(kcos_func(self.grid), kcos_func(self.grid))
        self.sigma = np.asarray(np.bmat([[np.zeros((self.size, self.size)), np.identity(self.size)],
                                         [-1 * np.identity(self.size), np.zeros((self.size, self.size))]]))
        self.identity = np.identity(2 * self.size)

    # EVOLUTION

    def evolve(self, dt, hamiltonian):

        # ODE solver parameters: absolute and relevant error
        abserr = 1.0e-8
        relerr = 1.0e-6

        # Create the time samples for the output of the ODE solver.
        t = [0, dt]

        # Call the ODE solver
        # ODE solver works with array of equations thus the equation for self.gamma needs to be reshaped
        amplitude_sol = odeint(amplitude_update, self.amplitude, t, args=(self, hamiltonian),
                               atol=abserr, rtol=relerr)
        gamma_sol = odeint(gamma_update, self.gamma.reshape(self.gamma.size), t,
                           args=(self, hamiltonian), atol=abserr, rtol=relerr)
        # phase_sol = odeint(phase_update, self.phase, t, args=(self, hamiltonian),
        #                   atol=abserr, rtol=relerr)

        # Overrite the solution to its container
        # Reshape the solution to the form of its container
        self.amplitude = amplitude_sol[-1]
        self.gamma = gamma_sol[-1].reshape((2 * self.size, 2 * self.size))
        #self.phase = phase_sol[-1]

    # OBSERVABLES

    def get_PhononNumber(self):

        return 0.5 * np.dot(self.amplitude * self.amplitude, self.dV) +\
            0.5 * np.dot(np.diagonal(self.gamma - self.identity), self.dV)

    def get_PhononMomentum(self):

        return 0.5 * np.dot(self.kcos, self.amplitude * self.amplitude * self.dV) +\
            0.5 * np.dot(self.kcos * np.diagonal(self.gamma - self.identity), self.dV)

    # def get_DynOverlap(self):
    #     # dynamical overlap/Ramsey interferometry signal
    #     # this parn needs a substantial modfication
    #     NB_vec = self.get_PhononNumber()
    #     exparg = -1j * self.phase - (1 / 2) * NB_vec
    #     return np.exp(exparg)
