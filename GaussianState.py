import numpy as np
from scipy.integrate import odeint


def equations_real_time(variables_t, t, GS, Ham):

    size = 2 * GS.size
    amplitude_t = variables_t[: size]
    gamma_t = variables_t[size:].reshape((size, size))

    amplitude_update = GS.sigma @ Ham.get_h_amplitude(amplitude_t, gamma_t, GS)

    right = (GS.sigma @ Ham.get_h_gamma(amplitude_t, gamma_t, GS)) @ gamma_t
    left = gamma_t @ (Ham.get_h_gamma(amplitude_t, gamma_t, GS) @ GS.sigma)

    gamma_update = np.reshape(right - left, size*size)


    return np.append(amplitude_update, gamma_update)


def equations_imaginary_time(variables_t, t, GS, Ham):

    # RESTORE MATRICES FROM ARRAYS
    size = 2 * GS.size
    amplitude_t = variables_t[: size]
    gamma_t = variables_t[size:].reshape((size, size))

    # CALCULATE RHS OF THE EQ OF MOTION: AMPLITUDE
    amplitude_update = - gamma_t @ Ham.get_h_amplitude(amplitude_t, gamma_t, GS)

    # CALCULATE RHS OF THE EQ OF MOTION: GAMMA
    right = (np.transpose(GS.sigma) @ Ham.get_h_gamma(amplitude_t, gamma_t, GS)) @ GS.sigma
    left = gamma_t @ (Ham.get_h_gamma(amplitude_t, gamma_t, GS) @ gamma_t)

    gamma_update = np.reshape(right - left, size*size)
    #print(right-left)

    return np.append(amplitude_update, gamma_update)


def equations_imaginary_time_mf(variables_t, t, GS, Ham):

    # RESTORE MATRICES FROM ARRAYS
    size = 2 * GS.size
    amplitude_t = variables_t[: size]
    gamma_t = variables_t[size:].reshape((size, size))

    # CALCULATE RHS OF THE EQ OF MOTION: AMPLITUDE
    amplitude_update = - gamma_t @ Ham.get_h_amplitude(amplitude_t, gamma_t, GS)

    # NO UPDATE FOR THE GAMMA EQ. OF MOTION!
    gamma_update = np.zeros(size*size)
    #print(right-left)

    return np.append(amplitude_update, gamma_update)


class GaussianState:
    # """ This is a class that stores information about the Gaussian state """

    def __init__(self, grid_space):

        # Description of the parameters of the grid
        self.size = grid_space.size()
        self.grid = grid_space
        self.dv = np.append(grid_space.dv(), grid_space.dv())

        # Variational parameters:
        # self.amplitude decribes the coherent part of the wavefunction
        # self.gamma decribes the gaussian part of the wavefunction
        # theire is no phase parameter since the equations of motion have to be modified to include the phase
        self.amplitude = np.zeros(2 * self.size, dtype=float)
        self.gamma = np.asarray(np.bmat([[np.identity(self.size), np.zeros((self.size, self.size))],
                                         [np.zeros((self.size, self.size)), np.identity(self.size)]]), dtype=float)

        # Construct the matrices of observables in the momentum basis:
        self.sigma = np.asarray(np.bmat([[np.zeros((self.size, self.size)), np.identity(self.size)],
                                         [-1 * np.identity(self.size), np.zeros((self.size, self.size))]]), dtype=float)
        self.identity = np.identity(2 * self.size, dtype=float)

    # EVOLUTION

    def evolve_imaginary_time_mf(self, dt, hamiltonian):

        # ODE solver parameters: absolute and relevant error
        abserr = 1.0e-8
        relerr = 1.0e-6

        # Create the time samples for the output of the ODE solver.
        t = [0, dt]

        # Call the ODE solver that solves sumultaneously for the Coherent and Gaussian parts
        # ODE solver works with array of equations thus the equation for self.gamma needs to be reshaped

        variables_t = np.append(self.amplitude, self.gamma.reshape(self.gamma.size))
        solution = odeint(equations_imaginary_time_mf, variables_t, t,
                          args=(self, hamiltonian), atol=abserr, rtol=relerr)

        # Overwrite the solution to its container
        # Reshape the solution to the form of its container
        self.amplitude = solution[-1][: 2 * self.size]
        self.gamma = solution[-1][2 * self.size:].reshape((2 * self.size, 2 * self.size))

    def evolve_imaginary_time(self, dt, hamiltonian):

        # ODE solver parameters: absolute and relevant error
        abserr = 1.0e-8
        relerr = 1.0e-6

        # Create the time samples for the output of the ODE solver.
        t = [0, dt]

        # Call the ODE solver that solves sumultaneously for the Coherent and Gaussian parts
        # ODE solver works with array of equations thus the equation for self.gamma needs to be reshaped

        variables_t = np.append(self.amplitude, self.gamma.reshape(self.gamma.size))
        solution = odeint(equations_imaginary_time, variables_t, t,
                          args=(self, hamiltonian), atol=abserr, rtol=relerr)

        # Overwrite the solution to its container
        # Reshape the solution to the form of its container
        self.amplitude = solution[-1][: 2 * self.size]
        self.gamma = solution[-1][2 * self.size:].reshape((2 * self.size, 2 * self.size))

    def evolve_real_time(self, dt, hamiltonian):

        # ODE solver parameters: absolute and relevant error
        abserr = 1.0e-8
        relerr = 1.0e-6

        # Create the time samples for the output of the ODE solver.
        t = [0, dt]

        # Call the ODE solver that solves sumultaneously for the Coherent and Gaussian parts
        # ODE solver works with array of equations thus the equation for self.gamma needs to be reshaped

        variables_t = np.append(self.amplitude, self.gamma.reshape(self.gamma.size))
        solution = odeint(equations_real_time, variables_t, t,
                          args=(self, hamiltonian), atol=abserr, rtol=relerr)

        # Overwrite the solution to its container
        # Reshape the solution to the form of its container
        self.amplitude = solution[-1][: 2 * self.size]
        self.gamma = solution[-1][2 * self.size:].reshape((2 * self.size, 2 * self.size))

    # OBSERVABLES

    def get_PhononNumber(self):
        # returns the phonon number

        return 0.5 * np.dot(self.amplitude * self.amplitude, self.dv) + 0.25 * np.dot(np.diagonal(self.gamma - self.identity), self.dv)

    def get_Zfactor(self):
        # returns quasiparticle weight

        expon = np.exp(-1./2* (self.amplitude) @ np.linalg.inv(self.gamma + self.identity)
                       @ (self.amplitude * self.dv))
        determ = np.sqrt(np.linalg.det(1./2 *(self.gamma + self.identity)))

        return expon / determ

    def get_energy(self, Ham):

        energy_t =  Ham.gnum + Ham.h_frohlich @ (self.amplitude *self.dv) \
               + 1./4 * (self.amplitude) @ Ham.h_two_phon @ (self.amplitude *self.dv) \
               + 1./4 * np.diagonal(Ham.h_two_phon @ (self.gamma - self.identity)) @ self.dv

        return energy_t