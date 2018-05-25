import numpy as np
from scipy.integrate import odeint
import dynamics_functions as df

class GaussianState:
    # """ This is a class that stores information about the Gaussian state """

    def __init__(self, grid_space):

        # Description of the parameters of the grid
        self.size = grid_space.size()
        self.grid = grid_space
        self.dv = np.append(grid_space.dv(), grid_space.dv())
        self.volume_k = np.sum(grid_space.dv())

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

        self.unitary_rotation = np.block([[np.identity(self.size), \
                                           np.identity(self.size)], \
                                          [ -1j*np.identity(self.size), \
                                            1j*np.identity(self.size)]])

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
        solution = odeint(df.equations_imaginary_time_mf, variables_t, t,
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
        solution = odeint(df.equations_imaginary_time, variables_t, t,
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
        solution = odeint(df.equations_real_time, variables_t, t,
                          args=(self, hamiltonian), atol=abserr, rtol=relerr)

        # Overwrite the solution to its container
        # Reshape the solution to the form of its container
        self.amplitude = solution[-1][: 2 * self.size]
        self.gamma = solution[-1][2 * self.size:].reshape((2 * self.size, 2 * self.size))

    def evolve_real_time_mf(self, dt, hamiltonian):

        # ODE solver parameters: absolute and relevant error
        abserr = 1.0e-8
        relerr = 1.0e-6

        # Create the time samples for the output of the ODE solver.
        t = [0, dt]

        # Call the ODE solver that solves sumultaneously for the Coherent and Gaussian parts
        # ODE solver works with array of equations thus the equation for self.gamma needs to be reshaped

        variables_t = np.append(self.amplitude, self.gamma.reshape(self.gamma.size))
        solution = odeint(df.equations_real_time_mf, variables_t, t,
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

