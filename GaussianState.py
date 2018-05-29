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
        # self.phi describes the global phase of the wavefunction
        # self.lambda1 describes the correlation matrix forming the Lie group

        self.amplitude = np.zeros(2 * self.size, dtype=float)
        self.gamma = np.asarray(np.bmat([[np.identity(self.size), np.zeros((self.size, self.size))],
                                         [np.zeros((self.size, self.size)), np.identity(self.size)]]), dtype=float)
        self.phi = 0 + 1j * 0
        self.lambda1 = np.asarray(np.zeros((self.size, self.size)))


        # Construct the matrices of observables in the momentum basis:
        self.sigma = np.asarray(np.bmat([[np.zeros((self.size, self.size)), np.identity(self.size)],
                                         [-1 * np.identity(self.size), np.zeros((self.size, self.size))]]), dtype=float)
        self.identity = np.identity(2 * self.size, dtype=float)

        self.unitary_rotation = np.block([[np.identity(self.size), \
                                           np.identity(self.size)], \
                                          [ -1j*np.identity(self.size), \
                                            1j*np.identity(self.size)]])


    # EVOLUTION REAL TIME: GAUSSIAN STATE WITH PHASE

    def evolve_real_time_w_phase(self, dt, hamiltonian):

        # ODE solver parameters: absolute and relevant error
        abserr = 1.0e-8
        relerr = 1.0e-6

        # Create the time samples for the output of the ODE solver.
        t = [0, dt]

        # Call the ODE solver that solves sumultaneously for the Coherent and Gaussian parts
        # ODE solver works with array of equations thus the equation for self.gamma needs to be reshaped

        variables_t = np.append(self.amplitude, self.gamma.reshape(self.gamma.size)\
                                , self.phi, self.lambda1.reshape(self.lambda1.size))

        solution = odeint(df.equations_real_time_w_phase, variables_t, t,
                          args=(self, hamiltonian), atol=abserr, rtol=relerr)

        # Overwrite the solution to its container
        # Reshape the solution to the form of its container
        size = self.size
        ind_gamma = 2 * size
        ind_phi = ind_gamma + 4 * size * size
        ind_lambda = ind_phi + 1

        self.amplitude = solution[-1][: ind_gamma]
        self.gamma = solution[-1][ind_gamma:ind_phi].reshape((2 * size, 2 * size))
        self.phi = solution[-1][ind_phi:ind_phi + 1]
        self.lambda1 = solution[-1][ind_lambda:ind_lambda + size * size].reshape((size, size))

    # EVOLUTION REAL TIME: GAUSSIAN STATE

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


    # EVOLUTION IMAGINARY TIME: GAUSSIAN STATE

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


    # EVOLUTION REAL TIME: COHERENT STATE

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

    # EVOLUTION IMAGINARY TIME: COHERENT STATE

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



    # OBSERVABLES

    def get_PhononNumber(self):
        # returns the phonon number

        return 0.5 * np.dot(self.amplitude * self.amplitude, self.dv) \
               + 0.25 * np.dot(np.diagonal(self.gamma - self.identity), self.dv)

    def get_Zfactor(self):
        # returns the quasiparticle weight
        # is this correct?

        # construct a volume element such that the inverse would produce a correct result
        # which follows from the Taylor expansion
        dv_x_1 = np.outer(self.dv, np.ones(2* self.size))

        expon = np.exp(-1./2* (self.amplitude) @ np.linalg.inv(dv_x_1 * self.gamma + self.identity)
                       @ (self.dv * self.amplitude ))

        determ = np.sqrt(np.linalg.det(1./2 *(self.gamma + self.identity)))

        return expon / determ

    def get_energy(self, Ham):

        [aIBi, mB, n0, gBB] = Ham.Params
        gnum = Ham.gnum

        dv_x_dv = np.outer(self.dv, self.dv)
        dv_x_1 = np.outer(self.dv, np.ones(2 * self.size))

        energy_t =  gnum * n0 + Ham.h_frohlich @ (self.amplitude * self.dv) \
               + 1./4 *  self.amplitude @ (dv_x_dv * Ham.h_two_phon) @ self.amplitude \
               + 1./4 * np.trace((self.gamma - self.identity) @ (dv_x_1 * Ham.h_two_phon) )

        return energy_t

