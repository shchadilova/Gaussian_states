import numpy as np
import polaron_functions as pf


class PolaronHamiltonianGaussian:
        # """ This is a class that stores information about the Hamiltonian"""

    def __init__(self, GS, Params):

        # Params = [aIBi, mB, n0, gBB]
        self.Params = Params

        self.gnum = pf.g_func(GS.grid, *Params)
        self.epsilon_grid = pf.epsilon_func(GS.grid, *Params)
        self.omega_grid = pf.omega_func(GS.grid, *Params)
        self.wk_grid = pf.wk_func(GS.grid, *Params)
        self.wk_inv_grid = pf.wk_inv_func(GS.grid, *Params)

        # creates the Frochlich part of the Hamiltonian
        self.h_frohlich = pf.h_frohlich_func(self.gnum, self.wk_grid, GS.size)

        # create 2ph Hamiltonian
        self.h_two_phon = pf.two_phonon_func(GS.grid, self.gnum, self.epsilon_grid, self.omega_grid, self.wk_grid, self.wk_inv_grid, GS.size)

    def get_h_amplitude(self, amplitude_t, gamma_t, Gaussian_state):

        return 2*(self.h_frohlich + 1./4*self.h_two_phon @ amplitude_t + 1./4 * amplitude_t @ self.h_two_phon)

    def get_h_gamma(self, amplitude_t, gamma_t, Gaussian_state):

        return self.h_two_phon

