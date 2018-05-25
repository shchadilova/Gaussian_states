import numpy as np
import polaron_functions as pf


class PolaronHamiltonianGaussian:
        # """ This is a class that stores information about the Hamiltonian"""

    def __init__(self, GS, Params):

        # Params = [aIBi, mB, n0, gBB]
        self.Params = Params
        size = GS.size

        self.gnum = pf.g_func(GS.grid, *Params)
        self.epsilon_grid = pf.epsilon_func(GS.grid, *Params)
        self.omega_grid = pf.omega_func(GS.grid, *Params)
        self.wk_grid = pf.wk_func(GS.grid, *Params)
        self.wk_inv_grid = pf.wk_inv_func(GS.grid, *Params)

        # creates the Frochlich part of the Hamiltonian
        self.h_frohlich = pf.h_frohlich_func(self.gnum, self.wk_grid, size)

        # create 2ph Hamiltonian
        self.h_two_phon = pf.two_phonon_func(GS.volume_k, self.gnum, self.omega_grid, self.wk_grid, self.wk_inv_grid, size)

        # create Hamiltonian to evolve the pase of the wave function
        h_omega_temp = 1. / 2 * np.conjugate(np.transpose(GS.unitary_rotation)) @ self.h_two_phon @ GS.unitary_rotation

        # the resulting hamiltonian is real, so take the real part
        self.h_omega = np.real(h_omega_temp[0:size, 0:size])
        self.h_omega_bar = np.real(h_omega_temp[size:2 * size, 0:size])

    def get_h_amplitude(self, amplitude_t, gamma_t, GS):

        return 2 * self.h_frohlich + 1./2 * self.h_two_phon @ (amplitude_t * GS.dv)

    def get_h_gamma(self, amplitude_t, gamma_t, GS):

        return self.h_two_phon




