import numpy as np
import polaron_functions as pf


class PolaronHamiltonianGaussian:
        # """ This is a class that stores information about the Hamiltonian"""

    def __init__(self, Gaussian_state, Params):

        # Params = [P, aIBi, mI, mB, n0, gBB]
        self.Params = Params

        self.gnum = pf.g(Gaussian_state.grid, *Params)
        self.Omega0_grid = pf.omega0(Gaussian_state.grid, *Params)
        self.Wk_grid = pf.Wk(Gaussian_state.grid, *Params)
        self.Wki_grid = 1 / self.Wk_grid
        self.kcos = pf.kcos_func(Gaussian_state.grid)

    def get_h_amplitude(self, Gaussian_state):

        [P, aIBi, mI, mB, n0, gBB] = self.Params
        # Here I need an original grid
        dv = Gaussian_state.grid.dV()

        # Split variables into x and p
        [x_t, p_t] = np.split(Gaussian_state.amplitude, 2)
        PB_t = Gaussian_state.get_PhononMomentum()

        h_x = 2 * self.gnum * np.sqrt(n0) * self.Wk_grid +\
            x_t * (self.Omega0_grid - self.kcos * (P - PB_t) / mI) +\
            self.gnum * self.Wk_grid * np.dot(self.Wk_grid, x_t * dv)
        h_p = p_t * (self.Omega0_grid - self.kcos * (P - PB_t) / mI) +\
            self.gnum * self.Wki_grid * np.dot(self.Wki_grid, p_t * dv)

        return np.append(h_x, h_p)

    def get_h_gamma(self, Gaussian_state):

        [P, aIBi, mI, mB, n0, gBB] = self.Params
        # Here I need an original grid
        dv = Gaussian_state.grid.dV()

        return Gaussian_state.gamma * 0.


# def phase_update(variables_t, t, coherent_state, hamiltonian):

#     [P, aIBi, mI, mB, n0, gBB] = hamiltonian.Params

#     # Here I need the original grid
#     dv = coherent_state.grid.dV()

#     # Split variables into x and p
#     [x_t, p_t] = np.split(coherent_state.amplitude, 2)
#     PB_t = coherent_state.get_PhononMomentum()

#     return hamiltonian.gnum * n0 + hamiltonian.gnum * np.sqrt(n0) * np.dot(hamiltonian.Wk_grid, x_t * dv) +\
#         (P**2 - PB_t**2) / (2 * mI)
