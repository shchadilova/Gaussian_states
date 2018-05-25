import numpy as np

# ---- BASIC FUNCTIONS ----


def eB(k, mB):
    # free boson dispersion relation

    return k**2 / (2 * mB)


def w(k, gBB, mB, n0):
    # Bogoliubov phonon dispersion relation

    return np.sqrt(eB(k, mB) * (eB(k, mB) + 2 * gBB * n0))


# ---- COMPOSITE FUNCTIONS ----


def g_func(grid_space, aIBi, mB, n0, gBB):
    # gives bare interaction strength constant

    k_max = grid_space.arrays['k'][-1]
    mR = mB

    return 1 / ((mR / (2 * np.pi)) * aIBi - (mR / np.pi**2) * k_max)

def epsilon_func(grid_space, aIBi, mB, n0, gBB):
    # gives the disprsion relation of free bosons on the grid
    names = list(grid_space.arrays.keys())
    functions_omega = [lambda k: eB(k, mB)]

    return grid_space.function_prod(names, functions_omega)

def omega_func(grid_space, aIBi, mB, n0, gBB):
    # gives the disprsion relation of Bogoliubov phonons on the grid

    names = list(grid_space.arrays.keys())
    functions_omega = [lambda k: w(k, gBB, mB, n0)]

    return grid_space.function_prod(names, functions_omega)

def wk_func(grid_space, aIBi, mB, n0, gBB):
    # calculates interaction vertex

    names = list(grid_space.arrays.keys())
    functions_wk = [lambda k: np.sqrt(eB(k, mB) / w(k, gBB, mB, n0))]
    jacobian =  grid_space.jacobian()

    return grid_space.function_prod(names, functions_wk)*jacobian


def wk_inv_func(grid_space, aIBi, mB, n0, gBB):
    # calculates interaction vertex

    names = list(grid_space.arrays.keys())
    functions_wk = [lambda k: np.sqrt(w(k, gBB, mB, n0) / eB(k, mB))]
    jacobian = grid_space.jacobian()

    return grid_space.function_prod(names, functions_wk)*jacobian


def h_frohlich_func(gnum, wk_grid, size):
    # Creates the linear part of the Hamiltonian (Frohlich)

    return gnum * np.append(wk_grid, np.zeros(size))


def two_phonon_func(volume_k, gnum,  omega_grid, wk_grid, wk_inv_grid, size):
    # Creates the quadratic part of the Hamiltonian (2ph)

    B1 = gnum * np.outer(wk_grid, wk_grid )
    B2 = gnum * np.outer(wk_inv_grid, wk_inv_grid)

    # Kinetic energy scales with the volume of the system, keep this volume dependence explicitly
    # Note: this part should be changed in case of the finite mass corrections!
    A = 1./volume_k * np.diag(omega_grid)

    return np.block([[A + B1, np.zeros((size, size))], [np.zeros((size, size)), A + B2]])
