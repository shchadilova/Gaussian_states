from timeit import default_timer as timer
import numpy as np
import os
import Grid
import GaussianState
import PolaronHamiltonianGaussian

import sys

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 12, 'text.usetex': True})


# ----------------------------------------
# Grid Initialization
# ----------------------------------------
k_max = 1
k_step = 0.5

grid_space = Grid.Grid("3D")
grid_space.init1d('k', k_step, k_max, k_step)

#test grid
#print(grid_space.return_array1d('k'))
#print(grid_space.jacobian())

# ----------------------------------------
# Initialization of the Gaussian state
# ----------------------------------------
gs = GaussianState.GaussianState(grid_space)

# ----------------------------------------
# Initialization PolaronHamiltonian
# ----------------------------------------
# this code is for infinime mass impurity
# thus mI and P are not parameters
mB = 1
n0 = 1
gBB = (4 * np.pi / mB) * 0.065
aIBi = -1.24729

Params = [aIBi, mB, n0, gBB]
ham = PolaronHamiltonianGaussian.PolaronHamiltonianGaussian(gs, Params)

print(ham.h_omega)
print(ham.h_omega_bar)

#print(ham.h_two_phon)
#print(ham.gnum )

# ----------------------------------------
# Imaginary Time evolution
# ----------------------------------------
tMax = 100
dt = 0.1

# start = timer()
#
# tVec = np.arange(0, tMax, dt)
# NB_Vec = np.zeros(tVec.size, dtype=float)
# Zfactor_Vec = np.zeros(tVec.size, dtype=float)
# energy_vec = np.zeros(tVec.size, dtype=float)
#
# for ind, t in enumerate(tVec):
#     NB_Vec[ind] = gs.get_PhononNumber()
#     Zfactor_Vec[ind] = gs.get_Zfactor()
#     energy_vec[ind] = gs.get_energy(ham)
#
#     gs.evolve_real_time(dt, ham)
#
#
# end = timer()
#
# print(end - start)
#
# # save data
# data = [ham.Params, tVec, NB_Vec, Zfactor_Vec, energy_vec]
#
# dirpath = os.path.dirname(os.path.realpath(__file__))
# np.save(dirpath + '/data/gsrt_aIBi:%.2f.npy' % (aIBi), data)
#
# print(energy_vec[-1])
# print(NB_Vec[-1])
# print(Zfactor_Vec[-1])
#
# # ----------------------------------------
# # Analysis
# # ----------------------------------------
#
# figN, axN = plt.subplots()
# axN.plot(tVec, NB_Vec, 'k-')
# axN.set_xlabel('Time ($t$)')
# axN.set_ylabel('$N_{ph}$')
# axN.set_title('Number of Phonons')
# #figN.savefig('quench_PhononNumber.pdf')
#
# figZ, axZ = plt.subplots()
# axZ.plot(tVec, Zfactor_Vec, 'k-')
# axZ.set_xlabel('Time ($t$)')
# axZ.set_ylabel('$Z$')
# axZ.set_title('Zfactor')
#
# figE, axE = plt.subplots()
# axE.plot(tVec, energy_vec, 'k-')
# axE.set_xlabel('Time ($t$)')
# axE.set_ylabel('$E$')
# axE.set_title('Energy')
#
# plt.show()
