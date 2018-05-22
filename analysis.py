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
# Load data
# ----------------------------------------
# -1., -0.5, -0.1, 0, 0.1, 0.5, 1., 2.
aIBi = 2.

dirpath = os.path.dirname(os.path.realpath(__file__))

[Params_gs_rt, tVec_gs_rt, NB_Vec_gs_rt, Zfactor_Vec_gs_rt, energy_vec_gs_rt] = \
    np.load(dirpath + '/data/gsrt_aIBi:%.2f.npy' % (aIBi))

[Params_mf_rt, tVec_mf_rt, NB_Vec_mf_rt, Zfactor_Vec_mf_rt, energy_vec_mf_rt] = \
    np.load(dirpath + '/data/mfrt_aIBi:%.2f.npy' % (aIBi))

# [Params_gs_it, tVec_gs_it, NB_Vec_gs_it, Zfactor_Vec_gs_it, energy_vec_gs_it] = \
#     np.load(dirpath + '/data/gsit_aIBi:%.2f.npy' % (aIBi))

[Params_mf_it, tVec_mf_it, NB_Vec_mf_it, Zfactor_Vec_mf_it, energy_vec_mf_it] = \
    np.load(dirpath + '/data/mfit_aIBi:%.2f.npy' % (aIBi))


# print(energy_vec_gs_it[-1])
# print(energy_vec_mf_it[-1])
# ----------------------------------------
# Modify data
# ----------------------------------------

# size = len(NB_Vec_gs_it)
# NB_Vec_gs_it = np.ones(size) * NB_Vec_gs_it[-1]
# Zfactor_Vec_gs_it= np.ones(size) * Zfactor_Vec_gs_it[-1]
Zfactor_Vec_gs_rt= np.sqrt(Zfactor_Vec_gs_rt)

size = len(NB_Vec_mf_it)
NB_Vec_mf_it = np.ones(size) * NB_Vec_mf_it[-1]
Zfactor_Vec_mf_it = np.ones(size) * Zfactor_Vec_mf_it[-1]
Zfactor_Vec_mf_rt= np.sqrt(Zfactor_Vec_mf_rt)

# ----------------------------------------
# Plot
# ----------------------------------------

# figN, axN = plt.subplots()
# axN.plot(tVec_gs_rt, NB_Vec_gs_rt, 'r-')
# axN.plot(tVec_mf_rt, NB_Vec_mf_rt, 'b--')
# axN.plot(tVec_gs_it, NB_Vec_gs_it, 'g--')
# axN.plot(tVec_mf_it, NB_Vec_mf_it, 'k--')
# axN.set_xlim([0,tVec_gs_rt[-1]])
# axN.set_xlabel('Time ($t$)')
# axN.set_ylabel('$N_{ph}$')
# axN.set_title('Number of Phonons')
# #figN.savefig('quench_PhononNumber.pdf')

figZ, axZ = plt.subplots()
axZ.plot(tVec_gs_rt, Zfactor_Vec_gs_rt, 'g-')
axZ.plot(tVec_mf_rt, Zfactor_Vec_mf_rt, 'k-')
# axZ.plot(tVec_gs_it, Zfactor_Vec_gs_it, 'g--')
axZ.plot(tVec_mf_it, Zfactor_Vec_mf_it, 'k--')
axZ.set_xlim([0,tVec_gs_rt[-1]])
axZ.set_ylim([0,1])
axZ.set_xlabel('Time ($t$)')
axZ.set_ylabel('$Z$')
axZ.set_title('Zfactor')


plt.show()
