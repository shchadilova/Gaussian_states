import numpy as np


# Equations of motion for Gaussian states without phase

def equations_real_time(variables_t, t, GS, Ham):

    # RESTORE MATRICES FROM ARRAYS
    size = 2 * GS.size
    amplitude_t = variables_t[: size]
    gamma_t = variables_t[size:].reshape((size, size))

    amplitude_update = GS.sigma @ Ham.get_h_amplitude(amplitude_t, gamma_t, GS)

    right = GS.sigma @ np.transpose(np.transpose(gamma_t) @ Ham.get_h_gamma(amplitude_t, gamma_t, GS) )
    left = (gamma_t @ Ham.get_h_gamma(amplitude_t, gamma_t, GS)) @ GS.sigma

    gamma_update = np.reshape(right - left, size*size)


    return np.append(amplitude_update, gamma_update)


def equations_imaginary_time(variables_t, t, GS, Ham):

    # RESTORE MATRICES FROM ARRAYS
    size = 2 * GS.size
    amplitude_t = variables_t[: size]
    gamma_t = variables_t[size:].reshape((size, size))

    # CALCULATE RHS OF THE EQ OF MOTION: AMPLITUDE

    amplitude_update = - gamma_t @ (Ham.get_h_amplitude(amplitude_t, gamma_t, GS))

    # CALCULATE RHS OF THE EQ OF MOTION: GAMMA
    dv_x_dv = np.outer(GS.dv, GS.dv)

    right = (np.transpose(GS.sigma) @ (Ham.get_h_gamma(amplitude_t, gamma_t, GS)) ) @ GS.sigma
    left = (gamma_t @ (Ham.get_h_gamma(amplitude_t, gamma_t, GS))) @ gamma_t

    gamma_update = np.reshape(right - left, size*size)
    #print(right-left)

    return np.append(amplitude_update, gamma_update)


# Equations of motion for Gaussian states WITH phase

def equations_real_time_w_phase(variables_t, t, GS, Ham):

    # partition variables

    size = GS.size
    ind_gamma = 2*size
    ind_phi = ind_gamma+4*size*size
    ind_lambda = ind_phi+1

    amplitude_t = variables_t[: ind_gamma]
    gamma_t = variables_t[ind_gamma:ind_phi].reshape((2*size, 2*size))
    phi_t = variables_t[ind_phi:ind_phi+1]
    lambda_t = variables_t[ind_lambda:ind_lambda+size*size].reshape((size, size))

    # variables update
    # CALCULATE RHS OF THE EQ OF MOTION: AMPLITUDE

    amplitude_update = GS.sigma @ Ham.get_h_amplitude(amplitude_t, gamma_t, GS)

    # CALCULATE RHS OF THE EQ OF MOTION: GAMMA
    right = (GS.sigma @ Ham.get_h_gamma(amplitude_t, gamma_t, GS)) @ gamma_t
    left = gamma_t @ (Ham.get_h_gamma(amplitude_t, gamma_t, GS) @ GS.sigma)

    gamma_update = np.reshape(right - left, size*size)

    # CALCULATE RHS OF THE EQ OF MOTION: PHI
    dv= GS.grid.dv()
    phi_update = ham.gnum + 1./2.*ham.h_frohlich @(amplitude_t * dv) \
                 + np.trace( np.conjugate(np.transpose(ham.h_omega_bar)) @ (dv * lambda_t ) )

    # CALCULATE RHS OF THE EQ OF MOTION: LAMBDA

    lambda_update = 1./2* ham.h_omega_bar + ham.h_omega @ (dv * lambda_t) \
                    + lambda_t @ (dv* np.transpose(ham.h_omega)) \
                    + 2* lambda_t @ (dv * ham.h_omega_bar * dv) @ lambda_t

    return np.append(amplitude_update, gamma_update, phi_update, -1j*lambda_update)



# Equations of motion for Coherent states

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

def equations_real_time_mf(variables_t, t, GS, Ham):

    # RESTORE MATRICES FROM ARRAYS
    size = 2 * GS.size
    amplitude_t = variables_t[: size]
    gamma_t = variables_t[size:].reshape((size, size))

    # CALCULATE RHS OF THE EQ OF MOTION: AMPLITUDE
    amplitude_update = GS.sigma @ Ham.get_h_amplitude(amplitude_t, gamma_t, GS)

    # NO UPDATE FOR THE GAMMA EQ. OF MOTION!
    gamma_update = np.zeros(size*size)
    #print(right-left)

    return np.append(amplitude_update, gamma_update)
