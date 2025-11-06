import numpy as np
from scipy.stats import multivariate_normal


def mfenkf(xf, xf_rom, xf_fom_in_rom, Phi, y, H, R, R_rom, infl, infl_rom):
    """
    This runs one iteration of the MFEnKF
    :param xf: ndarray: particles/samples in FOM space --- nXN (primary/principle variate)
    :param xf_fom_in_rom: ndarray: particles/sample of FOM in ROM space --- rXN (control variate)
    :param xf_rom: ndarray:  particles in rom space --- rXN_rom (ancillary variate)
    :param Phi: ndarray: pod basis matrix --- nXn_rom
    :param y: ndarray: sparse observations
    :param H: ndarray: Linearized observation operator (TODO: non-linearized case)
    :param R: ndarray: Observation error covariance (consider diagonal for now)
    :param R_rom: ndarray: Observation error covariance for ROM particles(consider)
    :param infl: scalar:  inflation for fom particles
    :param infl_rom: scalar: inflation for rom particles
    :return:xa: ndarray: the analysis particles in FOM space
    :return: xa_rom: ndarray: the analysis particles in ROM space
    """
    # r is the rom dimension
    # n is the FOM dimension
    # o is the number of observations

    # define the return arguments
    xa = np.zeros_like(xf)
    xa_rom = np.zeros_like(xf_rom)

    # calculate preliminaries
    n = xf.shape[0]
    N = xf.shape[1]
    N_rom = xf_rom.shape[1]
    N1 = np.sqrt(N-1)
    N_rom1 = np.sqrt(N_rom-1)

    num_of_obs = R.shape[0]  # check

    # project the fom onto the rom space  (control variate)
    # xf_fom_in_rom = phi.T @ xa

    # inflate the prior covariances of all three variates
    xf_mean = np.mean(xf, axis=1, keepdims=True)
    xf = xf_mean + infl * (xf - xf_mean)

    xf_fom_in_rom_mean = np.mean(xf_fom_in_rom, axis=1, keepdims=True)
    xf_fom_in_rom = xf_fom_in_rom_mean + infl * (xf_fom_in_rom - xf_fom_in_rom_mean)

    xf_rom_mean = np.mean(xf_rom, axis=1, keepdims=True)
    xf_rom = xf_rom_mean + infl_rom * (xf_rom - xf_rom_mean)

    # find relevant means
    mu_X = np.mean(xf, axis=1, keepdims=True)  # n x 1
    mu_Uhat = np.mean(xf_fom_in_rom, axis=1, keepdims=True)  # shape: (r, 1)
    mu_U = np.mean(xf_rom, axis=1, keepdims=True)  # shape: (r, 1)

    # Projected onto observation space
    HX = H @ xf  # o x N
    HPUhat = H @ (Phi @ xf_fom_in_rom)  # shape: (o, N)
    HPU = H @ (Phi @ xf_rom)  # shape: (o, Nrom)

    # Means of projected observations
    mu_HX = np.mean(HX, axis=1, keepdims=True)  # shape: (o, 1)
    mu_HPUhat = np.mean(HPUhat, axis=1, keepdims=True)  # shape: (o, 1)
    mu_HPU = np.mean(HPU, axis=1, keepdims=True)  # shape: (o, 1)

    # calculate anomalies
    # Center and scale ensemble anomalies
    AX = (1.0 / N1) * (xf - mu_X)  # shape: (n, N)
    AUhat = (1.0 / N1) * (xf_fom_in_rom - mu_Uhat)  # shape: (r, N)
    AU = (1.0 / N_rom1) * (xf_rom - mu_U)  # shape: (r, Nrom)

    AHX = (1.0 / N1) * (HX - mu_HX)  # shape: (o, N1)
    AHPUhat = (1.0 / N1) * (HPUhat - mu_HPUhat)  # shape: (o, N1)
    AHPU = (1.0 / N_rom1) * (HPU - mu_HPU)  # shape: (o, Nrom)

    # calculated background/forecast combined mean of total variate
    zf_mean = np.mean(xf, axis=1) - 0.5 * Phi @ (np.mean(xf_fom_in_rom, axis=1) - np.mean(xf_rom, axis=1))

    # calculate cross covariances
    sigma_XHX = AX @ AHX.T                  # shape: (n, o)
    sigma_UhatHPUhat = AUhat @ AHPUhat.T    # shape: (r, o)
    sigma_XHPUhat = AX @ AHPUhat.T          # shape: (n, o)
    sigma_UhatHX = AUhat @ AHX.T            # shape: (r, o)
    sigma_UHPU = AU @ AHPU.T                # shape: (n, o)

    sigma_HXHX = AHX @ AHX.T                # shape: (o, o)
    sigma_HXHPUhat = AHX @ AHPUhat.T        # shape: (o, o)
    sigma_HPUhatHX = AHPUhat @ AHX.T        # shape: (o, o)
    sigma_HPUHPU = AHPU @ AHPU.T            # shape: (o, o)
    sigma_HPUhatHPUhat = AHPUhat @ AHPUhat.T# shape: (o, o)

    sigma_ZHZ = (
            sigma_XHX
            + 0.25 * Phi @ sigma_UhatHPUhat
            - 0.5 * sigma_XHPUhat
            - 0.5 * Phi @ sigma_UhatHX
            + 0.25 * Phi @ sigma_UHPU
    )                                       # shape: (n, o)

    sigma_HZHZ = (
            sigma_HXHX
            + 0.25 * sigma_HPUhatHPUhat
            - 0.5 * sigma_HXHPUhat
            - 0.5 * sigma_HPUhatHX
            + 0.25 * sigma_HPUHPU
    )                                       # shape: (o, o)

    # compute the Kalman Gain
    K = np.linalg.solve(sigma_HZHZ + R, sigma_ZHZ.T).T  # shape: (n, o)

    mean_zero = np.zeros(num_of_obs)

    # noise samples with zero mean
    samplenoise_X = multivariate_normal.rvs(mean=mean_zero, cov=R, size=N).T  # shape: (o, N)
    samplenoise_U = multivariate_normal.rvs(mean=mean_zero, cov=R_rom, size=N_rom).T  # shape: (o, Nrom)

    A_samplenoise_X = (1 / N1) * (samplenoise_X - np.mean(samplenoise_X, axis=1, keepdims=True))  # shape: (o, N)
    A_samplenoise_U = (1 / N_rom1) * (samplenoise_U - np.mean(samplenoise_U, axis=1, keepdims=True))  # shape: (o, Nrom)

    AXa = AX - K @ (AHX - A_samplenoise_X)                      # shape: (n, N)
    AUhata = AUhat - Phi.T @ K @ (AHPUhat - A_samplenoise_X)    # shape: (r, N)
    AUa = AU - Phi.T @ K @ (AHPU - A_samplenoise_U)             # shape: (r, Nrom)

    mu_Hhat_Zb = mu_HX - 0.5 * (mu_HPUhat - mu_HPU)             # shape: (o, 1)

    za_mean = zf_mean - K @ (mu_Hhat_Zb.reshape(num_of_obs) - y)  # shape: (n, 1)
    # print(Phi.T.shape)
    # print(za_mean.shape)
    # print(AUa.shape)
    # update the ensembles
    xa = za_mean.reshape(n, 1) + N1 * AXa
    xa_rom = Phi.T @ za_mean.reshape(n, 1) + N_rom1*AUa

    # we discard the control variate here
    # return
    return xa, xa_rom





