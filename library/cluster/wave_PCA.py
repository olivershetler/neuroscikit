import numpy as np

def wave_PCA(cv):
    """
    Principal Component Analysis of standardized waveforms from a
    given (unstandardized) waveform covariance matrix cv(nSamp,nSamp).

    Args:
        cv: nSamp x nSamp wavefrom covariance matrix (unnormalized)

    Returns:
        pc: column oriented principal components (Eigenvectors)
        rpc: column oriented Eigenvectors weighted with their relative amplitudes
        ev: eigenvalues of SVD (= std deviation of data projected onto pc)
        rev: relative eigenvalues so that their sum = 1

    """
    sd = np.sqrt(np.diag(cv)).reshape((-1, 1))  # row wise standard deviation

    # standardized covariance matrix
    cvn = np.divide(cv, np.multiply(sd, sd.T))

    u, ev, pc = np.linalg.svd(cvn)

    # the pc is transposed in the matlab version
    pc = pc.T

    ev = ev.reshape((-1, 1))

    rev = ev / np.sum(ev)  # relative eigne values so that their sum = 1

    rpc = np.multiply(pc, rev.T)

    return pc, rpc, ev, rev
