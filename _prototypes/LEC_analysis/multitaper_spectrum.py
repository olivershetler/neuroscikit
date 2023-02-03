import numpy as np
from spectrum import dpss, nextpow2
from numpy.fft import fft

def mtSpec(X, Y, Fs, bandWidth, NFFT, RemoveTemporalMean=True, RemoveEnsembleMean=True, nTapers=[]):
    """Single window multitaper spectra for bivariate sequences

    X and Y can be continuous np.variables, point processes or mixed
    e.g. if Y is a spike train,  00010001001..., the np.mean(Y(:, trial)) in
    each particular trial MUST have been subtracted from each timepoint in the
    same trial. If X or Y contain a continuous np.variable, the time series
    should be detrended (each trial individually, see Matlab function detrend).

    Args:
    X                   timepoints,trials)
    Y                   timepoints,trials)
    Fs                  sampling rate in Hertz
    BandWidth           full bandwidth, e.g. 4 Hz
    RemoveTemporalMean  subtract the temporal mean from each trial if True
    RemoveEnsembleMean  subtract the ensemble mean from each trial if True
    nTapers             number of tapers, default to empty

    Returns:
    Pxx                 X Power spectrum
    Pyy                 Y power spectrum
    Pxy                 cross-spectrum density
    XYphi               spectral phase
    Cxy                 spectral SQUARED coherence,
    F                   frequency axis, bandwidth
    nTapers             2 * N * W - 1, where W = BandWidth / (2 * Fs)

    Written by Wilson Truccolo NEUR 2110 2015, 2017, 2018
    Transalted into Python by Ewina Pun, tsam_kiu_pun@brown.edu on 06/13/2019
    """

    # check input attributes
    if X.shape != Y.shape:
        raise ValueError('X and Y have different dimensions, exiting now ...')
    if X.ndim > 2 or Y.ndim > 2:
        raise ValueError('X and Y should be 1-D or 2-D arrays.')
    if type(Fs) is not float:
        Fs = float(Fs)
    if type(NFFT) is not int:
        NFFT = int(NFFT)
    # reshape array
    if X.shape[0] == 1 or X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.shape[0] == 1 or Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    N = X.shape[0]
    ntrials = X.shape[1]

    dt = 1 / Fs
    F0 = Fs / float(N)
    Nyquist = Fs / 2
    # The half-bandwidth W is expressed in terms of frequencies in [-0.5, 0.5].
    # So we divide bandWidth by the sampling rate Fs.
    W = bandWidth / (2 * Fs)

    # Get the dpss
    E, V = dpss(N, roundUpHalf(N * W))

    # WT: May 2014 See Chronux dpsschk.m function
    # we need tapers to be such that integral of the square of each taper
    # equals 1 dpss computes tapers such that the SUM of squares equals 1
    # - so we need to multiply the dpss computed tapers by sqrt(N) to get
    # the right normalization
    E = E * np.sqrt(N)

    if not nTapers:
        # By convention,  the first 2 * NW eigenvalues / vectors are stored
        k = min(roundUpHalf(2 * N * W), N)
        k = max(k - 1, 1)
        nTapers = int(k)
    elif nTapers > E.shape[1]:
        raise ValueError('Too many tapers requested.')

    V = V[:nTapers]
    E = E[:, :nTapers]

    if not NFFT:
        # zero padding
        NFFT = int(2 ** nextpow2(N))

    if RemoveEnsembleMean:
        if ntrials == 1:
            X = X - np.mean(X)
            Y = Y - np.mean(Y)
        else:
            MX = np.mean(X, axis=1).reshape(-1, 1)
            MY = np.mean(Y, axis=1).reshape(-1, 1)
            X = X - np.tile(MX, (1, ntrials))
            Y = Y - np.tile(MY, (1, ntrials))
        # print('Removed ensemble mean.')

    Pxx = np.zeros((NFFT, 1))
    Pyy = np.zeros((NFFT, 1))
    Pxy = np.zeros((NFFT, 1))
    Cxy = np.zeros((NFFT, 1))

    for trial in range(ntrials):
        if RemoveTemporalMean:
            x = X[:, trial] - np.mean(X[:, trial])
            y = Y[:, trial] - np.mean(Y[:, trial])
            # print('Removed temporal mean.')
        else:
            x = X[:, trial]
            y = Y[:, trial]

        # Compute the windowed DFTs.
        # mrule says: I wasn't sure how best to do this
        x = E * np.tile(x, (nTapers, 1)).T
        y = E * np.tile(y, (nTapers, 1)).T
        # x = np.array([e * x for e in E.T]).T
        # y = np.array([e * y for e in E.T]).T
        # if N <= NFFT:
        xf = fft(x, NFFT, 0)
        yf = fft(y, NFFT, 0)
        # else:  # Wrap the data modulo NFFT  if N > NFFT
        #     # use CZT to compute DFT on NFFT  evenly spaced samples around the
        #     # unit circle:
            # xf = czt(x, NFFT, 0)
            # yf = czt(y, NFFT, 0)
        wt = np.ones((nTapers, 1))
        # Park estimate
        #wt = V(:)

        Xx2 = np.dot((np.abs(xf) ** 2), wt) / nTapers #average across tappers
        Yy2 = np.dot((np.abs(yf) ** 2), wt) / nTapers
        Xy2 = np.dot((yf * np.conj(xf)), wt) / nTapers

        Pxx = Pxx + Xx2
        Pyy = Pyy + Yy2
        Pxy = Pxy + Xy2

    Pxx = Pxx / ntrials
    Pyy = Pyy / ntrials
    Pxy = Pxy / ntrials
    XYphi = np.angle(Pxy)
    # coherence function estimate
    Cxy = np.abs(Pxy)**2 / (Pxx * Pyy)

    # Select first half
    #factor - 2 to get one - sided
    #factor - 2 is not needed for dc
    if (X.imag == 0).all() and (Y.imag == 0).all():   # if x and y are not complex
        if NFFT % 2:    # NFFT  odd
            nf = int((NFFT + 1) / 2)
            # freq = np.linspace(F0, Nyquist - df, nf - 1)
            # freq = np.concatenate((np.array([0]), freq))

            # mrule says: python's lack of a sane way to concatenate numpy
            # arrays made direct translation from matlab tricky. This is
            # functionally the same
            # Pxx_unscaled = Pxx[select, :] # Take only [0, pi] or [0, pi)
            # Pxx = np.array(Pxx_unscaled) #copy
            # Pxx[1:] *= 2  #Scale. Only DC is a unique point and doesn't get doubled

            # Pyy_unscaled = Pyy[select, :] # Take only [0, pi] or [0, pi)
            # Pyy = np.array(Pyy_unscaled) #copy
            # Pyy[1:] *= 2  #Scale. Only DC is a unique point and doesn't get doubled

            # Pxy_unscaled = Pxy[select, :] # Take only [0, pi] or [0, pi)
            # Pxy = np.array(Pxy_unscaled) #copy
            # Pxy[1:] *= 2  #Scale. Only DC is a unique point and doesn't get doubled
        else:
            nf = int(NFFT / 2 + 1)
                # freq = np.linspace(F0, Nyquist, nf - 1)
            # freq = np.concatenate((np.array([0]), freq))

            # Pxx_unscaled = Pxx[select, :] # Take only [0, pi] or [0, pi)
            # Pxx = np.array(Pxx_unscaled) #copy
            # Pxx[1: - 1] *= 2 # scale,  don't double unique Nyquist point

            # Pyy_unscaled = Pyy[select, :] # Take only [0, pi] or [0, pi)
            # Pyy = np.array(Pyy_unscaled) #copy
            # Pyy[1: - 1] *= 2 # scale,  don't double unique Nyquist point

            # Pxy_unscaled = Pxy[select, :] # Take only [0, pi] or [0, pi)
            # Pxy = np.array(Pxy_unscaled) #copy
            # Pxy[1: - 1] *= 2 # scale,  don't double unique Nyquist point
        F = np.linspace(F0, Nyquist, nf - 1)
        F = np.insert(F, 0, 0)
    else:
        nf = NFFT
        F = np.linspace(F0, 2 * Nyquist, nf - 1)
        F = np.insert(F, 0, 0)

    # freq_vector = (select - 1).T * Fs / NFFT
    Pxx = Pxx[:nf, :]
    Pyy = Pyy[:nf, :]
    Pxy = Pxy[:nf, :]
    Cxy = Cxy[:nf, :]

    # Compute the PSD [Power/freq]
    # Power = energy in a give time period (length of a realization or trial),
    # thus divide by T (N/Fs); The power spectrum density has units of
    # (signal units)^/Hz;
    # This is equivalent to psd = 2 * dt^2 * 1/T * Pxx;
    # The multiplication by 2 gives the one-sided spectrum
    # The multiplication by dt^2 is done because Matlab's definition of DFT
    # does not include the factor dt. By computing the power, that becomes dt^2
    # Divide by the window length:
    Pxx = 2 * dt**2 * F0 * Pxx
    Pyy = 2 * dt**2 * F0 * Pyy
    Pxy = 2 * dt**2 * F0 * Pxy

    # Compute the PSD [Power / freq]
    # if Fs not None:
    #    Pxx = Pxx / Fs # Scale by the sampling frequency (i.e. multiply by sampling interval) to obtain the psd
    #    Pyy = Pyy / Fs
    #    Pxy = Pxy / Fs
    #    units = 'Hz'
    # else:
    #    Pxx = Pxx / (2. * pi) # Scale the power spectrum by 2 * pi to obtain the psd
    #    Pyy = Pyy / (2. * pi) # Scale the power spectrum by 2 * pi to obtain the psd
    #    Pxy = Pxy / (2. * pi)
    #    units = 'rad / sample'

    return np.squeeze(Pxx), np.squeeze(Pyy), np.squeeze(Pxy), XYphi, np.squeeze(Cxy), F, nTapers

def roundUpHalf(val):
    # blindly round .5 up (python 3 default banking rounding to even numbers)
    if (float(val) % 1) >= 0.5:
        x = np.ceil(val)
    else:
        x = round(val)
    return x
