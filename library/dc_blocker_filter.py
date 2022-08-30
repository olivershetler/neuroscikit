import numpy as np
from scipy import signal

def dcblock(data, fc, fs=None, analog_val=False, self=None):

    """This method will return the filter coefficients for a DC Blocker Filter"""

    if fs is None:
        Fc = fc

    else:
        Fc = 2 * fc / fs

    p = (np.sqrt(3) - 2 * np.sin(np.pi * Fc)) / (np.sin(np.pi * Fc) +
                                                 np.sqrt(3) * np.cos(np.pi * Fc));

    b = np.array([1, -1])
    a = np.array([1, -p])

    if len(data) != 0:
        if len(data.shape) > 1:

            # filtered_data = np.zeros((data.shape[0], data.shape[1]))
            filtered_data = signal.filtfilt(b, a, data, axis=1)

        else:
            # filtered_data = signal.lfilter(b, a, data)
            filtered_data = signal.filtfilt(b, a, data)


    if len(data) != 0:
        return filtered_data