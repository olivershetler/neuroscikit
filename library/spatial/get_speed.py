import numpy as np

def speed2D(x, y, t):
    '''calculates an averaged/smoothed speed'''

    N = len(x)
    v = np.zeros((N, 1))

    for index in range(1, N-1):
        v[index] = np.sqrt((x[index + 1] - x[index - 1]) ** 2 + (y[index + 1] - y[index - 1]) ** 2) / (
        t[index + 1] - t[index - 1])

    v[0] = v[1]
    v[-1] = v[-2]
    v = v.flatten()

    kernel_size = 12
    kernel = np.ones(kernel_size) / kernel_size
    v_convolved = np.convolve(v, kernel, mode='same')

    return v_convolved
