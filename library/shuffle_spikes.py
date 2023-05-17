import numpy as np
import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)


import library.opexebo.errors as errors


def shuffle_spikes(ts: np.ndarray, pos_x: np.ndarray,
                   pos_y: np.ndarray, pos_t: np.ndarray) -> list:

    '''
        Shuffles spike and position data.

        Params:
            ts (np.ndarray):
                Timestamps of spike events
            pos_x, pos_y, pos_t (np.ndarray):
                Arrays of x,y coordinates as well as position timestamps

        Returns:
            np.ndarray: shuffled_spikes
            --------
            shuffled_spikes:
                Array where columns are shuffled x and y spike coordinates respectively.
    '''

    # spike_x = np.zeros((1,len(ts)))
    # spike_y = np.zeros((1,len(ts)))
    # shuffled_spike_xy = np.zeros((2,len(ts)))
    # Compute a shuffling between 20 and 100

    # shuffled_times = _shuffle(ts, 20, 1, t_start=min(ts), t_stop = max(ts))[0]

    # shuffled_spikes = []
    # # For each shuffled time, find and place the corresponding
    # # spike x and y coordinates into the shuffled_spikes array
    # for i, single_shuffled_time in enumerate(shuffled_times):
    #     for j, time in enumerate(single_shuffled_time):
    #         index = np.abs(pos_t - time).argmin()
    #         spike_x[0][j] = pos_x[index]
    #         spike_y[0][j] = pos_y[index]

    #     shuffled_spike_xy[0] = spike_x
    #     shuffled_spike_xy[1] = spike_y
    #     shuffled_spikes.append(shuffled_spike_xy.copy())

    shuffled_times = _single_shuffle(ts, 20, t_start=min(ts), t_stop = max(ts))[0]

    # shuffled_spikes = []
    spike_x = []
    spike_y = []
    # For each shuffled time, find and place the corresponding
    # spike x and y coordinates into the shuffled_spikes array
    for j, time in enumerate(shuffled_times):
        index = np.abs(pos_t - time).argmin()
        spike_x.append(pos_x[index])
        spike_y.append(pos_y[index])

    # shuffled_spike_xy[0] = spike_x
    # shuffled_spike_xy[1] = spike_y
    # shuffled_spikes.append(shuffled_spike_xy.copy())

    assert len(shuffled_times.squeeze()) == len(ts), 'Shuffled times {} and original times {} are not the same length'.format(len(shuffled_times), len(ts))

    return np.array(spike_x).squeeze(), np.array(spike_y).squeeze(), shuffled_times.squeeze()

def _single_shuffle(times, offset_lim, t_start, t_stop):
    iterations = 1

    if offset_lim >= 0.5 * (t_stop - t_start):
        offset_lim = int(0.5 * (t_stop - t_start))
        if offset_lim == 0.5 * (t_stop - t_start):
            offset_lim -= 1

    # Argument checking begins here
    if not isinstance(times, np.ndarray):
        raise errors.ArgumentError(
            "`times` must be 1 Numpy array ({})".format(type(times))
        )
    if not times.ndim == 1:
        raise errors.ArgumentError("`times` must be a 1D array ({})".format(times.ndim))
    if not np.isfinite(times).all():
        raise errors.ArgumentError("`times` cannot include non-finite or NaN values")

    if offset_lim <= 0:
        raise errors.ArgumentError(
            "`offset_lim` must be greater than zero ({}".format(offset_lim)
        )
    if not np.isfinite(offset_lim):
        raise errors.ArgumentError(
            "`offset_lim` must be finite ({})".format(offset_lim)
        )

    if not np.isfinite(iterations):
        raise errors.ArgumentError("`iterations` must be finite".format(iterations))

    if not np.isfinite(t_start):
        raise errors.ArgumentError("`t_start` must be finite".format(t_start))
    if not np.isfinite(t_stop):
        raise errors.ArgumentError("`t_stop` must be finite".format(t_stop))

    if t_start is None:
        t_start = min(times)
    if t_stop is None:
        t_stop = max(times)

    if t_start > min(times):
        raise errors.ArgumentError(
            "`t_start` must be greater than or equal to `min(times)`"
        )
    if t_stop < max(times):
        raise errors.ArgumentError(
            "`t_stop` must be less than or equal to `max(times)`"
        )

    if t_start == t_stop:
        raise errors.ArgumentError(
            "`t_start` and `t_stop` cannot be identical ({})".format(t_start)
        )

    if offset_lim >= 0.5 * (t_stop - t_start):
        raise errors.ArgumentError(
            "`offset_lim` must be less than half of the time span ({}, {})".format(
                offset_lim, t_stop - t_start
            )
        )
    # argument checking ends here

    # Main logic begins here
    increments_base = np.random.RandomState().rand(
        iterations
    )  # uniformly distributed in [0,1]
    increments = (
        t_start + offset_lim + (increments_base * (t_stop - t_start - 2 * offset_lim))
    )

    output = times + increments
    output = output.squeeze()

    # Circularising: i.e. folding times outside the boundary back inside the
    # boundary, and then re-ordering by the updated, refolded times
    out_of_bounds = output > t_stop
    output[out_of_bounds] = output[out_of_bounds] - (t_stop - t_start)

    assert len(output.shape) == 1, "output.shape is not 1D ({})".format(output.shape)
    # output.sort(axis=1)  # sort along each row independently of all other rows.
    output = np.sort(output)

    return output, increments


""""""""""""""""""""""""""" From Opexebo https://pypi.org/project/opexebo/ """""""""""""""""""""""""""

def _shuffle(
    times: np.ndarray,
    offset_lim: float,
    iterations: int,
    t_start: float = None,
    t_stop: float = None,
):
    """
    Duplicate the provided time series ``iterations`` number of times. Each
    duplicate will be incremented circularly by a random value not smaller than
    ``offset_lim``.

    Circular incrementation results in (the majority) of time _differences_
    remaining preserved

    * Initially, we have a time series, ``times``, both with
      values in the range ``[min(times), max(times)]``. ``t_start`` may be smaller than
      ``min(times)``, and ``t_stop`` may be larger than ``max(times)``
    * ``iterations`` number of duplicates of ``times`` are created.
    * In each iteraction, a random increment ``T`` is generated, and added to
      each value in that iteration, such that values now fall into the range
      ``[min(times)+T, max(times)+T]``. ``max(times)+T`` may exceed ``t_stop``.
    * All timestamps matching ``t_n > t_stop`` are mapped back into the range
    ``[t_start, t_stop]`` by subtracting ``(t_stop-t_start)``
    * The iteration is re-ordered by value (moving those beyond the far edge
      back to the beginning)

    Parameters
    ----------
    times : np.ndarray
        1D array of floats. Time series data to be shuffled
    offset_lim : float
        Minimum offset from the original time values. Each iteration is
        incremented by a random value evenly distributed in the range
        ``[offset_lim, t_stop-offset_lim]``
    iterations : int
        Number of repeats of ``times`` to be returned
    t_start : float, optional
        Lower bound of time domain. Must meet the criteria ``t_start <= min(times)``
        Defaults to ``min(times)``
    t_stop : float, optional
        Upper bound of time domain. Must meet the criteria ``t_stop >= max(times)``
        Defaults to ``max(times)``

    Returns
    -------
    output : np.ndarray
        iterations x N array of times. A single iteration is accessed as
        ``output[i]``
    increments : np.ndarray
        1D array of offset values that were used
    """
    # Argument checking begins here
    if not isinstance(times, np.ndarray):
        raise errors.ArgumentError(
            "`times` must be 1 Numpy array ({})".format(type(times))
        )
    if not times.ndim == 1:
        raise errors.ArgumentError("`times` must be a 1D array ({})".format(times.ndim))
    if not np.isfinite(times).all():
        raise errors.ArgumentError("`times` cannot include non-finite or NaN values")

    if offset_lim <= 0:
        raise errors.ArgumentError(
            "`offset_lim` must be greater than zero ({}".format(offset_lim)
        )
    if not np.isfinite(offset_lim):
        raise errors.ArgumentError(
            "`offset_lim` must be finite ({})".format(offset_lim)
        )

    if iterations < 2:
        raise errors.ArgumentError(
            "qiterations must be a positive integer greater than 1 ({})".format(
                iterations
            )
        )
    if not np.isfinite(iterations):
        raise errors.ArgumentError("`iterations` must be finite".format(iterations))

    if not np.isfinite(t_start):
        raise errors.ArgumentError("`t_start` must be finite".format(t_start))
    if not np.isfinite(t_stop):
        raise errors.ArgumentError("`t_stop` must be finite".format(t_stop))

    if t_start is None:
        t_start = min(times)
    if t_stop is None:
        t_stop = max(times)

    if t_start > min(times):
        raise errors.ArgumentError(
            "`t_start` must be greater than or equal to `min(times)`"
        )
    if t_stop < max(times):
        raise errors.ArgumentError(
            "`t_stop` must be less than or equal to `max(times)`"
        )

    if t_start == t_stop:
        raise errors.ArgumentError(
            "`t_start` and `t_stop` cannot be identical ({})".format(t_start)
        )

    if offset_lim >= 0.5 * (t_stop - t_start):
        raise errors.ArgumentError(
            "`offset_lim` must be less than half of the time span ({}, {})".format(
                offset_lim, t_stop - t_start
            )
        )
    # argument checking ends here

    # Main logic begins here
    increments_base = np.random.RandomState().rand(
        iterations
    )  # uniformly distributed in [0,1]
    increments = (
        t_start + offset_lim + (increments_base * (t_stop - t_start - 2 * offset_lim))
    )

    # Stack copies of `times`, one per row, for `iterations` number of rows
    # stack copies of `increments`, one per column, for `times.size` number of columns
    # We get two identically shaped arrays that can just be added together to perform the increments.
    output = np.repeat(times[np.newaxis, :], iterations, axis=0)
    increments_arr = np.repeat(increments[:, np.newaxis], times.size, axis=1)

    output = increments_arr + output

    # Circularising: i.e. folding times outside the boundary back inside the
    # boundary, and then re-ordering by the updated, refolded times
    out_of_bounds = output > t_stop
    output[out_of_bounds] = output[out_of_bounds] - (t_stop - t_start)

    output.sort(axis=1)  # sort along each row independently of all other rows.

    return output, increments

