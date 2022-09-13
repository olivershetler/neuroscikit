import numpy as np

def get_firing_rate_vs_time(times: np.ndarray, pos_t: np.ndarray, window: int) -> tuple:

    '''
        Computes firing rate as a function of time

        Params:
            times (np.ndarray):
                Array of timestamps of when the neuron fired
            pos_t (np.ndarray):
                Time array of entire experiment
            window (int):
                Defines a time window in milliseconds.

            *Example*
            window: 400 means we will attempt to collect firing data in 'bins' of
            400 millisecods before computing the firing rates.

        Returns:
            tuple: firing_rate, firing_time
            --------
            rate_vector (np.ndarray):
                Array containing firing rate data across entire experiment
            firing_time: (np.ndarray):
                Timestamps of when firing occured
        '''

    # Initialize zero time elapsed, zero spike events, and zero bin times.
    time_elapsed = 0
    number_of_elements = 1
    bin_time = [0,0]

    # Initialzie empty firing rate and firing time arrays
    firing_rate = [0]
    firing_time = [0]

    # Collect firing data in bins of 400ms
    for i in range(1, len(times)):
        if time_elapsed == 0:
            # Set a bin start time
            bin_time_start = times[i-1]

        # Increment time elapsed and spike number as spike event times are iterated over
        time_elapsed += (times[i] - times[i-1])
        number_of_elements += 1

        # If the elapsed time exceeds 400ms
        if time_elapsed > (window/1000):
            # Set bin end time
            bin_time_end = bin_time_start + time_elapsed
            # Compute rate, and add element to firing_rate array
            firing_rate.append(number_of_elements/time_elapsed)
            firing_time.append( (bin_time_start + bin_time_end)/2 )
            # Reset elapsed time and spiek events number
            time_elapsed = 0
            number_of_elements = 0



    rate_vector = np.zeros((len(pos_t), 1))
    index_values = []
    for i in range(len(firing_time)):
        index_values.append(  (np.abs(pos_t - firing_time[i])).argmin()  )

    firing_rate = np.array(firing_rate).reshape((len(firing_rate), 1))
    rate_vector[index_values] = firing_rate

    return rate_vector, firing_time
