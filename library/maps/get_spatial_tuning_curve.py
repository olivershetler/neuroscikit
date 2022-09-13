import numpy as np
#from opexebo.analysis import place_field, border_score
from opexebo.analysis import tuning_curve, angular_occupancy

def _smooth(array: np.ndarray, window: int) -> np.ndarray:

    '''
        Smooths an array using sliding wsindow approach

        Params:
            array (np.ndarray):
                Array to be smoothed
            window (int):
                Number of points to be smoothed at a time as a sliding window

        Returns:
            np.ndarray:
                smoothed_array
    '''

    # Initialize empty array
    smoothed_array = np.zeros((len(array), 1))

    # Iterate over array using sliding window and compute averages
    for i in range(len(array) - window + 1):
        current_average = sum(array[i:i+window]) / window
        smoothed_array[i:i+window] = current_average

    return smoothed_array

def _get_head_direction(pos_x: np.ndarray, pos_y: np.ndarray) -> np.ndarray:

    '''
        Will compute the head direction angle of subject over experiemnt.
        Params:
            pos_x, pos_y (np.ndarray):
                Arrays of x and y coordinates.

        Returns:
            np.ndarray:
                angles
            --------
            angles:
                Array of head direction angles in radians, reflecting what heading the subject
                during the course of the sesison.
    '''

    last_point = [0,0]  # Keep track of the last x,y point
    last_angle = 0      # Keep track of the most previous computed angle
    angles = []         # Will accumulate angles as they are computed

    # Iterate over the pos_x and pos_y points
    for i in range(len(pos_x)):

        # Grab the current point
        current_point = [float(pos_x[i]), float(pos_y[i])]

        # If the last point is the same as the current point
        if (last_point[0] == current_point[0]) and (last_point[1] == current_point[1]):
            # Retain the same angle
            angle = last_angle

        else:
            # Compute the arctan (i.e the angle formed by the line defined by both points
            # and the horizontal axis [range -180 to +180])
            # Uses the formula arctan( (y2-y1) / (x2-x1))
            Y = current_point[1] - last_point[1]
            X = current_point[0] - last_point[0]
            angle = math.atan2(Y,X) * (180/np.pi) # Convert to degrees

        # Append angle value to list
        angles.append(angle)

        # Update new angle and last_point values
        last_point[0] = current_point[0]
        last_point[1] = current_point[1]
        last_angle = angle

    # Scale angles between 0 to 360 rather than -180 to 180
    angles = np.array(angles)
    angles = (angles + 360) % 360

    # Convert to radians
    angles = angles * (np.pi/180)

    return angles

# called in batch_processing only
def compute_tuning_curve(pos_x: np.ndarray, pos_y: np.ndarray, pos_t: np.ndarray,
                         spiket: np.ndarray, smoothing: int) -> tuple:

    '''
        Compute a polar plot of the average directional firing of a neuron.

        Params:
            pos_x, pos_y, pos_t (np.ndarray):
                Arrays of x and y coordinates, and timestamps
            spiket (np.ndarray):
                Timestamps of when spike events occured
            smoothing (int):
                Smoothing factor for angle data

        Returns:
            tuple: tuned_data, spike_angles, ang_occ, bin_array
            --------
            tuned_data (np.ndarray):
                Tuning curve
            spike_angles (np.ndarray):
                Angles at which spike occured
            ang_occ (np.ndarray):
                Histogram of occupanices within bins of angles
            bin_array (np.ndarray):
                Bins of angles (360 split into 36 bins of width 10 degrees)
    '''

    # Compute head direction angles
    hd_angles = _get_head_direction(pos_x, pos_y)

    # Split angle range (0 to 360) into angle bins
    bin_array = np.linspace(0,2*np.pi,36)

    # Compute histogram of occupanices in each bin
    ang_occ = angular_occupancy(pos_t.flatten(), hd_angles.flatten(), bin_width=10)

    # Extract spike angles (i.e angles at which spikes occured)
    spike_angles = []
    for i in range(len(spiket)):
        index = np.abs(pos_t - spiket[i]).argmin()
        spike_angles.append(hd_angles[index])

    spike_angles = np.array(spike_angles)
    spike_angles = spike_angles.flatten()

    # Compute tuning curve and smooth
    tuned_data = tuning_curve(ang_occ[0], spike_angles, bin_width=10)
    tuned_data_masked = np.copy(tuned_data)
    bin_array = bin_array
    tuned_data[tuned_data == np.nan] = 0
    tuned_data = _smooth(tuned_data,smoothing)

    return tuned_data, spike_angles, ang_occ, bin_array
