class Position2D(Subject):
    def __init__(self, subject, space, input_dict):
        self.subject = subject
        self.limb = space
        if 't' in input_dict:
            self.t = input_dict['t']
        elif 'rate' in input_dict and 'x' in input_dict:
            self.t = np.arange(0, len(input_dict['x']) / input_dict['rate'], 1 / input_dict['rate'])
        if 'x' in input_dict:
            self.x = input_dict['x']
            assert len(self.x) == len(self.t)
        if 'y' in input_dict:
            self.y = input_dict['y']
            assert len(self.y) == len(self.t)

    def speed_from_locations(location: Location) -> np.ndarray:
        '''calculates an averaged/smoothed speed'''

        x = location.x
        y = location.y
        t = location.t

        N = len(x)
        v = np.zeros((N, 1))

        _speed_formula = lambda index, x, y, t: np.sqrt((x[index + 1] - x[index - 1]) ** 2 + (y[index + 1] - y[index - 1]) ** 2) / (t[index + 1] - t[index - 1])

        v = np.ndarray(map(_speed_formula, range(1, N-1), x, y, t))

        """
        for index in range(1, N-1):
            v[index] = np.sqrt((x[index + 1] - x[index - 1]) ** 2 + (y[index + 1] - y[index - 1]) ** 2) / (
            t[index + 1] - t[index - 1])
        """

        v[0] = v[1]
        v[-1] = v[-2]
        v = v.flatten()

        kernel_size = 12
        kernel = np.ones(kernel_size) / kernel_size
        v_convolved = np.convolve(v, kernel, mode='same')

        return v_convolved

    def filter_pos_by_speed(self, lower_speed: float, higher_speed: float, speed2d: np.ndarray, pos_x: np.ndarray, pos_y: np.ndarray, pos_t: np.ndarray) -> tuple:
        '''
            Selectively filters position values of subject travelling between
            specific speed limits.

            Params:
                lower_speed (float):
                    Lower speed bound (cm/s)
                higher_speed (float):
                    Higher speed bound (cm/s)
                pos_v (np.ndarray):
                    Array holding speed values of subject
                pos_x, pos_y, pos_t (np.ndarray):
                    X, Y coordinate tracking of subject and timestamps

            Returns:
                Tuple: new_pos_x, new_pos_y, new_pos_t
                --------
                new_pos_x, new_pos_y, new_pos_t (np.ndarray):
                    speed filtered x,y coordinates and timestamps
        '''

        # Initialize empty array that will only be populated with speed values within
        # specified bounds
        choose_array = []

        # Iterate and select speeds
        for index, element in enumerate(pos_v):
            if element > lower_speed and element < higher_speed:
                choose_array.append(index)

        # construct new x,y and t arrays
        new_pos_x = np.asarray([ float(pos_x[i]) for i in choose_array])
        new_pos_y = np.asarray([ float(pos_y[i]) for i in choose_array])
        new_pos_t = np.asarray([ float(pos_t[i]) for i in choose_array])

        return new_pos_x, new_pos_y, new_pos_t

