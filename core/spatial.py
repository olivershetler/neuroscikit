class Location(Subject):
    def __init__(self, subject, limb, input_dict):
        self.subject = subject
        self.limb = limb
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
        if 'z' in input_dict:
            self.z = input_dict['z']
            assert len(self.z) == len(self.t)
        if 'theta' in input_dict:
            self.theta = input_dict['theta']
            assert len(self.theta) == len(self.t)
        if 'r' in input_dict:
            self.r = input_dict['r']
            assert len(self.r) == len(self.t)
        if 'unit' in input_dict:
            self.units = input_dict['unit']

