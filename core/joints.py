import numpy as np

class HeadDirection2D():
    def __init__(self, theta:float, units):
        assert isinstance(theta, float)
        assert isinstance(units, Units) or units is None
        if units is not None:
            if units.theta == 'radians':
                assert min(theta) >= 0 and max(theta) <= 2*np.pi, 'theta must be between 0 and 2pi'
            if units.theta == 'degrees':
                assert min(theta) >= 0 and max(theta) <= 360, 'The Units object given theta must be between 0 and 360 degrees'
            else:
                raise ValueError('The Units object given theta must be either radians or degrees')
        self.theta = theta
        self.units = units
