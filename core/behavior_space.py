class BehaviorSpace():
    @abstractmethod
    def __init__(self, *geometry_args):
        pass

class RectangularSpace2D(BehaviorSpace):
    def __init__(self, x_min, x_max, y_min, y_max):
        self.geometry = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}

class CircularSpace2D(BehaviorSpace):
    def __init__(self, radius):
        self.geometry = {'radius': radius}