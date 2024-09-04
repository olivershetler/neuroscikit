from abc import ABC, abstractmethod

class Environment():
    def __init__():
        pass

class Arena(Environment):
    """This class is used to store the arena information.
    """
    def __init__(self, **kwargs):
        self.arena_id = arena_id
        self.kwargs = kwargs

class Arena2D(Environment):
    """This class is used to store the arena information.
    """
    def __init__(self, **kwargs):
        self.arena_id = arena_id

class RectangularArena2D(Arena2D):
    """This class is used to store the arena information.
    """
    def __init__(self, arena_id, x_width, y_height, unit, objects):
        self.arena_id = arena_id
        self.width = x_width
        self.height = y_height
        self.units = {'width': unit, 'height': unit}
        self.objects = objects

class DiskArena(Arena2D):
    """This class is used to store the arena information for disk arenas.
        It
    """
    def __init__(self, radius, unit, objects:dict):
        self.arena_id = arena_id
        self.radius = radius
        self.objects = objects

class VirtualArena1D(Arena):
    """This class is used to store the virtual arena information.
    """
    def __init__(self, **kwargs):
        self.arena_id = arena_id
        self.shape = shape
        self.kwargs = kwargs