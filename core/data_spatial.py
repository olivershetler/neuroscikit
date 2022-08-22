from abc import ABC, abstractmethod

class Environment():
    def

class Arena(Environment):
    """This class is used to store the arena information.
    """
    def __init__(self, **kwargs):
        self.arena_id = arena_id
        self.**kwargs = kwargs

class Arena2D(Environment):
    """This class is used to store the arena information.
    """
    def __init__(self, **kwargs):
        self.arena_id = arena_id

class DiskArena(Arena2D):
    """This class is used to store the arena information for disk arenas.
        It
    """
    def __init__(self, radius:dict, objects:dict):
        self.arena_id = arena_id
        self.radius = radius
        for obj in objects:
            self.objects[obj] = objects[obj]

class VirtualArena1D(Arena):
    """This class is used to store the virtual arena information.
    """
    def __init__(self, **kwargs):
        self.arena_id = arena_id
        self.shape = shape
        self.**kwargs = kwargs