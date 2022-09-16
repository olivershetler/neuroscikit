from abc import ABC, abstractmethod

class Implant(ABC):
    """This class is used to store the implant information.
    """
    @abstractmethod
    def __init__(self, implant_id, **kwargs):
        pass

class MouseImplant(Implant, MouseSubject):
    """This class is used to store the mouse implant information.
    """
    def __init__(self, Implant, MouseSubject: MouseImplant, **kwargs):
        pass

class MouseTetrode(MouseImplant):
    """This class is used to store the mouse tetrode information.
    """
    def __init__(self, tetrode_id, **kwargs):
        self.tetrode_id = tetrode_id

class MouseShank(MouseImplant):
    """This class is used to store the mouse shank information.
    """
    def __init__(self, shank_id, **kwargs):
        self.shank_id = shank_id