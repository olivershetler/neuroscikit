from abc import ABC, abstractmethod

class Subject(ABC):
    """This class is used to store the subject information.
    """
    @abstractmethod
    def __init__(self, subject_id, **kwargs):
        pass

class AnimalSubject():
    """This class is used to store the animal subject information.
    """
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    def

class MouseSubject():
    """This class is used to store the mouse subject information.
    """
    def __init__(self, mouse_id, mouse_name, mouse_age):
        self.mouse_id = mouse_id
        self.mouse_name = mouse_name
        self.mouse_age = mouse_age

class HumanSubject(Subject):
    """This class is used to store the human subject information.
    """
    def __init__(self, subject_id: int, subject_name: tuple, demographics: dict):
        self.subject_id = subject_id
        self.subject_name = subject_name
        for key, value in demographics.items():
            setattr(self, key, value)