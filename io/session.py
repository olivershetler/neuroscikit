from abc import ABC, abstractmethod

#TODO: add temperature and lighting to metadata

class Session(ABC):
    """This class is used to store the session information.
    It contains both metadata and data.
    """
    @abstractmethod
    def __init__(self, session_id, date, **kwargs):
        pass

class SessionMetaData():
    """This class is used to store the session metadata.
    """
    def __init__(self, session_id, date, Subject, Experimenter, **kwargs):
        pass

class AnimalSessionMetaData():
    """This class is used to store the session metadata.
    """
    def __init__(self, session_id, date, animal_id, experimenter, animal_name, animal_age, animal):
        self.session_id = session_id
        self.date = date
        self.experimenter = experimenter
        self.animal_name = animal_name
        self.animal_age = animal_age
        self.animal = animal
        self.animal_id = animal_id

class HumanSessionMetadata():
    """This class is used to store the session metadata.
    """
    def __init__(self, session_id, date, subject_id, subject_name, demographics):
        self.session_id = session_id
        self.date = date
        self.subject_id = subject_id
        self.subject_name = subject_name
        self.demographics = demographics

class Experimenter():
    """This class is used to store the experimenter information.
    The experimentor's name and ORCID id are stored in this class.
    If the experimentor does not have an ORCID id, the ORCID id is set to None.
    However, it is reccommended that the experimentor get an ORCID id.
    """
    def __init__(self, experimenter, orcid):
        pass

