
import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from abc import ABC


# from core.core_utils import (
#     make_seconds_index_from_rate,
# )



class AnimalMetadata():
    def __init__(self, input_dict: dict):
        self._input_dict = input_dict

        self.animal_id, self.species, self.sex, self.age, self.weight, self.genotype, self.animal_notes = self._read_input_dict()


    def _read_input_dict(self):
        if 'animal_id' in self._input_dict:
            animal_id = self._input_dict['animal_id']
        if 'species' in self._input_dict:
            species = self._input_dict['species']
        if 'sex' in self._input_dict:
            sex = self._input_dict['sex']
        if 'age' in self._input_dict:
            age = self._input_dict['age']
        if 'weight' in self._input_dict:
            weight = self._input_dict['weight']
        if 'genotype' in self._input_dict:
            genotype = self._input_dict['genotype']
        if 'animal_notes' in self._input_dict:
            animal_notes = self._input_dict['animal_notes']

        return animal_id, species, sex, age, weight, genotype, animal_notes



class Animal():
    """
    Holds all sessions belonging to an animal, TO BE ADDED: ordered sequentially
    """
    ### Currently input is a list of dictionaries, once we save ordered sessions in x_io study class we will input nested dictionaries
    def __init__(self, input_dict_list: list):
        self._input_dict_list = input_dict_list

        self.sessions = self._read_input_dict()

    def _read_input_dict(self):
        sessions = []
        for i in range(len(self._input_dict_list)):
            session_dict = self._input_dict_list[i]
            session = AnimalSession(session_dict)
            sessions.append(session)
        return sessions


class AnimalSession():
    """
    A single session belonging to an animal instance.
    """
    def __init__(self, input_dict: dict):
        self._input_dict = input_dict

    def _read_input_dict(self):
        pass

class AnimalCell():
    """
    A single cell belonging to a session of an animal
    """
    def __init__(self, input_dict: dict):
        self._input_dict = input_dict

    def _read_input_dict(self):
        pass


"""

Animal has multiple AnimalSessions
AnimalSession has multiple AnimalCells
AnimalCells != SpikeTrainCluster
SpikeTrainCLuster has multiple SORTED event as SpikeTrains.
AnimalCells has multiple SpikeTrainClusters.
    If one trial per cell then AnimalCells has one SpikeTrain (as a SpikeTrainCluster with one SpikeTrain only)
    If multiple trials per cell then AnimallCells has multiple SpikeTrains (as a SpikeTrainCluster with as manny SpikeTrains as trials
AnimalSession != SpikeCluster
AnimalSession has multiple UNSORTED events as SpikeCluster
SpikeCluster has the UNSORTED events as Spikes

"""


