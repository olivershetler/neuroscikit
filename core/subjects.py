
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


class SessionMetadata():
    def __init__(self, input_dict: dict):
        self._input_dict = input_dict 

        self.metadata = self._read_input_dict()

    def _read_input_dict(self):
        core_metadata_instances = {} 
        
        for key in self._input_dict:
            core_metadata_instances[key] = self._input_dict[key]

        return core_metadata_instances

class StudyMetadata():
    def __init__(self, input_dict: dict):
        self._input_dict = input_dict 
    
    def _read_input_dict(self):
        pass


    
