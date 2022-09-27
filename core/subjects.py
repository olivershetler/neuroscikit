import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)



class AnimalMetadata():
    def __init__(self, input_dict: dict, **kwargs):
        self._input_dict = input_dict

        self.animal_id, self.species, self.sex, self.age, self.weight, self.genotype, self.animal_notes, self.session_metadata = self._read_input_dict()
        
        if 'session_metadata' in kwargs:
            if self.session_metadata != None: 
                print('Ses metadata is in the input dict and init fxn, init fnx will override')
            self.session_metadata = kwargs['session_metadata']


    def _read_input_dict(self):
        animal_id, species, sex, age, weight, genotype, animal_notes, session_metadata = None, None, None, None, None, None, None, None

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
        if 'session_metadata' in self._input_dict:
            session_metadata = self._input_dict['sessionn_metadata']

        return animal_id, species, sex, age, weight, genotype, animal_notes, session_metadata


class SessionMetadata():
    def __init__(self, input_dict: dict, **kwargs):
        self._input_dict = input_dict   

        self.session_object = None
        if 'session_object' in kwargs:
            self.session_object = kwargs['session_object']

        self.metadata = self._read_input_dict()

    def _read_input_dict(self):
        core_metadata_instances = {} 
        
        for key in self._input_dict:
            core_metadata_instances[key] = self._input_dict[key]

        return core_metadata_instances

    def _add_metadata(self, key, metadata_class):
        assert key == 'animal' or key == 'devices', 'Cann only add AnimalMetadata and DeviceMetadata objects to session'
        self.metadata[key] = metadata_class
        


class StudyMetadata():
    def __init__(self, input_dict: dict):
        self._input_dict = input_dict 
    
    def _read_input_dict(self):
        pass


    
