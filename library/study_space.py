from msilib.schema import Class
import os
import sys

from library.maps.spatial_spike_train import SpatialSpikeTrain2D

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.subjects import StudyMetadata, SessionMetadata
from library.ensemble_space import Cell, CellEnsemble, CellPopulation
from library.batch_space import SpikeClusterBatch, SpikeTrainBatch
from core.spikes import * 
from core.spatial import Position2D
from library.spike import sort_spikes_by_cell
from library.workspace import Workspace
from core.instruments import DevicesMetadata, ImplantMetadata, TrackerMetadata


class Session(Workspace):
    def __init__(self, input_dict={}):
        self._input_dict = input_dict

        self.session_data, self.session_metadata = self._read_input_dict()
        
        if not isinstance(self.session_metadata, SessionMetadata) == 0:
            self.session_metadata = SessionMetadata()
            device_metadata = DevicesMetadata(input_dict={}, session_metadata= self)
            self.session_metadata.metadata['devices'] = device_metadata
        
        self.animal_id = None
        self.set_animal_id()

    def get_animal_id(self):
        if self.animal_id == None:
            print('Need to add animal metadata to session')
        return self.animal_id

    def set_animal_id(self):
        if 'animal' in self.session_metadata.metadata: 
            self.animal_id = self.session_metadata.metadata['animal'].animal_id
            print('Animal ID set')

    def _read_input_dict(self):
        session_data = {}
        session_metadata = {}

        if 'data' in self._input_dict:
            session_data = self._input_dict['data']
        else:
            session_data = SessionData()

        if 'metadata' in  self._input_dict:
            session_metadata = self._input_dict['metadata']
        else:
            session_metadata = SessionMetadata()

        return session_data, session_metadata

    def get_object_instances(self):
        instances = {}
        instances ['animal_metadata'] = self.get_animal_metadata()
        device_metadata = self.get_devices_metadata()
        for key in device_metadata:
            instances[key] = device_metadata[key]
        spike_data = self.get_spike_data()
        for key in spike_data:
            instances[key] = spike_data[key]
        return instances

    def get_animal_metadata(self):
        if 'animal' in self.session_metadata.metadata:
            animal_metadata = self.session_metadata.metadata['animal']
            return animal_metadata
        else:
            print('No animal metadata found')

    def get_devices_metadata(self):
        device_dict = {}
        if 'devices' in self.session_metadata.metadata:
            for key in self.session_metadata.metadata['devices'].devices_dict:
                device_dict[key] = self.session_metadata.metadata['devices'].devices_dict[key]
        return device_dict

    def get_spike_data(self):
        spike_dict = {}
        if self.session_data.data != None:
            for key in self.session_data.data:
                if 'spike' in key:
                    spike_dict[key] = self.session_data.data[key]
        return spike_dict

    def get_position_data(self):
        pos_dict = {}
        if self.session_data.data != None:
            for key in self.session_data.data:
                if 'position' in key or 'pos' in key:
                    pos_dict[key] = self.session_data.data[key]
        return pos_dict

    def make_class(self, ClassName, input_dict: dict):
        class_object = ClassName(input_dict, session_metadata=self) 

        if 'Metadata' in str(ClassName) or 'metadata' in str(ClassName):
            if isinstance(class_object, TrackerMetadata):
                self.session_metadata.metadata['devices']._add_device('axona_led_tracker', class_object)
            elif isinstance(class_object, ImplantMetadata):
                self.session_metadata.metadata['devices']._add_device('implant', class_object)
        else:
            if isinstance(class_object, SpikeCluster):
                self.session_data.data['spike_cluster'] = class_object
            elif isinstance(class_object, SpikeTrain):
                self.session_data.data['spike_train'] = class_object
            elif isinstance(class_object, Position2D):
                self.session_data.data['position'] = class_object
            elif isinstance(class_object, SpatialSpikeTrain2D):
                self.session_data.data['spatial_spike_train'] = class_object

        return class_object


class Study(Workspace):
    def __init__(self, study_metadata, input_dict: dict):
        assert isinstance(study_metadata, StudyMetadata), 'The argument must be a Study object'
        self._input_dict = input_dict

        self.sessions = self._read_input_dict()

        self.animal_ids = self._get_animal_ids()

    def _get_animal_ids(self):
        animal_ids = []
        for session in self.sessions:
            animal_ids.append(session.animal_id)
        return animal_ids

    def _read_input_dict(self):
        sessions = []
        for key in self._input_dict:
            assert isinstance(self._input_dict[key], Session), 'All arguments must be Session(Workspace) objects'
            sessions.append(self._input_dict[key])

        return sessions

    def add_session(self, session):
        assert isinstance(session, Session), 'The argument must be a Session(Workspace) object'
        self.sessions.append(session)

    def _sort_session_by_animal(self):
        animal_sessions = {}

        for id in self.animal_ids:
            animal_sessions[id] = []

        for i in range(len(self.sessions)):
            animal_id = self.sessions[i].animal_id
            assert animal_id in self.animal_ids
            animal_sessions[animal_id].append(self.sessions[i])

            ### ....
            ### NEED TO EXTEND THIS TO MAKE animal_sessions[animal_id] a dictionary and not list of dictionaries
            ### Keys will be ordered/sequential sessions. Have to save start date/time from read_tetrode_cut and use to order
            ### ...

        return animal_sessions

    def make_animals(self):
        animal_sessions = self._sort_session_by_animal()
        animals = []

        for animal in animal_sessions:
            animal_instance = Animal(animal)
            animals.append(animal_instance)

        self.animals = animals

    def get_animals(self):
        if self.animals == None:
            self.make_animals()
        return self.animals
      

class SessionData():
    def __init__(self, input_dict={}):
        self._input_dict = input_dict 

        self.data = self._read_input_dict()

    def _read_input_dict(self):
        core_data_instances = {} 
        
        for key in self._input_dict:
            core_data_instances[key] = self._input_dict[key]

        return core_data_instances

    
    def _add_session_data(self, key, data_class):
        assert 'Metadata' not in key and 'metadata' not in key, 'Cannot add metadata class to session data'
        self.data[key] = data_class



class Animal(Workspace):
    """
    Holds all sessions belonging to an animal, TO BE ADDED: ordered sequentially in time 
    """
    ### Currently input is a list of dictionaries, once we save ordered sessions in x_io study class we will input nested dictionaries
    def __init__(self, input_dict: dict):
        self._input_dict = input_dict

        self.sessions = []
        self.ensembles = []

        self.sessions, self.ensembles = self._read_input_dict()

        self.population = CellPopulation()

        self.animal_id = self.sessions[0].animal_id


    def _read_input_dict(self):
        sessions = []
        ensembles = []
        for key in self._input_dict:
            session = self._input_dict[key]
            assert isinstance(session, Session)
            # session is instance of SessionWorkspace, has SessionData and SessionMetadata
            # AnimalSession will hold Cells which hold SpikeTrains from SessionData
            cell_ensemble = self.add_session(session)
            sessions.append(session)
            ensembles.append(cell_ensemble)
        return sessions, ensembles

    def _extract_core_classes(self, session):
        core_data = {}
        keys = list(session.session_data.data.keys())
        for i in range(len(keys)):
            core_type = session.session_data.data[keys[i]] 
            core_data = self._check_core_type(keys[i], core_type, core_data)

        return core_data

    def _check_core_type(self, key, core_type, core_data):
        if isinstance(core_type, SpikeTrain):
            core_data[key] = core_type
        elif isinstance(core_type, SpikeClusterBatch):
            core_data[key] = core_type
        elif isinstance(core_type, Position2D):
            core_data[key] = core_type
        else:
            print('Session data class is not valid, check inputs')
        return core_data

    def add_session(self, session):
        core_data = self._extract_core_classes(session)
        assert 'spike_cluster' in core_data, 'Need cluster label data to sort valid cells'
        spike_cluster = core_data['spike_cluster']
        assert isinstance(spike_cluster, SpikeClusterBatch)
        good_sorted_cells, good_sorted_waveforms = sort_spikes_by_cell(spike_cluster)
        print('Session data added, spikes sorted by cell')
        ensemble = CellEnsemble()
        for i in range(len(good_sorted_cells)):
            cell_dict = {'events': good_sorted_cells[i], 'signal': good_sorted_waveforms[i], 'session_metadata': session.session_metadata}
            cell = Cell(cell_dict)
            ensemble.add_cell(cell)
        
        self.ensembles.append(ensemble)
        self.sessions.append(session)
        

    
