

class Workspace():
    def __init__(self):
        pass


class Session(Workspace):
    def __init__(self, input_dict: dict):
        self._input_dict = input_dict

        self.session_data, self.session_metadata = self._read_input_dict()


    def _read_input_dict(self):
        session_data = None
        session_metadata = None

        if 'data' in self._input_dict:
            session_data = self._input_dict['data']
        else:
            print('No session data provided')

        if 'metadata' in  self._input_dict:
            session_metadata = self._input_dict['metadata']
        else:
            print('Mo session metadata provided')

        return session_data, session_metadata

    def get_core_instances(self):
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


class Study(Workspace):
    def __init__(self, study_metadata, input_dict: dict):
        assert isinstance(study_metadata, StudyMetadata), 'The argument must be a Study object'
        self._input_dict = input_dict

        self.sessions = self._read_input_dict()

    def _read_input_dict(self):
        sessions = []
        for key in self._input_dict:
            assert isinstance(self._input_dict[key], Session), 'All arguments must be SessionWorkspace objects'
            sessions.append(self._input_dict[key])

        return sessions

    def add_session(self, session):
        assert isinstance(session, Session), 'The argument must be a SessionWorkspace object'
        self.sessions.append(session)

        

class SessionData():
    def __init__(self, input_dict: dict):
        self._input_dict = input_dict 

        self.data = self._read_input_dict()

    def _read_input_dict(self):
        core_data_instances = {} 
        
        for key in self._input_dict:
            core_data_instances[key] = self._input_dict[key]

        return core_data_instances

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