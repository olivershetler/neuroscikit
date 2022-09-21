


class Workspace():
    def __init__(self):
        pass


class SessionContainer(Workspace):
    def __init__(self, input_dict: dict):
        self._input_dict = input_dict

        self.session_data, self.session_metadata = self._read_input_dict()


    def _read_input_dict(self):
        session_data = None
        session_metadata = None

        if 'session_data' in self._input_dict:
            session_data = self._input_dict['session_data']
        else:
            print('No session data provided')

        if 'session_metadata' in  self._input_dict:
            session_metadata = self._input_dict['session_metadata']
        else:
            print('Mo session metadata provided')

        return session_data, session_metadata

    def get_animal_metadata(self):
        if 'animal' in self.session_metadata:
            animal_metadata = self.session_metadata['animal']
            return animal_metadata
        else:
            print('No animal metadata found')

    def get_devices_metadata(self):
        device_dict = {}
        if 'devices' in self.session_metadata:
            for key in self.session_metadata['devices']:
                device_dict[key] = self.session_metadata['devices'][key]
        return device_dict

    def get_spike_data(self):
        spike_dict = {}
        if self.session_data != None:
            for key in self.session_data:
                if 'spike' in key:
                    spike_dict[key] = self.session_data[key]
        return spike_dict

    def get_position_data(self):
        pass


class StudyContainer(Workspace):
    def __init__(self, input_dict: dict):
        pass





class Session():
    """
    Top level session class, holds an instance of the animal in the session + the devices used for acquisition and associated data
    """
    def __init__(self, input_dict: dict):
        self._input_dict = input_dict

        # animal dictionary holdss animal metadata
        # device dict is a dictionary with all devices
        animal_dict, devices_dict = self._read_input_dict()

        self.animal = AnimalMetadata(animal_dict)

        self.devices = DevicesMetadata(devices_dict)


        # for device_dict in device_dicts:
        #     session_device = DevicesMetadata(device_dict)
        #     self.devices.append(session_device.device)

    def _read_input_dict(self):
        if 'animal' in self._input_dict:
            animal_dict = self._input_dict['animal']
        else:
            print('No animal metadata added to the session')

        if 'devices' in self._input_dict:
            devices_dict = self._input_dict['devices']
            # devices_dicts = []
            # implant_present = False
            # for key in self._input_dict['devices']:
            #     if 'implant' in key:
            #         implant_present = True
            #     devices_dicts.append(self._input_dict['devices'][key])
            # if implant_present == False:
            #     print('No neural data added to the sessions')
        else:
            print('No devices added to the session')

        return animal_dict, devices_dict


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


class DevicesMetadata():
    def __init__(self, input_dict: dict):
        self._input_dict = input_dict

        self.devices_dict = self._read_input_dict()

    def _read_input_dict(self):
        devices = {}
        for key in self._input_dict:
            device_dict = self._input_dict[key]

            if key == 'implant':
                implant = ImplantMetadata(device_dict)
                devices[key] = implant
            elif key == 'axona_led_tracker':
                tracker = TrackerMetadata(device_dict)
                devices[key] = tracker

            # ... continue with more device types

        return devices


class ImplantMetadata(DevicesMetadata):
    def __init__(self, input_dict: dict):
        self._input_dict = input_dict

        self.implant_id, self.implant_geometry, self.implant_type, self.implant_data, self.wire_length, self.wire_length_units, self.implant_units = self._read_input_dict()

    def _read_input_dict(self):
        implant_id = None
        implant_geometry = None
        implant_type = None
        implant_data = None
        wire_length = None
        wire_length_units = None
        implant_units = None

        if 'implant_id' in self._input_dict:
            implant_id = self._input_dict['implant_id']
        if 'implant_geometry' in self._input_dict:
            implant_geometry = self._input_dict['implant_geometry']
        if 'implant_type' in self._input_dict:
            implant_type = self._input_dict['implant_type']
        if 'implant_data' in self._input_dict:
            implant_data = self._input_dict['implant_data']
        if 'wire_length' in self._input_dict:
            wire_length = self._input_dict['wire_length']
        if 'wire_length_units' in self._input_dict:
            wire_length_units = self._input_dict['wire_length_units']
        if 'implant_units' in self._input_dict:
            implant_units = self._input_dict['implant_units']

        return implant_id, implant_geometry, implant_type, implant_data, wire_length, wire_length_units, implant_units



class TrackerMetadata(DevicesMetadata):
    def __init__(self, input_dict: dict):
        self._input_dict = input_dict

        self.led_tracker_id, self.led_location, self.led_position_data, self.x, self.y, self.time, self.arena_height, self.arena_width = self._read_input_dict()

    def _read_input_dict(self):
        led_tracker_id = None
        led_location = None
        led_position_data = None
        x = None
        y = None
        time = None
        arena_height = None
        arena_width = None

        if 'led_tracker_id' in self._input_dict:
            led_tracker_id = self._input_dict['led_tracker_id']
        if 'led_location' in self._input_dict:
            led_location = self._input_dict[led_location]
        if 'led_position_data' in self._input_dict:
            led_position_data = self._input_dict['led_position_data']
        if 'x' in self._input_dict:
            x = self._input_dict['x']
        if 'y' in self._input_dict:
            y = self._input_dict['y']
        if 'time' in self._input_dict:
            time = self._input_dict['time']
        if 'arena_height' in self._input_dict:
            arena_height = self._input_dict['arena_height']
        if 'arena_width' in self._input_dict:
            arena_width = self._input_dict['arena_width']

        return led_tracker_id, led_location, led_position_data, x, y, time, arena_height, arena_width



