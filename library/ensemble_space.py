import os, sys

# from prototypes.wave_form_sorter.sort_cell_spike_times import sort_cell_spike_times

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.spikes import *
from library.workspace import Workspace

class CellPopulation(Workspace): 
    def __init__(self, input_dict=None):
        self._input_dict = input_dict

        self.ensembles = []
        
        if self._input_dict != None:
            self.ensembles = self._read_input_dict()

    def _read_input_dict(self):
        ensembles = []
        for key in self._input_dict:
            ensemble = self._input_dict[key]
            assert isinstance(ensemble, CellEnsemble)
            ensembles.append(ensemble) 

        return ensembles

    def add_ensemble(self, ensemble):
        assert isinstance(ensemble, CellEnsemble)
        self.ensembles.append(ensemble)

class CellEnsemble(Workspace):
    """
    To manipulate groups of cells, flexible class, optional input is instances of Cell, can also add cells individually
    """
    def __init__(self, input_dict=None, **kwargs):
        self._input_dict = input_dict
        
        self.cells = []

        if self._input_dict != None:
            self.cells = self._read_input_dict()

        if 'sessison_metadata' in kwargs:
            self.session_metadata == kwargs['session_metadata']
            self.animal_id = self.session_metadata.metadata['animal'].animal_id
        else:
            self.session_metadata = None
            self.animal_id = None

    def _read_input_dict(self):
        cells = []
        for key in self._input_dict:
            cell = self._input_dict[key]
            assert isinstance(cell, Cell)
            cells.append(cell) 

        return cells

    def add_cell(self, cell):
        assert isinstance(cell, Cell)
        self.cells.append(cell)


class Cell(Workspace):
    """
    A single cell belonging to a session of an animal
    """
    def __init__(self, input_dict: dict, **kwargs):
        self._input_dict = input_dict

        self.event_times, self.signal, self.session_metadata = self._read_input_dict()

        self.stats_dict = self._init_stats_dict()
        # self.stats_dict = {}

        if 'sessison_metadata' in kwargs and self.session_metadata is None:
            self.session_metadata == kwargs['session_metadata']
        elif self.session_metadata is not None:
            self.animal_id = self.session_metadata.metadata['animal'].animal_id
            self.time_index = self.session_metadata.session_object.time_index
        else:
            self.animal_id = None
            self.session_metadata = None
            self.time_index = None
 

        

    def _read_input_dict(self):
        event_times = None
        waveforms = None 

        if 'event_times' in self._input_dict:
            event_times = self._input_dict['event_times']
        else:
            print('No event data provided to Cell')

        if 'signal' in self._input_dict:
            signal = self._input_dict['signal']
        else:
            print('No signal data provided')

        if 'session_metadata' in self._input_dict:
            session_metadata = self._input_dict['session_metadata']
        else:
            print('No session metadata provided, cannot effectively track cells')

        return event_times, signal, session_metadata

    def _init_stats_dict(self):
        stats_dict = {}
        path = 'library'
        dir_names = [x[1] for x in os.walk(path)][0]
        
        for dir in dir_names:
            if dir != 'tests' and 'cache' not in dir:
                stats_dict[dir] = {}

        return stats_dict

