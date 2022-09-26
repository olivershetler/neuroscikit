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
    def __init__(self, input_dict=None):
        self._input_dict = input_dict
        
        self.cells = []

        if self._input_dict != None:
            self.cells = self._read_input_dict()

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
    def __init__(self, input_dict: dict):
        self._input_dict = input_dict

        self.events, self.signal, self.session_metadata = self._read_input_dict()

    def _read_input_dict(self):
        events = None
        waveforms = None 

        if 'events' in self._input_dict:
            events = self._input_dict['events']
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

        return events, signal, session_metadata

