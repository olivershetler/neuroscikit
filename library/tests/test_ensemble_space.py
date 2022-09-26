import os, sys

# from prototypes.wave_form_sorter.sort_cell_spike_times import sort_cell_spike_times

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.ensemble_space import CellEnsemble, CellPopulation, Cell
from core.core_utils import *
from core.subjects import SessionMetadata

def test_cell():
    events = make_1D_timestamps()
    waveforms = make_waveforms(4, 100, 50)
    session_metadata = SessionMetadata({'session_id': 'id0'})
    cell = Cell({'event_times': events, 'signal': waveforms, 'session_metadata': session_metadata})

    assert isinstance(cell, Cell)
    assert type(cell.events) == list 
    assert waveforms == cell.signal

def test_cell_ensemble():
    cells = {}
    for i in range(5):
        events = make_1D_timestamps()
        waveforms = make_waveforms
        session_metadata = SessionMetadata({'session_id': 'id0'})
        cell = Cell({'events': events, 'signal': waveforms, 'session_metadata': session_metadata})
        cells['cell_'+ str(i+1)] = cell

    ensemble = CellEnsemble(cells)

    assert isinstance(ensemble, CellEnsemble)
    assert type(ensemble.cells) == list

    events = make_1D_timestamps()
    waveforms = make_waveforms
    session_metadata = SessionMetadata({'session_id': 'id0'})
    cell_new = Cell({'event_times': events, 'signal': waveforms, 'session_metadata': session_metadata})

    ensemble.add_cell(cell_new)

    assert ensemble.cells[-1] == cell_new

def test_cell_population():
    cells = {}
    for i in range(5):
        events = make_1D_timestamps()
        waveforms = make_waveforms
        session_metadata = SessionMetadata({'session_id': 'id0'})
        cell = Cell({'events': events, 'signal': waveforms, 'session_metadata': session_metadata})
        cells['cell_'+ str(i+1)] = cell

    ensemble = CellEnsemble(cells)

    population = CellPopulation()
    population.add_ensemble(ensemble)

    assert type(population.ensembles) == list
    assert len(population.ensembles) == 1

